import argparse
import os
import numpy as np
from tqdm import tqdm

from mypath import Path
from dataloaders import make_data_loader
from modeling.sync_batchnorm.replicate import patch_replication_callback
from modeling.deeplab import *

from util.loss import SegmentationLosses
from util.calculate_weights import calculate_weigths_labels
from util.lr_scheduler import LR_Scheduler
from util.saver import Saver
from util.summaries import TensorboardSummary
from util.metrics import Evaluator
from pathlib import Path as PathlibPath

# added GLOBAL for best.pt model loading for testing
from modeling.backbone import resnet, xception, drn, mobilenet
from torch.nn.modules.conv import Conv2d
from torch.nn.modules.batchnorm import BatchNorm2d
from torch.nn.modules.activation import ReLU
from torch.nn.modules.pooling import MaxPool2d, AdaptiveAvgPool2d
from torch.nn.modules.container import Sequential
from modeling.aspp import ASPP, _ASPPModule
from modeling.decoder import Decoder
from modeling.fcn import *
from torchvision.models.segmentation.fcn import FCN, FCNHead
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.models.resnet import Bottleneck

# Allowlist the DeepLab class for safe loading
torch.serialization.add_safe_globals(
    [set, 
    DeepLab, 
    resnet.ResNet, 
    MaxPool2d, AdaptiveAvgPool2d,
    Conv2d, 
    BatchNorm2d, 
    ReLU,
    Sequential,
    resnet.Bottleneck,
    ASPP, _ASPPModule,
    Decoder,
    torch.nn.modules.dropout.Dropout,
    FCNResNet101, FCN, FCNHead, IntermediateLayerGetter, Bottleneck # for FCN
])
from torchinfo import summary

# added for using paths for weights and datasets
'''
has not been used yet
'''
import sys
FILE = PathlibPath(__file__).resolve()
ROOT = FILE.parents[0]  # root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = PathlibPath(os.path.relpath(ROOT, PathlibPath.cwd()))  # relative


class Trainer(object):
    def __init__(self, args):
        self.args = args

        # Define Saver
        self.saver = Saver(args)
        self.saver.save_experiment_config()
        # Define Tensorboard Summary
        self.summary = TensorboardSummary(self.saver.experiment_dir)
        self.writer = self.summary.create_summary()
        
        # Define Dataloader
        kwargs = {'num_workers': args.workers, 'pin_memory': True}
        self.train_loader, self.val_loader, self.test_loader, self.nclass = make_data_loader(args, **kwargs)

        if args.model=='deeplabv3+':
            # Define network
            model = DeepLab(num_classes=self.nclass,
                            backbone=args.backbone,
                            output_stride=args.out_stride,
                            sync_bn=args.sync_bn,
                            freeze_bn=args.freeze_bn)

            train_params = [{'params': model.get_1x_lr_params(), 
                            'lr': args.lr},
                            {'params': model.get_10x_lr_params(),
                            'lr': args.lr * 10}]   
        else: 
            # assert that the backbone is resnet
            assert args.backbone=='resnet', "FCN only supports resnet backbone(currently using ResNet101)"
            model = FCNResNet101(self.nclass, 512)

            train_params = [{'params': model.parameters(), 
                        'lr': args.lr}]
            
        # weights
        if self.args.weights:
            weights=PathlibPath(self.args.weights)
            # assert that weights is a valid path
            assert os.path.exists(weights)
            # print to inform of loading the weights
            print(f'Loading weights: model: {self.args.model}, backbone: {self.args.backbone}')
            try:
                # Attempt to load the model weights
                model = torch.load(weights, weights_only=True)
            except Exception as e:
                # If an exception occurs, print the error and raise an AssertionError or handle it as needed
                print(f"Error occurred while loading weights: {e}")
                raise AssertionError("Weights loading failed!") 
            print('Successfully loaded weights.\n')

        # freeze layers
        if self.args.freeze:
            if self.args.model == 'deeplabv3+' or self.args.model=='deeplabv3plus':
                assert self.args.model=='deeplabv3+', "model name should be either deeplabv3+ or fcn"
                if self.args.freeze=='encoder':
                    for param in model.backbone.parameters():
                        param.requires_grad=False

                    for param in model.aspp.parameters():
                        param.requires_grad=False
                else:
                    for param in model.decoder.parameters():
                        param.requires_grad=False
            elif self.args.model=='fcn':
                assert self.args.model=='fcn', "model name should be either deeplabv3+ or fcn"
                if self.args.freeze=='encoder':
                    for param in model.model.backbone.parameters():
                        param.requires_grad=False
                else:
                    for param in model.model.classifier.parameters():
                        param.requires_grad=False
                    for param in model.model.aux_classifier.parameters():
                        param.requires_grad=False

        print(summary(
            model=model, 
            input_size=(args.batch_size, 3, args.img, args.img), # make sure this is "input_size", not "input_shape"
            # col_names=["input_size"], # uncomment for smaller output
            col_names=["input_size", "output_size", "num_params", "trainable"],
            col_width=20,
            row_settings=["var_names"]
        ))
        
        # Define Optimizer
        optimizer = torch.optim.SGD(train_params,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay, 
                                    nesterov=args.nesterov)

        # Define Criterion
        # whether to use class balanced weights
        if args.use_balanced_weights:
            classes_weights_path = os.path.join(Path.db_root_dir(args.dataset), args.dataset+'_classes_weights.npy')
            if os.path.isfile(classes_weights_path):
                weight = np.load(classes_weights_path)
            else:
                weight = calculate_weigths_labels(args.dataset, self.train_loader, self.nclass)
            weight = torch.from_numpy(weight.astype(np.float32))
        else:
            weight = None
        self.criterion = SegmentationLosses(
            weight=weight, 
            cuda=args.cuda).build_loss(mode=args.loss_type)
        self.model, self.optimizer = model, optimizer
        
        # Define Evaluator
        self.evaluator = Evaluator(self.nclass)
        # Define lr scheduler
        self.scheduler = LR_Scheduler(
            args.lr_scheduler, 
            args.lr,
            args.epochs, 
            len(self.train_loader))

        # Using cuda
        if args.cuda:
            self.model = torch.nn.DataParallel(
                self.model, 
                device_ids=self.args.gpu_ids)
            patch_replication_callback(self.model)
            self.model = self.model.cuda()

        # Resuming checkpoint
        '''
        TODO: resume feature for pre-trained .pt model
        '''
        self.best_pred = 0.0
        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'" .format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            if args.cuda:
                self.model.module.load_state_dict(checkpoint['state_dict'])
            else:
                self.model.load_state_dict(checkpoint['state_dict'])
            if not args.ft:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.best_pred = checkpoint['best_pred']
            print("=> loaded checkpoint '{}' (epoch {})"
                .format(args.resume, checkpoint['epoch']))

        # Clear start epoch if fine-tuning
        if args.ft:
            args.start_epoch = 0

        # for testing on test-set, loading best.pt
        self.best_model=None

    def training(self, epoch):
        train_loss = 0.0
        self.model.train()
        tbar = tqdm(self.train_loader)
        num_img_tr = len(self.train_loader)
        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']

            # print(f"inside train function: change the shape of the target tensor")
            # target = torch.unsqueeze(target, dim=1)

            # print(f"inside train() funciton: image and target size::")
            # print(image.size(),target.size())

            if self.args.cuda:
                image, target = image.cuda(), target.cuda()

            self.scheduler(self.optimizer, i, epoch, self.best_pred)
            self.optimizer.zero_grad()
            output = self.model(image)
            
            """
            Output of the FCN comes out as 
            """
            if self.args.model=='fcn':
                output = output['out']

            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            tbar.set_description('Train loss: %.3f' % (train_loss / (i + 1)))
            self.writer.add_scalar('train/total_loss_iter', loss.item(), i + num_img_tr * epoch)
            
            """
            TODO: need to find the problem for some value of batchsize
            """
            # Show 10 * 3 inference results each epoch
            # if i % (num_img_tr // 10) == 0:
            #     global_step = i + num_img_tr * epoch
            #     self.summary.visualize_image(self.writer, self.args.dataset, image, target, output, global_step)

            
        self.writer.add_scalar('train/total_loss_epoch', train_loss, epoch)
        # print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
        # print('Loss: %.3f' % train_loss)

        if self.args.no_val:
            # save checkpoint every epoch
            is_best = False
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.module.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
            }, is_best)

        return train_loss

            
    def validation(self, epoch):
        self.model.eval()
        self.evaluator.reset()
        tbar = tqdm(self.val_loader, desc='\r')
        test_loss = 0.0
        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
            with torch.no_grad():
                output = self.model(image)
                if self.args.model=='fcn':
                    output=output['out']
            loss = self.criterion(output, target)
            test_loss += loss.item()
            tbar.set_description('Validation loss: %.3f' % (test_loss / (i + 1)))   # need to understand why this is added
            pred = output.data.cpu().numpy()
            target = target.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            # Add batch sample into evaluator
            self.evaluator.add_batch(target, pred)

        # Fast test during the training
        Acc = self.evaluator.Pixel_Accuracy()
        Acc_class = self.evaluator.Pixel_Accuracy_Class()
        mIoU = self.evaluator.Mean_Intersection_over_Union()
        FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()
        self.writer.add_scalar('val/total_loss_epoch', test_loss, epoch)
        self.writer.add_scalar('val/mIoU', mIoU, epoch)
        self.writer.add_scalar('val/Acc', Acc, epoch)
        self.writer.add_scalar('val/Acc_class', Acc_class, epoch)
        self.writer.add_scalar('val/fwIoU', FWIoU, epoch)
        print('Validation:')
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
        print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(Acc, Acc_class, mIoU, FWIoU))
        print('Loss: %.3f' % test_loss)

        new_pred = mIoU
        if new_pred > self.best_pred:
            print(f'Score improved: {self.best_pred} --> {new_pred}')
            is_best = True
            self.best_pred = new_pred
            self.best_model = self.model
            self.saver.best_model = self.best_model
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.module.state_dict(), # for training with parallel GPU's
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
            }, is_best)
    
    def test(self, model_path=None):
        if self.args.model=='deeplabv3+':
            # for testing the model should be the best.pt
            model = DeepLab(num_classes=self.nclass,
                        backbone=self.args.backbone,
                        output_stride=self.args.out_stride,
                        sync_bn=self.args.sync_bn,
                        freeze_bn=self.args.freeze_bn) # freezes the BatchNorm layers, meaning they won’t update their running mean and variance during evaluation.
        else: 
            model = FCNResNet101(self.nclass, 512)

        if not model_path: #if weights is given
            # print(self.saver.best_model_path)
            model_path = PathlibPath(self.saver.best_model_path)
            model = torch.load(self.saver.best_model_path, weights_only=True)
            if torch.cuda.device_count() > 1:
                model = torch.nn.DataParallel(model)
        else:
            # model.load_state_dict(torch.load(model_path, weights_only=True))
            torch.load(model_path, weights_only=True)
        model.eval()
        evaluator = Evaluator(self.nclass)
        evaluator.reset()
        tbar = tqdm(self.test_loader, desc='\r')
        test_loss = 0.0
        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
            with torch.no_grad():
                output = model(image)
                if self.args.model=='fcn':
                    output=output['out']
            loss = self.criterion(output, target)
            test_loss += loss.item()
            tbar.set_description('Test loss: %.3f' % (test_loss / (i + 1)))   
            pred = output.data.cpu().numpy()
            target = target.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            # Add batch sample into evaluator
            evaluator.add_batch(target, pred)

        # Fast test during the training 
        '''
        The above comment is here since, i copied from evaluation, 
        don't know why this is fast. 
        '''
        Acc = evaluator.Pixel_Accuracy()
        Acc_class = evaluator.Pixel_Accuracy_Class()
        mIoU = evaluator.Mean_Intersection_over_Union()
        FWIoU = evaluator.Frequency_Weighted_Intersection_over_Union()
        self.writer.add_scalar('test/total_loss_epoch', test_loss)
        self.writer.add_scalar('test/mIoU', mIoU)
        self.writer.add_scalar('test/Acc', Acc)
        self.writer.add_scalar('test/Acc_class', Acc_class)
        self.writer.add_scalar('test/fwIoU', FWIoU)
        print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(Acc, Acc_class, mIoU, FWIoU))
        print('Loss: %.3f' % test_loss)


def main():
    parser = argparse.ArgumentParser(description="PyTorch DeeplabV3Plus Training")
    parser.add_argument('--model', type=str, default='deeplabv3+',
                        choices=['deeplabv3+','deeplabv3plus','fcn'])
    parser.add_argument('--backbone', type=str, default='resnet',
                        choices=['resnet', 'xception', 'drn', 'mobilenet'],
                        help='backbone name (default: resnet)')
    parser.add_argument('--out_stride', type=int, default=16,
                        help='network output stride (default: 8)')
    """
    TODO: dataset arg should input the dataset path, not be hard-coded to certain datasets.  
    """
    parser.add_argument('--dataset', type=str, default='china_xrays_dataset',
                        choices=['bishnumati', 'bagmati', 'pascal', 'coco', 'cityscapes','darwinlungs','u4_dataset','u5_dataset','china_xrays_dataset','japan_xrays_dataset','montgomery_xrays_dataset','nih_xrays_dataset'],
                        help='dataset name (default: pascal)')
    parser.add_argument('--use-sbd', action='store_true', default=False,
                        help='whether to use SBD dataset (default: True)')
    parser.add_argument('--workers', type=int, default=4,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--img', '--imgz', '--img-size', type=int, default=256,
                        help='base image size')
    parser.add_argument('--crop_size', type=int, default=256,
                        help='crop image size')
    parser.add_argument('--sync-bn', type=bool, default=None,
                        help='whether to use SynchronizedBatchNorm2d or BatchNorm2d (default: auto)')
    parser.add_argument('--freeze-bn', type=bool, default=False,
                        help='whether to freeze bn parameters (default: False)')
    parser.add_argument('--loss-type', type=str, default='ce',
                        choices=['ce', 'focal'],
                        help='loss func type (default: ce)')
    # training hyper params
    parser.add_argument('--epochs', type=int, default=None, metavar='N',
                        help='number of epochs to train (default: auto)')
    parser.add_argument('--start_epoch', type=int, default=0,
                        metavar='N', help='start epochs (default:0)')
    parser.add_argument('--batch-size', type=int, default=None,
                        metavar='N', help='input batch size for \
                                training (default: auto)')
    parser.add_argument('--test-batch-size', type=int, default=None,
                        metavar='N', help='input batch size for \
                                testing (default: auto)')
    parser.add_argument('--use-balanced-weights', action='store_true', default=False,
                        help='whether to use balanced weights (default: False)')
    # optimizer params
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (default: auto)')
    parser.add_argument('--lr-scheduler', type=str, default='poly',
                        choices=['poly', 'step', 'cos'],
                        help='lr scheduler mode: (default: poly)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        metavar='M', help='momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        metavar='M', help='w-decay (default: 5e-4)')
    parser.add_argument('--nesterov', action='store_true', default=False,
                        help='whether use nesterov (default: False)')
    # cuda, seed and logging
    parser.add_argument('--no-cuda', action='store_true', default=
                        False, help='disables CUDA training')
    parser.add_argument('--gpu-ids', type=str, default='0',
                        help='use which gpu to train, must be a \
                        comma-separated list of integers only (default=0)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    # checking point
    parser.add_argument('--resume', type=str, default=None,
                        help='put the path to resuming file if needed')
    parser.add_argument('--checkname', type=str, default=None,
                        help='set the checkpoint name')
    # finetuning pre-trained models
    """
    Don't know what other reason was this used for so not using this right now
    """
    parser.add_argument('--ft', action='store_true', default=False,
                        help='finetuning on a different dataset')
    parser.add_argument('--weights', type=str, default=None,
                        help='path to the model weights.pt')
    parser.add_argument('--freeze', type=str, default=None, choices=['encoder', 'decoder'],
    help='choose encode or decoder')
    # evaluation option
    parser.add_argument('--eval-interval', type=int, default=1,
                        help='evaluuation interval (default: 1)')
    parser.add_argument('--no-val', action='store_true', default=False,
                        help='skip validation during training')
    # experiment detail
    parser.add_argument("--project", default='train',help="save to project/experiment_{}")

    args = parser.parse_args()
    print(args)
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')

    if args.sync_bn is None:
        if args.cuda and len(args.gpu_ids) > 1:
            args.sync_bn = True
        else:
            args.sync_bn = False

    # default settings for epochs, batch_size and lr
    if args.epochs is None:
        epoches = {
            'bishnumati':100,
            'bagmati':100,
            'coco': 30,
            'cityscapes': 200,
            'pascal': 50,
            'darwinlungs' : 100,
            'u4_dataset' : 100,
            'u5_dataset':100,
            'china_xrays_dataset':100,
            'japan_xrays_dataset':100,
            'montgomery_xrays_dataset':100,
            'nih_xrays_dataset':100,

        }
        args.epochs = epoches[args.dataset.lower()]

    if args.batch_size is None:
        args.batch_size = 4 * len(args.gpu_ids)

    if args.test_batch_size is None:
        args.test_batch_size = args.batch_size

    if args.lr is None:
        lrs = {
            'bishnumati':0.01, 
            'bagmati': 0.01,
            'coco': 0.1,
            'cityscapes': 0.01,
            'pascal': 0.007,
            # learning rate 
            'darwinlungs' : 0.01,
            'u4_dataset' : 0.01,
            'u5_dataset':0.01,
            'china_xrays_dataset':0.01,
            'japan_xrays_dataset':0.01,
            'montgomery_xrays_dataset':0.01,
            'nih_xrays_dataset':0.01,
        }
        args.lr = lrs[args.dataset.lower()] / (4 * len(args.gpu_ids)) * args.batch_size


    if args.checkname is None:
        args.checkname = 'deeplab-'+str(args.backbone)
    print(args)
    torch.manual_seed(args.seed)
    trainer = Trainer(args)
    print('Starting Epoch:', trainer.args.start_epoch)
    print('Total Epoches:', trainer.args.epochs)
    for epoch in range(trainer.args.start_epoch, trainer.args.epochs):
        print('\n\nTraining:')
        train_loss = trainer.training(epoch)
        print('Train Loss: %.3f' % train_loss)
        if not trainer.args.no_val and epoch % args.eval_interval == (args.eval_interval - 1):
            trainer.validation(epoch)
    
    # test-set evaluation
    print('\n\nTesting:')
    trainer.test()

    trainer.writer.close()

if __name__ == "__main__":
    main()