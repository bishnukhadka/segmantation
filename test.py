import torch
import argparse
from tqdm import tqdm 
import numpy as np
from modeling.backbone import resnet, xception, drn, mobilenet
from torch.nn.modules.conv import Conv2d
from torch.nn.modules.batchnorm import BatchNorm2d
from torch.nn.modules.activation import ReLU
from torch.nn.modules.pooling import MaxPool2d, AdaptiveAvgPool2d
from torch.nn.modules.container import Sequential
from modeling.aspp import ASPP, _ASPPModule
from modeling.decoder import Decoder
from modeling.fcn import *
from modeling.deeplabv3 import *
from modeling.deeplab import *
from torchvision.models.segmentation.fcn import FCN, FCNHead
from torchvision.models.segmentation.deeplabv3 import DeepLabV3, DeepLabHead, ASPPConv, ASPPPooling
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.models.resnet import Bottleneck
from torchvision.models.segmentation.deeplabv3 import ASPP as DeepLabv3ASPP
from torch.nn.modules.container import ModuleList
from dataloaders import make_data_loader
import os

from util.loss import SegmentationLosses
from util.calculate_weights import calculate_weigths_labels
from util.lr_scheduler import LR_Scheduler
from util.saver import Saver
from util.summaries import TensorboardSummary
from util.metrics import Evaluator
from pathlib import Path as PathlibPath
from mypath import Path

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
    FCNResNet101, FCN, FCNHead, IntermediateLayerGetter, Bottleneck, # for FCN
    DeepLabV3, DeepLabHead, DeepLabV3Resnet101, DeepLabv3ASPP, ModuleList, ASPPConv, ASPPPooling
])

def test(args):
        # Define Dataloader
        kwargs = {'num_workers': args.workers, 'pin_memory': True}
        train_loader, val_loader, test_loader, nclass = make_data_loader(args, **kwargs)
        if args.model=='deeplabv3+':
            # for testing the model should be the best.pt
            model = DeepLab(num_classes=nclass,
                        backbone=args.backbone,
                        output_stride=args.out_stride,
                        sync_bn=args.sync_bn,
                        freeze_bn= args.freeze_bn) # freezes the BatchNorm layers, meaning they won’t update their running mean and variance during evaluation.
        elif args.model=='deeplabv3':
            model= DeepLabV3Resnet101(nclass, 256)
        else: 
            model = FCNResNet101(nclass, 512)
        
        #assert that model_path is a valid path
        assert os.path.exists(args.weights), f"Path does not exist: {args.weights}"

        loaded_model = model.load_state_dict(torch.load(args.weights, weights_only=True))
        # loaded_model = torch.load(args.weights, weights_only=True)
        loaded_model.eval()

        # whether to use class balanced weights
        if args.use_balanced_weights:
            classes_weights_path = os.path.join(Path.db_root_dir(args.dataset), args.dataset+'_classes_weights.npy')
            if os.path.isfile(classes_weights_path):
                weight = np.load(classes_weights_path)
            else:
                weight = calculate_weigths_labels(args.dataset, train_loader, nclass)
            weight = torch.from_numpy(weight.astype(np.float32))
        else:
            weight = None
        criterion = SegmentationLosses(
            weight=weight, 
            cuda=args.cuda).build_loss(mode=args.loss_type)

        
        evaluator = Evaluator(nclass)
        evaluator.reset()
        tbar = tqdm(test_loader, desc='\r')
        test_loss = 0.0
        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']
            if args.cuda:
                image, target = image.cuda(), target.cuda()
            with torch.no_grad():
                output = loaded_model(image)
                if args.model=='fcn':
                    output=output['out']
                if args.model =='deeplabv3':
                    output=output['out']
            loss = criterion(output, target)
            test_loss += loss.item()
            tbar.set_description('Test loss: %.3f' % (test_loss / (i + 1)))   
            pred = output.data.cpu().numpy()
            target = target.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            # Add batch sample into evaluator
            evaluator.add_batch(target, pred)

        assert output != None, 'Error occured output is none'
        # Fast test during the training 
        '''
        The above comment is here since, i copied from evaluation, 
        don't know why this is fast. 
        '''
        Acc = evaluator.Pixel_Accuracy()
        Acc_class = evaluator.Pixel_Accuracy_Class()
        mIoU = evaluator.Mean_Intersection_over_Union()
        FWIoU = evaluator.Frequency_Weighted_Intersection_over_Union()
        print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(Acc, Acc_class, mIoU, FWIoU))
        print('Loss: %.3f' % test_loss)

def main():
    parser = argparse.ArgumentParser(description="PyTorch Segmentation Training")
    parser.add_argument('--model', type=str, default='deeplabv3+',
                        choices=['deeplabv3','deeplabv3+','deeplabv3plus','fcn'])
    parser.add_argument('--backbone', type=str, default='resnet',
                        choices=['resnet', 'xception', 'drn', 'mobilenet'],
                        help='backbone name (default: resnet)')
    parser.add_argument('--weights', type=str, default=None,
                        help='path to the model weights.pt')
    parser.add_argument('--out_stride', type=int, default=16,
                        help='network output stride (default: 8)')
    """
    TODO: dataset arg should input the dataset path, not be hard-coded to certain datasets.  
    """
    parser.add_argument('--dataset', type=str, default='china_xrays_dataset',
                        choices=['bishnumati', 'bagmati', 'pascal', 'coco', 'cityscapes','darwinlungs','u4_dataset','u5_dataset','china_xrays_dataset','japan_xrays_dataset','montgomery_xrays_dataset','nih_xrays_dataset'],
                        help='dataset name (default: pascal)')
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
    parser.add_argument('--batch-size', type=int, default=None,
                        metavar='N', help='input batch size for \
                                training (default: auto)')
    parser.add_argument('--use-balanced-weights', action='store_true', default=False,
                        help='whether to use balanced weights (default: False)')
    # cuda, seed and logging
    parser.add_argument('--no-cuda', action='store_true', default=
                        False, help='disables CUDA training')
    parser.add_argument('--gpu-ids', type=str, default='0',
                        help='use which gpu to train, must be a \
                        comma-separated list of integers only (default=0)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    # finetuning pre-trained models
    parser.add_argument('--pre-trained', '--pretrained', '--pre_trained', action='store_true', default=False)
    parser.add_argument('--ft', action='store_true', default=False,
                        help='finetuning on a different dataset')
    parser.add_argument('--freeze', type=str, default=None, choices=['encoder', 'decoder'],
    help='choose encoder or decoder')

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

    if args.batch_size is None:
        args.batch_size = 4 * len(args.gpu_ids)
    
    print(args)
    torch.manual_seed(args.seed)
    test(args)
    

if __name__ == "__main__":
    main()