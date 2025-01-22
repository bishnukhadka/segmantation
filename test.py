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
from dataloaders.utils import decode_seg_map_sequence

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

# def dice_score(
#     y_true: torch.tensor, 
#     y_pred: torch.tensor, 
#     smooth=1):
#     """
#     Dice Score for Semantic Segmentation
#     : need to add for other type
#     """
#     y_pred = torch.argmax(y_pred, dim=1)
#     y_true_flatten = torch.flatten(y_true)
#     y_pred_flatten = torch.flatten(y_pred)
#     # print(f'y_true.shape(): {y_true.shape}, y_pred_shape: {y_pred.shape}')
#     assert y_true_flatten.shape == y_pred_flatten.shape, "y_true and y_pred needs to be of the same dimensions"
#     intersection = y_true_flatten * y_pred_flatten
#     intersection = intersection.sum()
#     return (2*intersection+smooth)/(y_true_flatten.sum()+y_pred_flatten.sum()+smooth)


def test(args):
        num_of_image_to_visualize = 2
        # Define Dataloader
        kwargs = {'num_workers': args.workers, 'pin_memory': True}
        dataset = args.dataset
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
        
        # assert that model_path is a valid path
        assert os.path.exists(args.weights), f"Path does not exist: {args.weights}"

        # loaded_model = model.load_state_dict(torch.load(args.weights,weights_only=True))
        loaded_model = torch.load(args.weights, weights_only=True)
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
        Acc = evaluator.Pixel_Accuracy()
        Acc_class = evaluator.Pixel_Accuracy_Class()
        mIoU = evaluator.Mean_Intersection_over_Union()
        FWIoU = evaluator.Frequency_Weighted_Intersection_over_Union()
        d_score = evaluator.get_dice_score()
        precision, recall = evaluator.get_precision_and_recall()

        print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(Acc, Acc_class, mIoU, FWIoU))
        print('Loss: %.3f' % test_loss)
        print(f'mIoU: {mIoU:.4f}')
        print(f'Dice Score: {d_score:.4f}')
        print(f'Precision: {precision:.4f},\nRecall: {recall:.4f}')
        print(f'\n\nNote: \tThe results form test.py is not saved at any file,\n\tmake sure to note the results')
        # inference on the first three images of test loader. 
        print(f'\n\nMaking inference on the first images of the test loader and plotting the result.\n')
        
        images_for_inference, labels_for_inference = next(iter(test_loader))['image'][:num_of_image_to_visualize], next(iter(test_loader))['label'][:num_of_image_to_visualize]
        
        evaluator.reset()
        with torch.no_grad():
            output = loaded_model(images_for_inference.cuda())
            if args.model=='fcn':
                output=output['out']
            if args.model =='deeplabv3':
                output=output['out']

            inference = decode_seg_map_sequence(torch.max(output[:num_of_image_to_visualize], 1)[1].detach().cpu().numpy(),
                                                    dataset=dataset)

            target = decode_seg_map_sequence(torch.squeeze(labels_for_inference[:num_of_image_to_visualize], 1).detach().cpu().numpy(),
                                                    dataset=dataset)
            
            # visualize_image2(images_for_inference, target, inference )
            visualize_image(images_for_inference, target, inference )

        # inference on the first three images of test loader. 
        print(f'\n\nMaking inference on the first image of the test loader and plotting the result.\n')
        pred = output.data.cpu().numpy()
        # pred = inference.cpu().numpy()
        target = labels_for_inference.cpu().numpy()
        # print(f'type of pred: {type(pred)}, shape: {pred.shape}')
        # print(target.shape)
        pred = np.argmax(pred, axis=1)
        # Add batch sample into evaluator
        evaluator.add_batch(target, pred)


        assert output != None, 'Error occured output is none'
        print(f'Evaluation metric on the given image.')
        Acc = evaluator.Pixel_Accuracy()
        Acc_class = evaluator.Pixel_Accuracy_Class()
        mIoU = evaluator.Mean_Intersection_over_Union()
        FWIoU = evaluator.Frequency_Weighted_Intersection_over_Union()
        d_score = evaluator.get_dice_score()
        precision, recall = evaluator.get_precision_and_recall()

        print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(Acc, Acc_class, mIoU, FWIoU))
        print('Loss: %.3f' % test_loss)
        print(f'mIoU: {mIoU:.4f}')
        print(f'Dice Score: {d_score:.4f}')
        print(f'Precision: {precision:.4f},\nRecall: {recall:.4f}')
        print(f'\n\nNote: \tThe results form test.py is not saved at any file,\n\tmake sure to note the results')
        

import matplotlib.pyplot as plt
def visualize_image(images, labels, predictions):
    """
    Visualizes input images, ground truth labels, and model predictions.
    """
    print(f'Image size: {images.size(0)}')
    batch_size = images.size(0)
    fig, axes = plt.subplots(batch_size, 3, figsize=(12, 4 * batch_size))

    for i in range(batch_size):
        # Input image
        images[i] = images[i] * torch.tensor([0.229, 0.224, 0.225], device=images.device).view(3, 1, 1) + \
            torch.tensor([0.485, 0.456, 0.406], device=images.device).view(3, 1, 1)
        axes[i, 0].imshow(images[i].permute(1, 2, 0).cpu().numpy())
        axes[i, 0].axis("off")
        axes[i, 0].set_title("Input Image")

        # Ground truth
        axes[i, 1].imshow(labels[i].permute(1, 2, 0).cpu().numpy())
        axes[i, 1].axis("off")
        axes[i, 1].set_title("Ground Truth")

        # Prediction
        axes[i, 2].imshow(predictions[i].permute(1, 2, 0).cpu().numpy())
        axes[i, 2].axis("off")
        axes[i, 2].set_title("Prediction")

    plt.tight_layout()
    plt.show()

# def visualize_image2(images, labels, predictions, alpha=0.5):
#     assert 0 <= alpha <= 1, "Alpha must be between 0 and 1."
#     assert images.shape[0] == predictions.shape[0], "Batch size of images and predictions must match."
#     assert images.dim() == 4 and images.size(1) == 3, "Images must have shape (B, C, H, W) with C=3 (RGB)."
    
#     batch_size = images.shape[0]
    
#     # Normalize images to [0, 1] if they aren't already
#     if images.max() > 1:
#         images = images / 255.0

#     for i in range(batch_size):
#         image = images[i].permute(1, 2, 0).cpu().numpy()  # Convert to HWC format for plotting
#         mask = predictions[i].cpu().numpy()
        
#         # Normalize the mask to [0, 1]
#         mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-5)

#         # Create the overlay
#         overlay = image.copy()
#         overlay[mask > 0.5] = [1, 0, 0]  # Red overlay for mask regions

#         blended = (1 - alpha) * image + alpha * overlay

#         # Plot the blended image
#         plt.figure(figsize=(6, 6))
#         plt.imshow(blended)
#         plt.axis('off')
#         plt.title(f"Image {i+1} with Segmentation Mask")
#         plt.show()


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
    # # finetuning pre-trained models
    # parser.add_argument('--pre-trained', '--pretrained', '--pre_trained', action='store_true', default=False)
    # parser.add_argument('--ft', action='store_true', default=False,
    #                     help='finetuning on a different dataset')
    # parser.add_argument('--freeze', type=str, default=None, choices=['encoder', 'decoder'],
    # help='choose encoder or decoder')

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
    torch.manual_seed(args.seed)
    test(args)
    
if __name__ == "__main__":
    main()
