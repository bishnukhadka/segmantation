import argparse
from tkinter import image_names 
import torch
import numpy as np
import os
import torchvision
import random
from modeling.deeplab import *
import cv2 as cv
from dataloaders import make_data_loader
from tqdm import tqdm    
from PIL import Image
from modeling.sync_batchnorm.replicate import patch_replication_callback
from torchvision import transforms
from PIL import Image, ImageOps, ImageFilter
from scipy.special import softmax
from dataloaders import custom_transforms as tr


# # Added by Bishnu 
from torchvision.models.segmentation.deeplabv3 import DeepLabV3, DeepLabHead, ASPPConv, ASPPPooling
from modeling.fcn import *
from modeling.deeplabv3 import *
from modeling.deeplab import *
from modeling.backbone import resnet
from torch.nn.modules.pooling import MaxPool2d, AdaptiveAvgPool2d
from torchvision.models.segmentation.deeplabv3 import ASPP as DeepLabv3ASPP
from torchvision.models.segmentation.deeplabv3 import DeepLabV3, DeepLabHead, ASPPConv, ASPPPooling
from torchvision.models.segmentation.fcn import FCN, FCNHead
from torch.nn.modules.conv import Conv2d
from torch.nn.modules.batchnorm import BatchNorm2d
from torch.nn.modules.container import Sequential
from modeling.aspp import ASPP, _ASPPModule
from modeling.decoder import Decoder
from torch.nn.modules.activation import ReLU
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.models.resnet import Bottleneck
from torch.nn.modules.container import ModuleList
from util.calculate_weights import calculate_weigths_labels
from util.loss import SegmentationLosses
from mypath import Path
from pathlib import Path as PathLib

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

def save_output_images(weights, model, test_loader, SAVE_PATH_MODEL_OUTPUT,gt_paths,dataset_name):
    '''Saves the prediction image from segmentation model in local disk. 
        Supports only batch size 1 now.
    Args:
        weights: Path where model weights are saved
        test_loader: dataloader
        SAVE_PATH_MODEL_OUTPUT: save path for segmentation outputs
    '''
    
    model = torch.nn.DataParallel(model, device_ids=[0])
    patch_replication_callback(model)
    model = model.cuda()

    model = torch.load(weights)

    # start_epoch = checkpoint['epoch']
    
    # model.module.load_state_dict(checkpoint['state_dict'])


    model.eval()

    tbar = tqdm(test_loader, desc='\r')

    for i, sample in enumerate(tbar):

            image = sample['image'] 
            image_names = sample['image_name']  
            # print('sss',image_names[0],len(image_names))
            image = image.cuda()

            with torch.no_grad():
                output = model(image)
                output = output.data.cpu().numpy()  
            pred = output[0,:,:,:]
            pred = softmax(pred, axis=0)
            pred = np.argmax(pred, axis= 0)

            # pred = Image.fromarray((pred))
            name ,_,_ = image_names[0].partition('.')

# UPDATED BY UKESH
            if dataset_name == 'japan_xrays_dataset' or dataset_name == 'montgomery_xrays_dataset': 
                img = cv.imread(f"{gt_paths}\\masks\\{name}.png")
            else:
                img = cv.imread(f"{gt_paths}\\masks\\{name}_mask.png")
            h,w,c = img.shape 

            # pred.save()
            cv.imwrite(SAVE_PATH_MODEL_OUTPUT+'\\' + name+'.png' , pred, [cv.IMWRITE_PNG_BILEVEL, 1])
            
            preds = cv.imread(f"{SAVE_PATH_MODEL_OUTPUT}\\{name}.png") 
            resized = cv.resize(preds, (w, h), 0, 0, interpolation = cv.INTER_NEAREST)
            gray_image = cv.cvtColor(resized, cv.COLOR_BGR2GRAY)
            cv.imwrite(SAVE_PATH_MODEL_OUTPUT+'\\' + name+'.png' ,gray_image,[cv.IMWRITE_PNG_BILEVEL, 1])
            # print((SAVE_PATH_MODEL_OUTPUT +'\\'  


from dataloaders import custom_transforms as tr
from torchvision import transforms as T

from PIL import Image
# Preprocess the image
def preprocess_image(image_path, input_size=(256, 256)):
    """
    Prepares the image for inference.
    - image_path: Path to the image.
    - input_size: Expected input size for the model (H, W).
    """
    composed_transforms = T.Compose([
        T.Resize(input_size),  # Use torchvision's Resize
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    image = Image.open(image_path).convert("RGB")
    return composed_transforms(image).unsqueeze(0)  # Add batch dimension

def load_mask(mask_path):
    """
    Load a binary mask image from a given path, ensuring it is binary (0 and 1 values).

    Parameters:
    - mask_path (str): Path to the mask image file (single-channel).

    Returns:
    - mask (ndarray): Binary mask (values of 0 and 1).
    """
    mask = cv.imread(mask_path, cv.IMREAD_GRAYSCALE)  # Read in grayscale
    if mask is None:
        raise FileNotFoundError(f"Mask not found at path: {mask_path}")
    
    # Ensure mask is binary (0 and 1)
    mask = (mask > 0).astype(np.uint8)
    
    return mask

def load_image(image_path):
    """
    Load an image from a given path and convert it to RGB format.

    Parameters:
    - image_path (str): Path to the image file.

    Returns:
    - image (ndarray): Loaded image in RGB format.
    """
    image = cv.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")
    
    # Convert the image from BGR to RGB
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    return image

import matplotlib.pyplot as plt

def overlay_mask(image, mask, color=(175, 175, 0), alpha=0.5):
    """
    Overlays a binary segmentation mask on the image with transparency.

    Parameters:
    - image (ndarray): Input image in RGB format, shape (H, W, 3).
    - mask (ndarray): Binary mask (0 and 1 values), shape (H, W).
    - color (tuple): RGB color for the overlay (default red: (255, 0, 0)).
    - alpha (float): Transparency factor (0 = fully transparent, 1 = fully opaque).

    Returns:
    - overlayed_image (ndarray): Image with mask overlay.
    """
    # Ensure mask is binary (0 and 1)
    assert np.array_equal(np.unique(mask), [0, 1]), "Mask must only contain values 0 and 1."
    
    # Create a copy of the original image
    overlayed_image = image.copy()

    # Create a colored overlay for the mask regions
    mask_rgb = np.zeros_like(image, dtype=np.uint8)
    mask_rgb[mask == 1] = color  # Set the color for mask regions

    # Apply blending only to regions where mask == 1
    overlayed_image[mask == 1] = cv.addWeighted(
        image[mask == 1], 1 - alpha, mask_rgb[mask == 1], alpha, 0
    )
    
    return overlayed_image

def overlay_target_and_prediction_mask(
        image, 
        target, 
        prediction,    
        prediction_color = (233,150,122),   
        target_color = (173, 216, 229), 
        intersection_color = (255,255,0),
        alpha=0.5):
    """
    Overlays a binary segmentation mask on the image with transparency.

    Parameters:
    - target (ndarray): Input image in RGB format, shape (H, W).
    - prediction (ndarray): Binary mask (0 and 1 values), shape (H, W).
    - prediction_color (tuple): RGB color for the overlay.
    - target_color (tuple): RGB color for the overlay.
    - alpha (float): Transparency factor (0 = fully transparent, 1 = fully opaque).

    Returns:
    - overlayed_image (ndarray): Image with mask overlay.
    """
    # Ensure mask is binary (0 and 1)
    assert np.array_equal(np.unique(prediction), [0, 1]), "Mask must only contain values 0 and 1."
    assert len(prediction.shape) == 2 
    assert len(target.shape) == 2

    prediction_rgb = np.zeros((prediction.shape[0], prediction.shape[1], 3), dtype=np.uint8)
    print(f'Prection size: {prediction.shape}, prediction_rgb shape: {prediction_rgb.shape}')
    prediction_rgb[prediction == 1] = np.array(prediction_color, dtype=np.uint8)
    print(f'prediction rgb shape: {prediction_rgb.shape}')

    # Create a colored overlay for the mask regions
    target_rgb =np.zeros((target.shape[0], target.shape[1], 3), dtype=np.uint8)
    target_rgb[target == 1] = np.array(target_color, dtype=np.uint8)  # Set the color for mask regions
    print(f'target shape: {target.shape}, {np.unique(target)}')

    # Handle the intersection (where both target and prediction are 1)
    intersection = np.logical_and(target == 1, prediction == 1)
    print(f'intersection shape: {intersection.shape}, {type(intersection)}, {intersection[0][0]}')

    # prediction_rgb[target.astype(bool)] = np.array(target_color, dtype=np.uint8)
    # prediction_rgb[intersection] = np.array(intersection_color, dtype=np.uint8)

## giving color to the image itself. 
    combined_rgb = prediction_rgb.copy()


## overlay using cv2.addWeighted
    # Create a copy of the original image to overlay the masks
    combined_rgb = target_rgb.copy()
    # Apply target mask (blending with transparency)
    combined_rgb = cv.addWeighted(combined_rgb, 1 - alpha, target_rgb, alpha, 0)
    # Apply prediction mask (blending with transparency)
    combined_rgb = cv.addWeighted(combined_rgb, 1 - alpha, prediction_rgb, alpha, 0)
    
    return combined_rgb

from matplotlib.colors import ListedColormap

def plot_inference(image, target, prediction):
    """
    Display the image with the overlayed binary segmentation mask.

    Parameters:
    - image (ndarray): Input image in RGB format.
    - target (ndarray): Target mask (Ground Truth).
    - prediction (ndarray): 
    """
    target_color = [(0,0,0), (1,1,1)]
    prediction_color = [(0,0,0), (0,0.5,0)]
    # Overlay the mask on the image
    # overlayed_image_with_prediction = overlay_mask(image, prediction)

    overlay_target_and_prediction = overlay_target_and_prediction_mask(image, 
                                                                       target, 
                                                                       prediction,
                                                                       target_color=(255,255,255),
                                                                       prediction_color=(0,255,0),
                                                                       alpha=0.4
                                                                        )
    
    # Create a 1x4 grid of subplots
    # fig, axes = plt.subplots(1, 4, figsize=(16, 4), dpi=500)  # 1 row, 4 columns
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=500)  # 1 row, 4 columns

    # Plot data on each of the subplots
    axes[0].imshow(image)
    # axes[0].set_title("Image")
    axes[0].axis('off')  # Hide the axis
    
    axes[1].imshow(target,  cmap=ListedColormap(colors = target_color ))
    # axes[1].set_title("Target")
    axes[1].axis('off')

    # axes[2].imshow(prediction,cmap=ListedColormap(prediction_color))
    axes[2].imshow(overlay_mask(image, prediction, color=(0,175,0)))
    # axes[2].set_title("Prediction")
    axes[2].axis('off')

    # axes[3].imshow(overlay_target_and_prediction)
    # # axes[3].set_title("Overlay")
    # axes[3].axis('off')


    plt.tight_layout(pad=2)
    plt.show()

if __name__ == '__main__':
    # list_model = ['china_xrays_dataset']
    # list_dataset = ['montgomery_xrays_dataset']
    
    # for name in list_model:
    #     print(f'Loading Dataset model --> {name}')
        
    #     LOAD_PATH_MODEL = f'E:\\projects\\X-ray\\Deeplab-Xception-Lungs-Segmentation-master\\run\\{name}\\deeplab-resnet\\model_best.pth.tar' 
                
        # for datasets in list_dataset:
        #     print(f'Testing on Dataset --> {datasets}')
        #     parser = argparse.ArgumentParser(description="Inference For Best Model")
        #     parser.add_argument('--dataset', type=str, default=datasets, choices=['bagmati', 'bishnumati','pascal', 'camus','darwinlungs','china_xrays_dataset','japan_xrays_dataset','montgomery_xrays_dataset','nih_xrays_dataset','u4_dataset','u5_dataset','NIH_dataset','cheXpert_dataset'])
        #     parser.add_argument('--batch_size', type=int, default=1)
        #     parser.add_argument('--no-cuda', action='store_true', default=
        #                         False, help='disables CUDA training')
        #     parser.add_argument('--gpu-ids', type=str, default='0',
        #                         help='use which gpu to train, must be a \
        #                         comma-separated list of integers only (default=0)')
        #     # # added by uk
        #     parser.add_argument('--size', type=int, default=256,
        #                         help='image size')
            
        #     # # added by bishnu 
        #     parser.add_argument('--model', type=str, default='deeplabv3+',
        #                 choices=['deeplabv3','deeplabv3+','deeplabv3plus','fcn'])
        #     parser.add_argument('--backbone', type=str, default='resnet',
        #                         choices=['resnet', 'xception', 'drn', 'mobilenet'],
        #                         help='backbone name (default: resnet)')
        #     parser.add_argument('--weights', type=str, default=None,
        #                         help='path to the model weights.pt')
        #     parser.add_argument('--out_stride', type=int, default=16,
        #                         help='network output stride (default: 16)')
        #     args = parser.parse_args()
        #     kwargs = {'num_workers': 4, 'pin_memory': True}
        #     _, _,test_loader, nclass = make_data_loader(args, **kwargs)

            # SAVE_BASE_PATH = f'E:\\Data\\Dataset\\Results\\Segmentation\\segmentation_pretrain\\' 
            # if os.path.isdir(SAVE_BASE_PATH+name+'_model') is False:         
            #     os.mkdir(SAVE_BASE_PATH+name+'_model')

            # paths = SAVE_BASE_PATH+name+'_model\\'

            # if os.path.isdir(paths+args.dataset) is False:
            #     os.mkdir(paths+args.dataset)

            # model = DeepLab(num_classes=2,
            #                     backbone='resnet',
            #                     output_stride=16,
            #                     sync_bn=False,
            #                     freeze_bn=False)

            # args.cuda = not args.no_cuda and torch.cuda.is_available()
            # if args.cuda:
            #     try:
            #         args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
            #     except ValueError:
            #         raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')
            # n_classes = 2
        
            # gt_path_model = f"E:\\Data\\Dataset\\Processed_image\\lungs-segmentation-dataset\\{datasets}\\"
            # save_output_images(LOAD_PATH_MODEL,model,test_loader,os.path.join(paths,args.dataset),gt_path_model,datasets)

        parser = argparse.ArgumentParser(description="Inference")
        parser.add_argument('--image-path', type=str, default=None,
                            help='path to the image')
        parser.add_argument('--no-cuda', action='store_true', default=
                            False, help='disables CUDA training')
        parser.add_argument('--gpu-ids', type=str, default='0',
                            help='use which gpu to train, must be a \
                            comma-separated list of integers only (default=0)')
        # # added by uk
        parser.add_argument('--size', type=int, default=256,
                            help='image size')
        
        # # added by bishnu 
        parser.add_argument('--model', type=str, default='deeplabv3+',
                    choices=['deeplabv3','deeplabv3+','deeplabv3plus','fcn'])
        # parser.add_argument('--backbone', type=str, default='resnet',
        #                     choices=['resnet', 'xception', 'drn', 'mobilenet'],
        #                     help='backbone name (default: resnet)')
        parser.add_argument('--weights', type=str, default=None,
                            help='path to the model weights.pt')
        parser.add_argument('--out_stride', type=int, default=16,
                            help='network output stride (default: 16)')
        args = parser.parse_args()
        kwargs = {'num_workers': 4, 'pin_memory': True}
        image_for_inference = preprocess_image(args.image_path)

        # #  changed by Bishnu 
        # if args.model=='deeplabv3+':
        #     # for testing the model should be the best.pt
        #     model = DeepLab(num_classes=nclass,
        #                 backbone=args.backbone,
        #                 output_stride=args.out_stride,
        #                 sync_bn=args.sync_bn,
        #                 freeze_bn= args.freeze_bn) # freezes the BatchNorm layers, meaning they won’t update their running mean and variance during evaluation.
        # elif args.model=='deeplabv3':
        #     model= DeepLabV3Resnet101(nclass, 256)
        # else: 
        #     model = FCNResNet101(nclass, 512)
        
        # assert that model_path is a valid path
        assert os.path.exists(args.weights), f"Path does not exist: {args.weights}"

        # loaded_model = model.load_state_dict(torch.load(args.weights,weights_only=True))
        loaded_model = torch.load(args.weights, weights_only=True)
        loaded_model.eval()

        # # whether to use class balanced weights
        # if args.use_balanced_weights:
        #     classes_weights_path = os.path.join(Path.db_root_dir(args.dataset), args.dataset+'_classes_weights.npy')
        #     if os.path.isfile(classes_weights_path):
        #         weight = np.load(classes_weights_path)
        #     else:
        #         weight = calculate_weigths_labels(args.dataset, test_loader, nclass)
        #     weight = torch.from_numpy(weight.astype(np.float32))
        # else:
        #     weight = None

        # do not require losses   
        # criterion = SegmentationLosses(
        #     weight=weight, 
        #     cuda=args.cuda).build_loss(mode=args.loss_type)
        
        with torch.no_grad():
            output = loaded_model(image_for_inference.cuda())
            if args.model=='fcn':
                output=output['out']
            if args.model =='deeplabv3':
                output=output['out']


        # Remove batch dimension and move to CPU
        output = output.squeeze(0).cpu()
        
        # Convert mask to binary
        prediction = (output.argmax(dim=0) > 0).numpy().astype(np.uint8)  # Binary mask (0 and 1)
        print(f'prediction shape: {prediction.shape}, {np.unique(prediction)}')
        # # Prepare the original image for visualization
        original_image = load_image(image_path=args.image_path)

        mask_path = PathLib(args.image_path).parent.parent
        mask_file_name = PathLib(args.image_path).stem + '.png'
        mask_path = mask_path / "masks" / mask_file_name
        target = load_mask(mask_path)
        print(f'target size: {target.shape}, {np.unique(target)}')
        
        # Call visualization
        plot_inference(image=original_image, target=target, prediction=prediction)
