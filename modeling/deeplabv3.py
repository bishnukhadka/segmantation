import torch
from torchvision.models.segmentation import deeplabv3_resnet101

class DeepLabV3Resnet101(torch.nn.Module):
    def __init__(self, num_classes,size, pretrained=False):
        super(DeepLabV3Resnet101, self).__init__()
        if pretrained:
            # Load the pre-trained fcn_resnet101 model
            self.model = deeplabv3_resnet101(pretrained=True) 
            # self.model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet101', pretrained=True)
        else:
            # Load model with no weights
            self.model = deeplabv3_resnet101(
                                            weights=None, 
                                            weights_backbone=None)
        # Replace the final classifier to match the number of classes
        self.model.classifier[4] = torch.nn.Conv2d(size, num_classes, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x):
        return self.model(x)

# Example usage:
if __name__ == "__main__":
    num_classes = 2  # Change to the number of classes in your segmentation task
    size = 512
    model = DeepLabV3Resnet101(num_classes, size)
    print(model)
    input = torch.randn(4, 3, size, size)  # Change the input size as needed
    print(f'input.shape: {size}*{size}')
    output = model(input)
    print(output['out'].size())