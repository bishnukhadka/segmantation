import torch
from torchvision.models.segmentation import fcn_resnet101

class FCNResNet101(torch.nn.Module):
    def __init__(self, num_classes, size, pretrained=False):
        super(FCNResNet101, self).__init__()
        if pretrained:
            # Load the pre-trained fcn_resnet101 model
            self.model = fcn_resnet101(pretrained=True) # will give imagenet trained backbone
        else:
            # Load model with no weights
            self.model = fcn_resnet101(weights=None, weights_backbone=None)
        # Replace the final classifier to match the number of classes
        self.model.classifier[4] = torch.nn.Conv2d(size, num_classes, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x):
        return self.model(x)

# Example usage:
if __name__ == "__main__":
    num_classes = 2  # Change to the number of classes in your segmentation task
    size = 256
    model = FCNResNet101(num_classes,512)
    print(model)
    input = torch.randn(1, 3, size, size)  # Change the input size as needed
    print(input.shape)
    output = model(input)
    print(output['out'].size())