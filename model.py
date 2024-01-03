import torch as t
# import warnings
# warnings.filterwarnings("ignore")

class ResNet(t.nn.Module):

    def __init__(self):
        super(ResNet, self).__init__()

        self.layers = [
            t.nn.Conv2d(3, 64, 7, 2),
            t.nn.BatchNorm2d(64),
            t.nn.LeakyReLU(),
            t.nn.MaxPool2d(3, 2),
            ResBlock(64, 64, 1),
            ResBlock(64, 128, 2),
            ResBlock(128, 256, 2),
            ResBlock(256, 512, 2),
            t.nn.AdaptiveAvgPool2d((1, 1)),
            t.nn.Flatten(),
            t.nn.Linear(512, 2),
            t.nn.Sigmoid(),
            ]

        self.layers = t.nn.ModuleList(self.layers)

    def forward(self, input_tensor):
        x = input_tensor
        for layer in self.layers:
            x = layer.forward(x)
        return x


 
class ResBlock(t.nn.Module):

    def __init__(self, input_channels, output_channels, stride):
        super().__init__()
        self.identity_downsample = t.nn.Sequential(
            t.nn.Conv2d(input_channels, output_channels, kernel_size=1, stride=stride),
            t.nn.BatchNorm2d(output_channels)
        )
        
        self.layers =[
            t.nn.Conv2d(input_channels, output_channels, kernel_size = 3, stride = stride, padding = 1),
            t.nn.BatchNorm2d(output_channels),
            t.nn.LeakyReLU(),
            t.nn.Conv2d(output_channels, output_channels, kernel_size = 3, padding = 1),
            t.nn.BatchNorm2d(output_channels),
            t.nn.LeakyReLU()]

        self.layers = t.nn.ModuleList(self.layers)

    def forward(self, input_tensor):
        x = input_tensor
        for i, layer in enumerate(self.layers):
            x = layer.forward(x)
            if i == 4:
                input_tensor = self.identity_downsample(input_tensor)
                x += input_tensor
        return x