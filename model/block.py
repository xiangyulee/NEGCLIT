import torch
import torch.nn as nn









# model 1
class SimpleCNN(nn.Module):
    def __init__(self, in_channels, channels, kernel_sizes, num_classes,NE,NG, device=torch.device('cuda'),input_size=[32,32]):
        super(SimpleCNN, self).__init__()

        layers = []
        prev_channels = in_channels

        # Wrap channels and kernel_sizes in lists if they are integers
        channels = [channels] if isinstance(channels, int) else channels
        kernel_sizes = [kernel_sizes] if isinstance(kernel_sizes, int) else kernel_sizes

        for ch, kernel_size in zip(channels, kernel_sizes):
            conv = nn.Conv2d(prev_channels, ch, kernel_size, stride=1, padding=(kernel_size // 2))
            relu = nn.ReLU()
            # pool = nn.MaxPool2d(kernel_size=2, stride=2)

            layers.extend([conv, relu])

            prev_channels = ch
        conv = nn.Conv2d(prev_channels,in_channels,3, stride=1, padding=(3 // 2))
        relu = nn.ReLU()
        layers.extend([conv, relu])
        self.layers = nn.Sequential(*layers)

        self.num_classes = num_classes
        self.device = device
        # self.to(self.device)
        

        # Determine the output shape of the convolutional layers
        test_input = torch.randn(1,3,*input_size)
        test_input=test_input.to(device)
        test_input=NE.features(test_input)
        for layer in NG.NG_block:
            test_input=layer(test_input)
        self.layers=self.layers.to(device)
        test_output = self.layers(test_input)
        # print("SimpleCNN Convolutional output shape:", test_output.shape)

        conv_output_size = test_output.numel()

        # self.fc1.weight = nn.Parameter(self.fc1.weight.data.to(device))
        # self.fc1.bias = nn.Parameter(self.fc1.bias.data.to(device))
        self.to(self.device)

    def forward(self, x):
        x = self.layers(x)

        return x

# model 2
class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SEBlockCNN(nn.Module):
    def __init__(self, in_channels, channels, kernel_sizes, num_classes,NE,NG, device=torch.device('cuda'),input_size=[32,32]):
        super(SEBlockCNN, self).__init__()

        layers = []
        prev_channels = in_channels

        # Wrap channels and kernel_sizes in lists if they are integers
        channels = [channels] if isinstance(channels, int) else channels
        kernel_sizes = [kernel_sizes] if isinstance(kernel_sizes, int) else kernel_sizes

        for ch, kernel_size in zip(channels, kernel_sizes):
            conv = nn.Conv2d(prev_channels, ch, kernel_size, stride=1, padding=(kernel_size // 2))
            relu = nn.ReLU()
            # pool = nn.MaxPool2d(kernel_size=2, stride=2)

            layers.extend([conv, relu])

            prev_channels = ch
        conv = nn.Conv2d(prev_channels,in_channels,4, stride=1, padding=(4 // 2))
        relu = nn.ReLU()
        layers.extend([conv, relu])
        self.layers = nn.Sequential(*layers)

        self.num_classes = num_classes
        self.device = device

        # Determine the output shape of the convolutional layers
        test_input = torch.randn(1,3,*input_size)
        test_input=test_input.to(device)
        test_input=NE.features(test_input)
        for layer in NG.NG_block:
            test_input=layer(test_input)
        self.layers=self.layers.to(device)
        test_output = self.layers(test_input)
        # print("SEBlockCNN Convolutional output shape:", test_output.shape)

        conv_output_size = test_output.numel()



        # Add the SEBlock layers after each ReLU layer
        layers_with_se = []
        for layer in self.layers:
            layers_with_se.append(layer)
            if isinstance(layer, nn.ReLU):
                layers_with_se.append(SEBlock(prev_channels))

        self.layers = nn.Sequential(*layers_with_se)

        self.to(self.device)

    def forward(self, x):
        x = self.layers(x)


        return x