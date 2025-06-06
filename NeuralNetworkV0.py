from torch import nn


class NeuralNetworkV0(nn.Module):


    def __init__(self, in_shape, out_shape):
        super().__init__()
        self.block_1 = nn.Sequential(
            nn.Conv2d(in_channels=in_shape, 
                      out_channels=64, 
                      kernel_size=(3, 3),
                      stride=1, 
                      padding='same'),
            nn.ReLU(),
            nn.Conv2d(in_channels=64,
                     out_channels=64,
                     kernel_size=(3, 3),
                     stride=1,
                     padding='same'),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        )

        self.block_2 = nn.Sequential(
            nn.Conv2d(in_channels=64, 
                      out_channels=128, 
                      kernel_size=(3, 3),
                      stride=1, 
                      padding='same'),
            nn.ReLU(),
            nn.Conv2d(in_channels=128,
                     out_channels=128,
                     kernel_size=(3, 3),
                     stride=1,
                     padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=524288, out_features=out_shape)
        )
    

    def forward(self, x):
        return self.classifier(self.block_2(self.block_1(x)))
        