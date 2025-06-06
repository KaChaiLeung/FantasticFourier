from torch import nn


class NeuralNetworkV1(nn.Module):


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
            nn.ReLU(),
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

        self.block_3 = nn.Sequential(
            nn.Conv2d(in_channels=128, 
                      out_channels=256, 
                      kernel_size=(3, 3), 
                      stride=1, 
                      padding='same'),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, 
                      out_channels=256, 
                      kernel_size=(3, 3), 
                      stride=1, 
                      padding='same'),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, 
                      out_channels=256, 
                      kernel_size=(3, 3), 
                      stride=1, 
                      padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        )

        self.block_4 = nn.Sequential(
            nn.Conv2d(in_channels=256, 
                      out_channels=512, 
                      kernel_size=(3, 3), 
                      stride=1, 
                      padding='same'),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, 
                      out_channels=512, 
                      kernel_size=(3, 3), 
                      stride=1, 
                      padding='same'),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, 
                      out_channels=512, 
                      kernel_size=(3, 3), 
                      stride=1, 
                      padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        )

        self.block_5 = nn.Sequential(
            nn.Conv2d(in_channels=512, 
                      out_channels=512, 
                      kernel_size=(3, 3), 
                      stride=1, 
                      padding='same'),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, 
                      out_channels=512, 
                      kernel_size=(3, 3), 
                      stride=1, 
                      padding='same'),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, 
                      out_channels=512, 
                      kernel_size=(3, 3), 
                      stride=1, 
                      padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=32768, out_features=4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096, out_features=out_shape)
        )
    

    def forward(self, x):
        return self.classifier(
                    self.block_5(
                        self.block_4(
                            self.block_3(
                                self.block_2(
                                    self.block_1(x)
                                    )
                                )
                            )
                        )
                    )