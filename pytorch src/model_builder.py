from torch import nn,fft
class CNNModel(nn.Module):
    def __init__(self, input_shape, output_shape, hidden_layers):
        super().__init__()

        self.conv1 = nn.Sequential(


            nn.Conv2d(in_channels=input_shape, out_channels=hidden_layers, kernel_size=(3,3), stride=(1,1), padding=1),
            nn.BatchNorm2d(hidden_layers),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv2d(in_channels=hidden_layers, out_channels=hidden_layers*4, kernel_size=(3,3), stride=(1,1), padding=1),
            nn.BatchNorm2d(hidden_layers*4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)),
            nn.Dropout(0.3),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_layers*4, out_channels=hidden_layers*2, kernel_size=(3,3), stride=(1,1), padding=1),
            nn.BatchNorm2d(hidden_layers*2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv2d(in_channels=hidden_layers*2, out_channels=hidden_layers, kernel_size=(3,3), stride=(1,1), padding=1),
            nn.BatchNorm2d(hidden_layers),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)),
            nn.Dropout(0.3),

        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_layers * 15 * 15 , out_features=output_shape)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.classifier(x)
        return x





