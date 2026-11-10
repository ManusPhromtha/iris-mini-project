from torch import nn


### Build simple CNN
class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=7, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(32, 64, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 128, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3),
            nn.Conv2d(128, 128, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3),
        )

        self.fc = nn.Sequential(nn.Linear(768, 768), nn.ReLU(), nn.Linear(768, 512), nn.ReLU(), nn.Linear(512, 128))

        # self.fc6 = nn.Linear(768, 240)
        # self.fc7 = nn.Linear(240, 100)

    def forward(self, input):
        x = self.conv(input)
        x = x.view(-1, 768)
        x = self.fc(x)

        return x


custom_model = CNNModel()

if __name__ == "__main__":
    from torchinfo import summary

    print(summary(custom_model, input_size=(1, 640, 480)))
