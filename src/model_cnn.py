from torch import nn


### Build simple CNN
class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(32, 64, kernel_size=5, stride=2),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 128, kernel_size=5),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=3),
            nn.Conv2d(128, 128, kernel_size=3),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=3),
        )

        self.fc = nn.Sequential(
            nn.Linear(768, 768),
            nn.LeakyReLU(),
            nn.Linear(768, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 128),
        )

        # self.fc6 = nn.Linear(768, 240)
        # self.fc7 = nn.Linear(240, 100)

    def forward(self, input):
        x = self.conv(input)
        x = x.view(-1, 768)
        x = self.fc(x)

        return x


def get_model(type="custom", pretrained=True):
    from torchvision.models import get_model

    if type == "custom":
        return CNNModel()

    if pretrained:
        model = get_model(type, weights="DEFAULT")

        for param in model.parameters():
            param.requires_grad = False

        if hasattr(model, "fc"):
            for param in model.fc.parameters():
                param.requires_grad = True
        if hasattr(model, "classifier"):
            for param in model.classifier.parameters():
                param.requires_grad = True

        return model

    model = get_model(type, weights=None)
    return model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        prog="Model Info Script",
    )
    parser.add_argument("model_type", default="custom", help="Type of model")
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction)

    args = parser.parse_args()
    model_type = str(args.model_type)
    pretrained = bool(args.pretrained)

    from torchinfo import summary

    model = get_model(model_type, pretrained=pretrained)

    print(summary(model, input_size=(4, 3, 640, 480)))
