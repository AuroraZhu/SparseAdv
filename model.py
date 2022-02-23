import torch
import torch.nn as nn


class ld_encoder(nn.Module):

    def __init__(self):
        super().__init__()
        self.enconv1_1 = nn.Sequential(
            nn.ConvTranspose2d(3, 8, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8)
        )

        self.enconv1_2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(16)
        )

        self.enconv2_1 = nn.Sequential(
            nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8)
        )

        self.enconv2_2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(16)
        )

        self.deconv1_1 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32)
        )
        self.deconv1_2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(kernel_size=2)
        )

        self.deconv1_3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=3, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(3),
        )

        self.klconv1_1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),
            nn.MaxPool2d(kernel_size=2)
        )

        self.klconv1_2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(kernel_size=2)
        )

        self.klconv1_3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=30, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(30),
            # nn.MaxPool2d(kernel_size=2)
            nn.AvgPool2d(kernel_size=7)
        )


    def forward(self, x):
        x = self.enconv1_1(x)
        x = self.enconv1_2(x)
        x = self.enconv2_1(x)
        en_x = self.enconv2_2(x)

        x = self.deconv1_1(en_x)
        x = self.deconv1_2(x)
        de_x = self.deconv1_3(x)
        de_x2 = torch.tanh(de_x)

        x=self.klconv1_1(de_x)
        x=self.klconv1_2(x)
        x=self.klconv1_3(x)
        a,b=x.size(0),x.size(1)
        kl_x=x.view(a,b)

        return en_x,de_x2,kl_x

if __name__ == "__main__":

    net=ld_encoder()

    x = torch.randn(2, 3, 224, 224)
    y_en,y_de,kl_y = net(x)

    print('y', y_en.size())
    print('y', y_de.size())


