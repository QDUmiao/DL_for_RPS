import torch
import torch.nn as nn

class CBAM_ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(CBAM_ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio,bias=False),
            nn.Mish(),
            nn.Linear(in_channels // reduction_ratio, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        
        avg_out = self.avg_pool(x).view(b, c)
        avg_out = self.fc(avg_out).view(b, c, 1, 1)
        
        max_out = self.max_pool(x).view(b, c)
        max_out = self.fc(max_out).view(b, c, 1, 1)
        

        out = avg_out + max_out
        return x * out.expand_as(x)
    

class CBAM_SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(CBAM_SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(out)
        return x *  self.sigmoid(out)
    
    


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        # self.relu = nn.LeakyReLU()
        self.relu = nn.Mish()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(residual)
        out = self.relu(out)
        return out

class Model(nn.Module):
    def __init__(self, num_classes=1000):
        super(Model, self).__init__()

        self.feature = nn.Sequential(
            
            nn.Conv2d(3, 16, kernel_size=7, padding=4, stride=2),
            nn.BatchNorm2d(16),
            nn.Mish(),
            CBAM_ChannelAttention(16),
            CBAM_SpatialAttention(),
            nn.MaxPool2d(2),

            ResidualBlock(16, 32, stride=2),
            CBAM_ChannelAttention(32),
            CBAM_SpatialAttention(),
            nn.MaxPool2d(2),

            ResidualBlock(32, 64, stride=1),
            CBAM_ChannelAttention(64),
            CBAM_SpatialAttention(),
            nn.MaxPool2d(2),

            ResidualBlock(64, 128, stride=1),
            CBAM_ChannelAttention(128),
            CBAM_SpatialAttention(),
            nn.MaxPool2d(2),
            
            ResidualBlock(128, 256, stride=1),
            CBAM_ChannelAttention(256),
            CBAM_SpatialAttention(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(2304, 1024),
            nn.Mish(),
            nn.Dropout(0.5),
            
            nn.Linear(1024, 512),
            nn.Mish(),
            nn.Dropout(0.5),
            
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        if x.size(-1) != 224 or x.size(-2) != 224:
            x = torch.nn.functional.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)

        in_size = x.size(0)
        out = self.feature(x)
        out = out.view(in_size, -1)
        out = self.classifier(out)

        return out
