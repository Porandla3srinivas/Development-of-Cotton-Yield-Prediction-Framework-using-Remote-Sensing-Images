import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np
import cv2


class YOLO3DFeatureExtractor(nn.Module):
    def __init__(self, in_channels=1, out_channels=64):
        super(YOLO3DFeatureExtractor, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.pool = nn.MaxPool3d(2, 2)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, dim):
        super(TransformerBlock, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=8)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = x.permute(2, 0, 1)  # (B, C, D) → (D, B, C)
        attn_out, _ = self.attn(x, x, x)
        x = self.norm(attn_out + x)
        x = x.permute(1, 2, 0)  # Back to (B, C, D)
        return x


class UNetPPDecoder(nn.Module):
    def __init__(self, in_channels=64, out_channels=1):
        super(UNetPPDecoder, self).__init__()
        self.upconv1 = nn.ConvTranspose3d(in_channels, 32, kernel_size=2, stride=2)
        self.conv2 = nn.Conv3d(32, out_channels, kernel_size=1)

    def forward(self, x):
        x = F.relu(self.upconv1(x))
        x = torch.sigmoid(self.conv2(x))  # Output Segmentation Mask
        return x


class YOLOTransformerUNetPP(nn.Module):
    def __init__(self, in_channels=1, num_classes=1):
        super(YOLOTransformerUNetPP, self).__init__()
        self.yolo = YOLO3DFeatureExtractor(in_channels, 64)
        self.transformer = TransformerBlock(64)
        self.unetpp = UNetPPDecoder(64, num_classes)

    def forward(self, x):
        x = self.yolo(x)          # Feature Extraction
        x = self.transformer(x)   # Context Awareness
        x = self.unetpp(x)        # Segmentation
        return x


def Model_3DYTUnetPlusPlus(input_image, GT):

    # Initialize Model
    model = YOLOTransformerUNetPP(in_channels=1, num_classes=1)

    # Forward Pass
    Segmented_img = model(input_image)
    return Segmented_img

