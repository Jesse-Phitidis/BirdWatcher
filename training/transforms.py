import torch
from torchvision.transforms import v2 as transforms
import torchvision.transforms.v2.functional as F
from PIL import Image
import random


class RandomLowResolution(torch.nn.Module):

    def __init__(self, factor_range=(1, 4), max_aniso=1, p=0.5):
        super().__init__()
        self.min_factor = factor_range[0]
        self.max_factor = factor_range[1]
        self.max_aniso = max_aniso
        self.p = p

    def forward(self, img):
        
        if random.random() > self.p:
            return img

        factor_h = random.uniform(self.min_factor, self.max_factor)
        factor_w = random.uniform(
            max(self.min_factor, factor_h / self.max_aniso),
            min(self.max_factor, factor_h * self.max_aniso)
            )

        if isinstance(img, torch.Tensor):
            orig_h, orig_w = img.shape[-2], img.shape[-1] # Tensor has shape (..., H, W)
        elif isinstance(img, Image.Image):
            orig_w, orig_h = img.size[-2], img.size[-1] # PIL has shape (W, H)
        
        low_h = max(1, int(orig_h / factor_h))
        low_w = max(1, int(orig_w / factor_w))

        img = F.resize(img, size=[low_h, low_w], interpolation=transforms.InterpolationMode.NEAREST)
        img = F.resize(img, size=[orig_h, orig_w], interpolation=transforms.InterpolationMode.BILINEAR)
        
        return img