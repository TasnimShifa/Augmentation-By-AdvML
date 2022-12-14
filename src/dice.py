import torch
import numpy as np

SMOOTH = 1e-6

def dice_pytorch(outputs: torch.Tensor, labels: torch.Tensor):
    # You can comment out this line if you are passing tensors of equal shape
    # But if you are passing output from UNet or something it will most probably
    # be with the BATCH x 1 x H x W shape
    outputs = outputs.squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W
    intersection = (outputs & labels).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
    union = (outputs | labels).float().sum((1, 2))        # Will be zzero if both are 0
    
    dice = 2. * ( intersection + SMOOTH) / (union + SMOOTH)  # We smooth our devision to avoid 0/0
    
    #thresholded = torch.clamp(20 * (dice - 0.5), 0, 10).ceil() / 10  # This is equal to comparing with thresolds
    thresholded = torch.clamp(dice, 0, 10) # This is equal to comparing with thresolds
    return thresholded