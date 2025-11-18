import torch
import torch.nn as nn
from model.train_utils.sam import SAM

def getmodel():
    model = SAM(512)

    # Initially freeze all parameters
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze adapter parameters in image_encoder
    for i, block in enumerate(model.image_encoder.blocks):
        if hasattr(block, "adapter"):
            for param in block.adapter.parameters():
                param.requires_grad = True
    
    # Unfreeze all parameters except image_encoder
    for name, param in model.named_parameters():
        # Only process parameters that are not in image_encoder
        if "image_encoder" not in name:
            param.requires_grad = True
    
    # Print trainable parameters (optional)
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"Trainable parameters: {name}")
    
    return model


if __name__ == "__main__":

    getmodel()




