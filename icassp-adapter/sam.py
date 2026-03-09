import torch
import torch.nn as nn
from model.train_utils.sam import SAM

def getmodel():
    model = SAM(512)



    # 初始冻结所有参数
    for param in model.parameters():
        param.requires_grad = False

    # 解冻image_encoder中的适配器参数
    for i, block in enumerate(model.image_encoder.blocks):
        if hasattr(block, "adapter"):
            for param in block.adapter.parameters():
                param.requires_grad = True
    
    # 解冻image_encoder以外的所有参数
    for name, param in model.named_parameters():
        # 只处理非image_encoder的参数
        if "image_encoder" not in name:
            param.requires_grad = True
    
    # 打印可训练参数（可选）
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"可训练参数: {name}")


    
    return model


if __name__ == "__main__":

    a = getmodel()




