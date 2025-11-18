# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from torch.nn import functional as F

from typing import List, Tuple, Type

from model.src.common import LayerNorm2d



class MaskDecoder(nn.Module):
    def __init__(
        self,
        *,
        transformer_dim: int,
        transformer: nn.Module,
        activation: Type[nn.Module] = nn.GELU,
        norm : Type[nn.Module] = nn.BatchNorm2d,
        act : Type[nn.Module] = nn.GELU
        

    ) -> None:
        """
        Predicts masks given an image and prompt embeddings, using a
        transformer architecture.

        Arguments:
          transformer_dim (int): the channel dimension of the transformer
          transformer (nn.Module): the transformer used to predict masks
          num_multimask_outputs (int): the number of masks to predict
            when disambiguating masks
          activation (nn.Module): the type of activation to use when
            upscaling masks
          iou_head_depth (int): the depth of the MLP used to predict
            mask quality
          iou_head_hidden_dim (int): the hidden dimension of the MLP
            used to predict mask quality
        """
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer

        self.iou_token = nn.Embedding(1, transformer_dim)
        self.mask_tokens = nn.Embedding(1, transformer_dim)

        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
            activation(),
        )
        #self.no_mask_embed = nn.Embedding(1, 256)

        self.output_hypernetworks_mlps = MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)

        # attention part

        self.conv_1d_embedding =nn.Sequential(
            nn.Conv1d(in_channels=32,out_channels=36,kernel_size=1,stride=1),
            nn.ReLU()
            )
        self.conv_1d_hs =nn.Sequential(
            nn.Conv1d(in_channels=1,out_channels=32,kernel_size=1,stride=1),
            nn.ReLU(),
            nn.Linear(32,36),
            nn.ReLU()
            )
        
        self.conv_q = nn.Sequential(
            nn.Conv2d(32,32,kernel_size=1,stride=1,padding=0),
            nn.ReLU()
        )

        self.conv_k = nn.Sequential(
            nn.Conv2d(32,32,kernel_size=1,stride=1,padding=0),
            nn.ReLU()
        )

        self.conv_v = nn.Sequential(
            nn.Conv2d(32,32,kernel_size=1,stride=1,padding=0),
            nn.ReLU()
        )

        self.attention_conv = nn.Sequential(
            nn.Conv2d(32,1,kernel_size=1,stride=1,padding=0),
            nn.Sigmoid()
        )




    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings:torch.Tensor,
        dense_prompt_embeddings:torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict masks given image and prompt embeddings.

        Arguments:
          image_embeddings (torch.Tensor): the embeddings from the image encoder
          image_pe (torch.Tensor): positional encoding with the shape of image_embeddings
          sparse_prompt_embeddings (torch.Tensor): the embeddings of the points and boxes
          dense_prompt_embeddings (torch.Tensor): the embeddings of the mask inputs
          multimask_output (bool): Whether to return multiple masks or a single
            mask.

        Returns:
          torch.Tensor: batched predicted masks
          torch.Tensor: batched predictions of mask quality
        """
        masks = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings
        )

        # Select the correct mask or masks for output

        # Prepare output
        return masks

    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predicts masks. See 'forward' for more details."""
        # Concatenate output tokens
        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(image_embeddings.shape[0],-1,-1)
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)
        if image_embeddings.shape[0] != tokens.shape[0]:
            src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        else:
            src = image_embeddings

        src = src + dense_prompt_embeddings
        pos_src = image_pe
        b, c, h, w = src.shape

        # Run the transformer
        hs, src = self.transformer(src, pos_src, output_tokens)



        hs = output_tokens[:,1:,:]

        hs = self.output_hypernetworks_mlps(hs)



        # Upscale mask embeddings and predict masks using the mask tokens
        src = src.transpose(1, 2).view(b, c, h, w).contiguous()
        upscaled_embedding = self.output_upscaling(src)

        b, c, h, w = upscaled_embedding.shape
   
        """

        Attention Part
        
        """

        new_hs = self.conv_1d_hs(hs)
        hs_q,hs_k,hs_v = torch.split(new_hs, 36//3, dim=-1)

        embedding = upscaled_embedding.view(b, c, h * w)
        embedding = self.conv_1d_embedding(embedding)

        q_e , k_e, v_e = torch.split(embedding, 36//3, dim=1)

        q = hs_q @ q_e

        
        q = q.reshape(q.size()[0],32,128,128)
        k = hs_k @ k_e
        k = k.reshape(q.size()[0],32,128,128)
        v = hs_v @ v_e
        v = v.reshape(q.size()[0],32,128,128)

        q = self.conv_q(q)
        k = self.conv_k(k)
        v = self.conv_v(v)

        atten = q*k*v 
        atten = self.attention_conv(atten)


        masks = (hs @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)
       
        return masks*atten, upscaled_embedding*atten


# Lightly adapted from
# https://github.com/facebookresearch/MaskFormer/blob/main/mask_former/modeling/transformer/transformer_predictor.py # noqa
class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x

