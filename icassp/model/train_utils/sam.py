import torch
from torch import nn
from model.src_new.mask_decoder import MaskDecoder
from model.src_new.ummfeb import Unet


from model.src.image_encoder import ImageEncoderViT 
from model.src.transformer import TwoWayTransformer
from model.src.common import LayerNorm2d
from model.src.prompt import PromptEncoder

from typing import Any, Optional, Tuple, Type

from reprlib import recursive_repr
import numpy as np

from model.src.block import MLFusion




from model.fusion_model.main import UMFEB

class SAM(nn.Module):
    def __init__(self, img_size = 512, norm = nn.BatchNorm2d, act = nn.ReLU):
        super().__init__()

        self.sigmoid = nn.Sigmoid()

        self.pe_layer = PositionEmbeddingRandom(256 // 2)


        self.image_embedding_size = [img_size // 16, img_size // 16]
        self.img_size = img_size

        self.image_encoder = ImageEncoderViT(depth=12,
            embed_dim=768,
            img_size=img_size,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=12,
            patch_size=16,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=[2, 5, 8, 11],
            window_size=14,
            out_chans=256)
        
        for _, param in self.image_encoder.named_parameters():
        
            param.requires_grad = False

        self.conv_mask_feature = nn.Sequential(
            nn.Conv2d(16,32,kernel_size=1,stride=1,padding=0),
            nn.ReLU()
        )

        self.mask_decoder1 = MaskDecoder(
            transformer=TwoWayTransformer(
                depth = 2,
                embedding_dim = 256,
                mlp_dim = 2048,
                num_heads = 8
            ),
            transformer_dim=256,
            norm = norm,
            act = act
        )

        self.mask_decoder2 = MaskDecoder(
            transformer=TwoWayTransformer(
                depth = 2,
                embedding_dim = 256,
                mlp_dim = 2048,
                num_heads = 8
            ),
            transformer_dim=256,
            norm = norm,
            act = act
        )

        self.mask_decoder3 = MaskDecoder(
            transformer=TwoWayTransformer(
                depth = 2,
                embedding_dim = 256,
                mlp_dim = 2048,
                num_heads = 8
            ),
            transformer_dim=256,
            norm = norm,
            act = act
        )

        self.mask_decoder4 = MaskDecoder(
            transformer=TwoWayTransformer(
                depth = 2,
                embedding_dim = 256,
                mlp_dim = 2048,
                num_heads = 8
            ),
            transformer_dim=256,
            norm = norm,
            act = act
        )

        checkpoint = torch.load('model_best.pth', weights_only=False)
        self.mask_decoder1.load_state_dict(checkpoint,strict=False)
        self.mask_decoder2.load_state_dict(checkpoint,strict=False)
        self.mask_decoder3.load_state_dict(checkpoint,strict=False)
        self.mask_decoder4.load_state_dict(checkpoint,strict=False)

        self.prompt_encoder1 = PromptEncoder(embed_dim = 256,
            image_embedding_size = [512//16,512//16],
            input_image_size = [512,512],
            mask_in_chans = 16,
            activation = nn.GELU
            )
        
        self.prompt_encoder2 = PromptEncoder(embed_dim = 256,
            image_embedding_size = [512//16,512//16],
            input_image_size = [512,512],
            mask_in_chans = 16,
            activation = nn.GELU
            )
        
        self.prompt_encoder3 = PromptEncoder(embed_dim = 256,
            image_embedding_size = [512//16,512//16],
            input_image_size = [512,512],
            mask_in_chans = 16,
            activation = nn.GELU
            )
        
        self.prompt_encoder4 = PromptEncoder(embed_dim = 256,
            image_embedding_size = [512//16,512//16],
            input_image_size = [512,512],
            mask_in_chans = 16,
            activation = nn.GELU
            )
        
        self.fusion_block = MLFusion(norm = norm, act = act)
        
        self.mask_embedding1 = Unet(32,32,16)
        self.mask_embedding2 = Unet(32,32,16)
        self.mask_embedding3 = Unet(32,32,16)
        self.mask_embedding4 = Unet(32,32,16)

        self.mask_get1 = nn.Sequential(
            nn.Conv2d(32,1,kernel_size=1,stride=1,padding=0),
            nn.Sigmoid()
        )
        
        self.mask_get2 = nn.Sequential(
            nn.Conv2d(32,1,kernel_size=1,stride=1,padding=0),
            nn.Sigmoid()
        )

        self.mask_get3 = nn.Sequential(
            nn.Conv2d(32,1,kernel_size=1,stride=1,padding=0),
            nn.Sigmoid()
        )
        
        self.mask_get4 = nn.Sequential(
            nn.Conv2d(32,1,kernel_size=1,stride=1,padding=0),
            nn.Sigmoid()
        )

        '''
        
        fusion image embedding
        
        '''

        '''

        self.fie3 = FusionModel(256,256)
        self.fie2 = FusionModel(256,256)
        self.fie1 = FusionModel(256,256)

        '''

        self.fie3 = UMFEB(256,2)
        self.fie2 = UMFEB(256,2)
        self.fie1 = UMFEB(256,2)


        '''
        
        fusion mask embedding
        
        '''

        self.fme3 = UMFEB(32,2)
        self.fme2 = UMFEB(32,3)
        self.fme1 = UMFEB(32,4)
        

    def get_dense_pe(self) -> torch.Tensor:
        """
        Returns the positional encoding used to encode point prompts,
        applied to a dense set of points the shape of the image encoding.

        Returns:
          torch.Tensor: Positional encoding with shape
            1x(embed_dim)x(embedding_h)x(embedding_w)
        """
        return self.pe_layer(self.image_embedding_size).unsqueeze(0)

    def forward(self, img):
        #get different layer's features of the image encoder
        img_pe = self.get_dense_pe()
        features_list = self.image_encoder(img)


        img_feature : list = self.fusion_block(features_list)

        img_feature1,img_feature2,img_feature3,img_feature4 = img_feature





        # fusion part 4


        mask_feature4 = img_feature4.reshape(img_feature[0].size(0), 16, 128, 128)
        mask_feature4 = self.conv_mask_feature(mask_feature4)

        #embedding mask features
        mask_feature4 = self.mask_embedding4(mask_feature4)


        mask4 = self.mask_get4(mask_feature4)

        promote_sparse4, promote_dense4= self.prompt_encoder4(points=None,boxes=None,masks=mask4)



        #extract intermediate outputs for deep supervision to prevent model overfitting on the detail enhancement module.
        
        coarse_mask4, feature4= self.mask_decoder4(img_feature[-1], 
                                                img_pe,
                                                promote_sparse4,
                                                promote_dense4)
        
        
        



        # fusion part 3

        mask_feature_list3 = [mask_feature4, feature4]

        mask_feature3 = self.fme3(mask_feature_list3)


        img_list3 = [img_feature4,img_feature3]
        img_feature3 = self.fie3(img_list3)


   
        #embedding mask features
        mask_feature3 = self.mask_embedding3(mask_feature3)
        mask3 = self.mask_get3(mask_feature3)

        promote_sparse3, promote_dense3= self.prompt_encoder3(points=None,boxes=None,masks=mask3)



        #extract intermediate outputs for deep supervision to prevent model overfitting on the detail enhancement module.
        
        coarse_mask3, feature3= self.mask_decoder3(img_feature3, 
                                                img_pe,
                                                promote_sparse3,
                                                promote_dense3)
        
        


        
        # fusion part 2

        mask_feature_list2 = [mask_feature3,mask_feature4,feature3]
        mask_feature2 = self.fme2(mask_feature_list2)

        img_list2 = [img_feature3, img_feature2]
        img_feature2 = self.fie2(img_list2)


   
        #embedding mask features
        mask_feature2 = self.mask_embedding2(mask_feature2)
        mask2 = self.mask_get2(mask_feature2)

        promote_sparse2, promote_dense2= self.prompt_encoder2(points=None,boxes=None,masks=mask2)



        #extract intermediate outputs for deep supervision to prevent model overfitting on the detail enhancement module.
        
        coarse_mask2, feature2= self.mask_decoder2(img_feature2, 
                                                img_pe,
                                                promote_sparse2,
                                                promote_dense2)
        
        # fusion part 1
        mask_feature_list1 = [mask_feature2,mask_feature3,mask_feature4,feature2]
        mask_feature1 = self.fme1( mask_feature_list1)

        img_list1 = [img_feature2,img_feature1]
        img_feature1 = self.fie1(img_list1)


   
        #embedding mask features
        mask_feature1 = self.mask_embedding1(mask_feature1)
        mask1 = self.mask_get1(mask_feature1)

        promote_sparse1, promote_dense1= self.prompt_encoder1(points=None,boxes=None,masks=mask1)



        #extract intermediate outputs for deep supervision to prevent model overfitting on the detail enhancement module.
        
        coarse_mask1, feature1= self.mask_decoder1(img_feature1, 
                                                img_pe,
                                                promote_sparse1,
                                                promote_dense1)
        


        


        

        coarse_mask4 = torch.nn.functional.interpolate(coarse_mask4,[self.img_size,self.img_size], mode = 'bilinear', align_corners = False)
        coarse_mask3 = torch.nn.functional.interpolate(coarse_mask3,[self.img_size,self.img_size], mode = 'bilinear', align_corners = False)
        coarse_mask2 = torch.nn.functional.interpolate(coarse_mask2,[self.img_size,self.img_size], mode = 'bilinear', align_corners = False)
        coarse_mask1 = torch.nn.functional.interpolate(coarse_mask1,[self.img_size,self.img_size], mode = 'bilinear', align_corners = False)

        return self.sigmoid(coarse_mask4),self.sigmoid(coarse_mask3),self.sigmoid(coarse_mask2),self.sigmoid(coarse_mask1)

    
class PositionEmbeddingRandom(nn.Module):
    """
    Positional encoding using random spatial frequencies.
    """

    def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = None) -> None:
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((2, num_pos_feats)),
        )

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """Positionally encode points that are normalized to [0,1]."""
        # assuming coords are in [0, 1]^2 square and have d_1 x ... x d_n x 2 shape
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        # outputs d_1 x ... x d_n x C shape
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, size: Tuple[int, int]) -> torch.Tensor:
        """Generate positional encoding for a grid of the specified size."""
        h, w = size
        device: Any = self.positional_encoding_gaussian_matrix.device
        grid = torch.ones((h, w), device=device, dtype=torch.float32)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w

        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        return pe.permute(2, 0, 1)  # C x H x W

    def forward_with_coords(
        self, coords_input: torch.Tensor, image_size: Tuple[int, int]
    ) -> torch.Tensor:
        """Positionally encode points that are not normalized to [0,1]."""
        coords = coords_input.clone()
        coords[:, :, 0] = coords[:, :, 0] / image_size[1]
        coords[:, :, 1] = coords[:, :, 1] / image_size[0]
        return self._pe_encoding(coords.to(torch.float))  # B x N x C

class partial:
    """New function with partial application of the given arguments
    and keywords.
    """

    __slots__ = "func", "args", "keywords", "__dict__", "__weakref__"

    def __new__(cls, func, /, *args, **keywords):
        if not callable(func):
            raise TypeError("the first argument must be callable")

        if hasattr(func, "func"):
            args = func.args + args
            keywords = {**func.keywords, **keywords}
            func = func.func

        self = super(partial, cls).__new__(cls)

        self.func = func
        self.args = args
        self.keywords = keywords
        return self

    def __call__(self, /, *args, **keywords):
        keywords = {**self.keywords, **keywords}
        return self.func(*self.args, *args, **keywords)

    @recursive_repr()
    def __repr__(self):
        qualname = type(self).__qualname__
        args = [repr(self.func)]
        args.extend(repr(x) for x in self.args)
        args.extend(f"{k}={v!r}" for (k, v) in self.keywords.items())
        if type(self).__module__ == "functools":
            return f"functools.{qualname}({', '.join(args)})"
        return f"{qualname}({', '.join(args)})"

    def __reduce__(self):
        return type(self), (self.func,), (self.func, self.args,
               self.keywords or None, self.__dict__ or None)

    def __setstate__(self, state):
        if not isinstance(state, tuple):
            raise TypeError("argument to __setstate__ must be a tuple")
        if len(state) != 4:
            raise TypeError(f"expected 4 items in state, got {len(state)}")
        func, args, kwds, namespace = state
        if (not callable(func) or not isinstance(args, tuple) or
           (kwds is not None and not isinstance(kwds, dict)) or
           (namespace is not None and not isinstance(namespace, dict))):
            raise TypeError("invalid partial state")

        args = tuple(args) # just in case it's a subclass
        if kwds is None:
            kwds = {}
        elif type(kwds) is not dict: # XXX does it need to be *exactly* dict?
            kwds = dict(kwds)
        if namespace is None:
            namespace = {}

        self.__dict__ = namespace
        self.func = func
        self.args = args
        self.keywords = kwds

if __name__ == "__main__":
    model = SAM().cuda()

    x = torch.randn(4,3,512,512).cuda()
    y = model(x)

    print(y.shape)