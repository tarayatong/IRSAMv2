"""Base model utilities for SAM-SPL.

This module provides adapter components that connect a SAM-style image encoder
with the project's custom decoder and mask heads. The key elements are:
- `SamAdaptor`: adapts SAM encoder outputs into multi-scale mask outputs.
- `DynamicConvBlock` and `build_dynamic_conv`: helpers to build dynamic
    convolutional blocks that optionally downsample based on stage count.

Only documentation strings have been added/updated; no computational logic
is changed by these edits.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from sam_spl.base_layer import DenseBlock, SELayer, VGGBlock, Res_CBAM_block
from sam_spl.UpBlock_layer import UpBlock_attention
from sam_spl.hieradet import Hiera
from sam_spl.pmt_generator import MultiScaleBlock, MultiScalePositionalEncoder

from sam_spl.transformer import TwoWayTransformer
from sam_spl.utils import LayerNorm2d, MLP
from sam_spl.image_encoder import ImageEncoder
from sam_spl.pmd import DySample

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    try:
        if classname.find("Conv") != -1:
            torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
        elif classname.find("Linear") != -1:
            torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
        elif classname.find("BatchNorm") != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)
    except AttributeError:
        pass

class DynamicConvBlock(nn.Module):
    """Dynamic convolutional block with optional downsampling stages.

    The block always starts with a 1x1 convolution + BatchNorm + GELU for
    channel fusion. Depending on the provided stage index ``n`` (must be
    ``<= 4``) the block will append up to ``4 - n`` 2x2 stride-2 convolutions
    each followed by GELU to perform additional downsampling.

    Parameters
    ----------
    skip_channel : int
        Number of input/output channels.
    n : int
        Stage index used to determine how many downsampling convolutions to
        append. Must satisfy ``n <= 4``.
    """
    def __init__(self, skip_channel, n):
        super().__init__()
        self.skip_channel = skip_channel
        if n > 4:
            raise ValueError(f"n must be <= 4, but got {n}")
        self.n = n
        self.conv_blocks = self._build_blocks()
    
    def _build_blocks(self):
        """Build and return the sequential convolutional layers.

        The returned module always begins with a 1x1 conv + BN + GELU and then
        contains (4 - n) blocks of 2x2 stride-2 conv + GELU.
        """
        layers = [
            nn.Conv2d(self.skip_channel, self.skip_channel, kernel_size=1, stride=1),
            nn.BatchNorm2d(self.skip_channel),
            nn.GELU()
        ]

        num_extra_blocks = 4 - self.n
        
        for _ in range(num_extra_blocks):
            layers.extend([
                nn.Conv2d(self.skip_channel, self.skip_channel, kernel_size=2, stride=2),
                nn.GELU()
            ])

        return nn.Sequential(*layers)
    
    def forward(self, x):
        """Forward pass through the constructed convolutional sequence.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor with shape (B, C, H, W), where C == ``skip_channel``.

        Returns
        -------
        torch.Tensor
            Output tensor after the conv sequence.
        """
        return self.conv_blocks(x)
    
def build_dynamic_conv(skip_channel: int, n: int) -> nn.Sequential:
    """Functional helper that returns an ``nn.Sequential`` with the same
    rules as :class:`DynamicConvBlock`.

    Parameters
    ----------
    skip_channel : int
        Number of channels.
    n : int
        Stage index; controls how many downsampling convs are added.

    Returns
    -------
    nn.Sequential
        Sequential module implementing the dynamic conv pattern.
    """
    if n > 4:
        raise ValueError(f"n must be <= 4, but got {n}")
    layers = [
        nn.Conv2d(skip_channel, skip_channel, kernel_size=1, stride=1),
        nn.BatchNorm2d(skip_channel),
        nn.GELU()
    ]

    for _ in range(4 - n):
        layers.extend([
            nn.Conv2d(skip_channel, skip_channel, kernel_size=2, stride=2),
            nn.GELU()
        ])
    
    return nn.Sequential(*layers)


class EmbeddingOptimizer(nn.Module):
    def __init__(self, decoder_dim, dense_low_channel, alpha_chn,num_mask_tokens=2, upsample_scale=2, use_proj=False):
        super().__init__()
        # Upsample modules (list of upsamplers)
        self.upsample = nn.Upsample(scale_factor=upsample_scale, mode='bilinear', align_corners=False)
        self.hypernetworks_mlp = nn.ModuleList([
            MLP(decoder_dim, decoder_dim, dense_low_channel, 3) 
            for _ in range(num_mask_tokens)
        ])
        self.alpha_head = nn.Sequential(
            nn.Conv2d(alpha_chn, 4, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(4),
            nn.GELU(),
            nn.Conv2d(4, 1, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(1),
            nn.GELU(),
        )
        self.use_proj = use_proj
        self.layer_norm = nn.LayerNorm(dense_low_channel)

    def forward(self, embedding, w_t, w_c, alpha_in):
        """
        Args:
            upscaled_embedding: [B, C, H, W]
            w_t: [B, C] Normalized target weight
            w_c: [B, C] Normalized clutter weight
            clt_features: [B, C, H, W] for alpha input
        """
        
        hyper_tgt = self.hypernetworks_mlp[0](w_t)
        hyper_clt = self.hypernetworks_mlp[1](w_c)
        # w_t = F.normalize(hyper_tgt, dim=1, eps=1e-6)
        # w_c = F.normalize(hyper_clt, dim=1, eps=1e-6)
        w_t = self.layer_norm(hyper_tgt)
        w_c = self.layer_norm(hyper_clt)
        w_c_ir = w_c - (w_c*w_t).sum(dim=1, keepdim=True)/ (w_t * w_t).sum(dim=1, keepdim=True) * w_t
        tgt_proj = (w_t[..., None, None] * embedding).sum(dim=1, keepdim=True)
        clt_proj = (w_c[..., None, None] * embedding).sum(dim=1, keepdim=True)
        target_mask = self.upsample(tgt_proj)
        clutter_mask = self.upsample(clt_proj)
        alpha = self.alpha_head(alpha_in)
        corrected_embedding = embedding + alpha * (w_t[..., None, None] * (tgt_proj - clt_proj) + w_c_ir[..., None, None] * (clt_proj - tgt_proj))
        
        return_dict = {
            "target_mask": target_mask,
            "clutter_mask": clutter_mask,
            "corrected_embedding": corrected_embedding,
            "w_t": w_t,
            "w_c": w_c,
            "alpha": alpha,
        }
        
        return return_dict

class SamAdaptor(nn.Module):
    """Adapter that converts SAM encoder outputs into multi-scale masks.

    The adapter fuses SAM multi-scale features with an auxiliary dense branch
    and optionally uses a transformer-based decoder (``decoder_transformer``)
    plus a hypernetwork to produce deep mask features. These features are
    then passed through upsampling blocks and skip connections to produce
    multi-scale mask tensors.

    See the constructor for parameter descriptions.
    """
    def __init__(
        self,
        sam_encoder: nn.Module,
        decoder_transformer: nn.Module,
        backbone_channel_list: list[int] = [384, 192, 96],
        stages=[1, 2, 7],
        block="res",
        dense_low_channels: list[int] = [96, 48, 24],
        num_mask_tokens=1,
        use_sam_decoder=True,
        pe_inch=[24, 48, 96],
        use_alpha=True,
    ):
        super().__init__()
        self.use_sam_decoder = use_sam_decoder
        self.use_alpha = use_alpha
        self.num_mask_tokens = num_mask_tokens
        self.dense_low_channels = backbone_channel_list + dense_low_channels[1:]
        self.pe_inch = pe_inch
        if block == "res":
            _block = Res_CBAM_block
        elif block == "vgg":
            _block = VGGBlock
        elif block == "dense":
            _block = DenseBlock

        _block = self._select_block(block)
        self.image_encoder = ImageEncoder(
            sam_encoder,
            _block=_block,
            backbone_channel_list=backbone_channel_list,
            stages=stages,
        )
        
        self.skip_channel_gen = dense_low_channels
        self.mask_channel_gen = [ch // 2 for ch in dense_low_channels]
        if self.use_sam_decoder:
            self.up_decoders, self.skip_convs = self._initialize_up_decoders_and_skip_convs()
        else:
            self.up_decoders, self.skip_convs = self._initialize_up_decoders_and_skip_convs2()

        self.reduction_convs = self._initialize_reduction_convs()

        # Projection block to match the dimensions of the dense low channels
        if self.use_sam_decoder:
            self.decoder_dim = decoder_transformer.embedding_dim
            self.decoder_transformer = decoder_transformer
            self.mask_token = nn.Embedding(self.num_mask_tokens, self.decoder_dim)
            self.output_upscaling = nn.Sequential(
                DySample(self.decoder_dim),
                # nn.ConvTranspose2d(self.decoder_dim, self.decoder_dim // 4, kernel_size=2, stride=2),
                # nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(self.decoder_dim, dense_low_channels[0], kernel_size=3, padding=1, stride=1),
                # nn.ConvTranspose2d(self.decoder_dim // 4, dense_low_channels[0], kernel_size=2, stride=2),
                nn.BatchNorm2d(dense_low_channels[0]),
                nn.GELU(),
            )
            
            # Initial weight generators from hs
            self.output_hypernetworks_mlp = nn.ModuleList([
                MLP(self.decoder_dim, self.decoder_dim, dense_low_channels[0], 3) 
                for _ in range(self.num_mask_tokens)
            ])
            
            # Use the new EmbeddingOptimizer
            self.embedding_optimizer = EmbeddingOptimizer(self.decoder_dim, dense_low_channels[0], dense_low_channels[0], self.num_mask_tokens, 4, True)
            self.embedding_optimizer_up = nn.ModuleList([
                EmbeddingOptimizer(dense_low_channels[i], dense_low_channels[i]//2, 1, self.num_mask_tokens, 2**(len(dense_low_channels)-1-i)) for i in range(len(dense_low_channels))
            ])
            self.image_pe_encoder = MultiScalePositionalEncoder(
                in_chans=pe_inch,
                down_times=[len(self.dense_low_channels) - i - 2 for i in range(len(pe_inch))],
            )
            self.proj_block = build_dynamic_conv(self.skip_channel_gen[0], len(stages))
            self.deep_conv_block = nn.Sequential(
                # nn.Conv2d(dense_low_channels[0], dense_low_channels[0], kernel_size=3, padding=1, stride=2),
                # nn.BatchNorm2d(dense_low_channels[0]),
                # nn.GELU(),                
                nn.Conv2d(backbone_channel_list[-2], self.decoder_dim, kernel_size=1, stride=1),
                LayerNorm2d(self.decoder_dim),
                nn.GELU(),
                nn.Conv2d(self.decoder_dim, self.decoder_dim, kernel_size=1, stride=1),
                nn.GELU(),
            )
        else:
            self.proj_block = nn.Sequential(
                nn.Conv2d(self.dense_low_channels[0], self.dense_low_channels[1], kernel_size=1, stride=1),
                nn.BatchNorm2d(self.dense_low_channels[1]),
                nn.GELU(),
                nn.Conv2d(self.dense_low_channels[1], self.dense_low_channels[1], kernel_size=1, stride=1),
                nn.GELU(),
            )

        
        self.apply(weights_init_kaiming)

    def _select_block(self, block: str) -> nn.Module:
        """Select an encoder block class by name.

        Parameters
        ----------
        block : str
            One of 'res', 'vgg', or 'dense'.

        Returns
        -------
        nn.Module
            The block class corresponding to the provided name. Defaults to
            :class:`Res_CBAM_block` if an unknown name is given.
        """
        blocks = {"res": Res_CBAM_block, "vgg": VGGBlock, "dense": DenseBlock}
        return blocks.get(block, Res_CBAM_block)  # Default to Res_CBAM_block if not found

    def _initialize_up_decoders_and_skip_convs(self) -> tuple:
        """Initialize up-sampling decoder blocks and skip convolution modules.

        Returns
        -------
        tuple
            A tuple ``(up_decoders, skip_convs)`` where each element is an
            ``nn.ModuleList`` containing the corresponding modules.
        """

        up_decoders = nn.ModuleList()
        skip_convs = nn.ModuleList()
        for in_ch in self.skip_channel_gen:
            up_decoders.append(UpBlock_attention(in_ch, in_ch // 2))
            skip_convs.append(
                nn.Sequential(
                    SELayer(in_ch),
                    nn.BatchNorm2d(in_ch),
                    nn.GELU(),
                    nn.Conv2d(in_ch, in_ch, kernel_size=1, stride=1),
                    nn.GELU(),
                )
            )

        return up_decoders, skip_convs

    def _initialize_up_decoders_and_skip_convs2(self) -> tuple:
        """Variant initialization that uses ``self.dense_low_channels[1:]``
        as the input channel list for skip modules.

        Returns
        -------
        tuple
            ``(up_decoders, skip_convs)`` as ``nn.ModuleList`` objects.
        """

        up_decoders = nn.ModuleList()
        skip_convs = nn.ModuleList()
        for in_ch in self.dense_low_channels[1:]:
            up_decoders.append(UpBlock_attention(in_ch, in_ch // 2))
            skip_convs.append(
                nn.Sequential(
                    SELayer(in_ch),
                    nn.BatchNorm2d(in_ch),
                    nn.GELU(),
                    nn.Conv2d(in_ch, in_ch, kernel_size=1, stride=1),
                    nn.GELU(),
                )
            )
        return up_decoders, skip_convs

    def _initialize_reduction_convs(self) -> nn.ModuleList:
        """Create 1x1 conv layers to reduce deep feature maps to single-channel
        masks.

        Returns
        -------
        nn.ModuleList
            ModuleList of ``nn.Conv2d(ch, 1, kernel_size=1)`` for each mask
            generation channel.
        """
        reducttion_conv = nn.ModuleList()
        for ch in self.mask_channel_gen:
            reducttion_conv.append(nn.Conv2d(ch, 1, kernel_size=1, stride=1))
        return reducttion_conv

    def _load_sam_checkpoint(self, ckpt_path):
        """Load SAM checkpoint parameters from the given path, if provided.

        The function attempts to load the checkpoint on CPU and then calls
        ``self.load_state_dict`` with ``strict=False`` to allow partial
        compatibility.
        """
        if ckpt_path is not None:
            sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)["model"]
            # sd = {k.replace("sam_mask_decoder.transformer", "decoder_transformer"): v for k, v in sd.items()}
            unexpected_keys, missing_keys = self.load_state_dict(sd, strict=False)
        print("Finish loading sam2 checkpoint")

    def _freeze_encoder(self):
        """Freeze parts of the image encoder while leaving promote generator
        and decoder transformer parameters trainable.

        This helper sets ``requires_grad`` appropriately based on parameter
        name patterns.
        """
        for name, para in self.named_parameters():
            if "image_encoder.trunk" in name and "promote_genertor" not in name:
                para.requires_grad_(False)
            if "image_encoder.neck" in name and "promote_genertor" not in name:
                para.requires_grad_(True)
            elif "decoder_transformer" in name:
                para.requires_grad_(True)

    def print_param_quantity(self):
        """Print a simple summary of parameter quantities (in millions).

        The printed value reports the total model parameter count adjusted to
        exclude the frozen encoder trunk while including the promote generator
        parameters.
        """
        trunk_param = sum(p.numel() for p in self.image_encoder.trunk.parameters()) / 1_000_000
        pmtg_param = sum(p.numel() for p in self.image_encoder.trunk.promote_genertor.parameters()) / 1_000_000
        all_param = sum(p.numel() for p in self.parameters()) / 1_000_000
        print(f"The parameter number of the model is {all_param - trunk_param + pmtg_param:.2f}M")

    def _process_deep_features(self, features: dict) -> list:
        """Process deep features with the decoder transformer and hypernetwork.

        This method:
        - Selects the deepest available image embeddings.
        - Computes multi-scale positional encodings.
        - Runs the decoder transformer producing token features and a
          transformed source representation.
        - Applies a hypernetwork MLP to obtain scaling factors which are used
          to modulate the upscaled embedding.
        - Projects and refines the result with ``proj_block`` and then uses
          upsampling decoder blocks and skip connections to create a list of
          multi-scale deep feature maps (not yet reduced to single-channel
          masks).

        Parameters
        ----------
        features : dict
            Dictionary returned by :class:`ImageEncoder` expected to contain
            keys ``"dense_embeds"`` and ``"sam_backbone_embeds"``.

        Returns
        -------
        list
            List of multi-scale deep features.
        """
        masks = []
        dense_features, sam_feature = features["dense_embeds"], features["sam_backbone_embeds"]
        clt_features = features["clt_embeds"]
        try:
            image_embeddings = sam_feature[-2]
        except IndexError:
            image_embeddings = dense_features[-1]

        pe_input = dense_features + sam_feature
        pe_input = pe_input[:len(self.pe_inch)]
        image_pe = self.image_pe_encoder(pe_input)

        B, C, W, H = image_embeddings.shape
        src = self.deep_conv_block(image_embeddings)
        token = self.mask_token.weight.unsqueeze(0).expand(B, -1, -1)
        hs, src = self.decoder_transformer(src, image_pe, token)
        src = src.transpose(1, 2).contiguous().view(B, self.decoder_dim, W, H)
        upscaled_embedding = self.output_upscaling(src)
        if self.use_alpha:
            return_dicts=[]
            alpha_in = upscaled_embedding + clt_features[1]
            lowest_dict = self.embedding_optimizer(upscaled_embedding, hs[:,0,:], hs[:,1,:], alpha_in)
            corrected_embedding = lowest_dict["corrected_embedding"]
            return_dicts.append(lowest_dict)
            deep_feat = self.proj_block(corrected_embedding)
            deep_dict = lowest_dict
            for i, (feature_map, skip_conv, up_decoder) in enumerate(zip(dense_features[::-1], self.skip_convs, self.up_decoders)):
                deep_feat = up_decoder(deep_feat, skip_conv(feature_map))
                alpha_in = F.interpolate(deep_dict['alpha'], deep_feat.shape[-2:], mode='bilinear', align_corners=False)
                deep_dict = self.embedding_optimizer_up[i](deep_feat, deep_dict["w_t"], deep_dict["w_c"], alpha_in)
                deep_feat = deep_dict["corrected_embedding"]
                deep_mask = (deep_dict["w_t"][..., None, None] * deep_feat).sum(dim=1, keepdim=True)
                masks.append(deep_mask)
                return_dicts.append(deep_dict)
        else:
            corrected_embedding = upscaled_embedding
            return_dicts = None
            
            deep_feat = self.proj_block(corrected_embedding)
            for i, (feature_map, skip_conv, up_decoder) in enumerate(zip(dense_features[::-1], self.skip_convs, self.up_decoders)):
                deep_feat = up_decoder(deep_feat, skip_conv(feature_map))
                masks.append(deep_feat)
            
        return masks, return_dicts

    def _process_deep_features2(self, features: dict) -> list:
        """Alternative processing path used when SAM decoder is not enabled.

        This simpler path concatenates dense and SAM features, reverses the
        order and runs projection + upsampling modules to produce multi-scale
        deep features.
        """
        masks = []
        dense_features, sam_feature = features["dense_embeds"], features["sam_backbone_embeds"]

        dense_features = (dense_features + sam_feature)[::-1]

        deep_feat = self.proj_block(dense_features[0])

        for i, (feature_map, skip_conv, up_decoder) in enumerate(zip(dense_features[1:], self.skip_convs, self.up_decoders)):
            deep_feat = up_decoder(deep_feat, skip_conv(feature_map))
            masks.append(deep_feat)

        return masks[-len(self.mask_channel_gen):]

    def _generate_masks(self, deep_feats: list, image_size: list[int, int]) -> list:
        """Reduce multi-scale deep features to single-channel masks and resize.

        For each deep feature map, this method resizes it to ``image_size``
        using bilinear interpolation and applies a 1x1 conv to produce the
        final single-channel mask tensor.
        """
        masks = []
        for mask_conv, feature_map in zip(self.reduction_convs[::-1], deep_feats[::-1]):
            mask_0 = F.interpolate(feature_map, image_size, mode="bilinear", align_corners=False)
            # mask_0 = mask_conv(mask_0)
            masks.append(mask_0)

        return masks

    def forward(self, x: torch.tensor):
        """Forward method: produce multi-scale masks from input images.

        Parameters
        ----------
        x : torch.Tensor
            Input image tensor of shape (B, C, H, W).

        Returns
        -------
        list[torch.Tensor]
            List of mask tensors, each shaped (B, 1, H, W).
        """
        out_image_size = x.shape[-2:]
        features = self.image_encoder(x)
        if self.use_sam_decoder:
            masks, return_dict = self._process_deep_features(features)
        else:
            masks = self._process_deep_features2(features)

        masks = self._generate_masks(masks, out_image_size)
        return masks, return_dict


def make_adaptor(
    backbone_channel_list: list[int] = [384, 192, 96],
    dense_low_channels: list[int] = [96, 48, 24],
    stages: list[int] = [1, 2, 7],
    global_att_blocks: list[int] = [5, 7, 9],
    window_pos_embed_bkg_spatial_size: list[int] = [7, 7],
    window_spec: list[int] = [8, 4, 16],
    block: str = "res",
    embed_dim=96,
    use_sam_decoder=True,
    pe_inch=[24, 48, 96],
    sam_ckpt_path=None,
    num_mask_tokens=2,
    use_alpha=True,
):
    """_summary_

    Args:
        backbone_channel_list (list[int], optional): The list of encoder channels. Defaults to [384, 192, 96].
        out_dim (int, optional): Number of masks in the final output. Defaults to 4.
        down_times (int, optional): Times of feature map dimensionality drop for shallow feature extraction. Defaults to 3.
        stages (list[int], optional): The stages of hieradet. Defaults to [1, 2, 7].
        global_att_blocks (list[int], optional): global attention blocks. Defaults to [5, 7, 9].
        window_pos_embed_bkg_spatial_size (list[int], optional): window size. Defaults to [7, 7].
        window_spec (list[int], optional): window spec. Defaults to [8, 4, 16].
        block (str, optional): The type of block used in encoder. Defaults to "res".

    Returns:
        nn.Module: sam adaptor
    """
    promote_generator = MultiScaleBlock(stages=stages, embed_dim=embed_dim)

    sam_encoder = Hiera(
        promote_genertor=promote_generator,
        embed_dim=embed_dim,
        num_heads=1,
        stages=stages,
        global_att_blocks=global_att_blocks,
        window_pos_embed_bkg_spatial_size=window_pos_embed_bkg_spatial_size,
        window_spec=window_spec,
    )

    decoder_transformer = TwoWayTransformer(
        depth=2,
        embedding_dim=256,
        mlp_dim=2048,
        num_heads=8,
    )

    predictor = SamAdaptor(
        sam_encoder=sam_encoder,
        decoder_transformer=decoder_transformer,
        backbone_channel_list=backbone_channel_list,
        stages=stages,
        block=block,
        dense_low_channels=dense_low_channels,
        use_sam_decoder=use_sam_decoder,
        pe_inch=pe_inch,
        num_mask_tokens=num_mask_tokens,
        use_alpha=use_alpha,
    )
    if sam_ckpt_path is not None:
        predictor._load_sam_checkpoint(sam_ckpt_path)
    return predictor
