from typing import Tuple, List
from torch import Tensor

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
import numpy as np

# from basicsr.utils.registry import ARCH_REGISTRY

# Define a global variable for training mode
# This will be set by the model builder
TRAINING_MODE = True


# non local attentive block
class DeepFeature(nn.Module):
    def __init__(self, dim=48):
        super(DeepFeature, self).__init__()
        
        self.proj1 = nn.Conv2d(dim,dim*2,1,1,0)
        self.dw_conv1 = nn.Conv2d(dim,dim,3,1,1,groups=dim)
        self.dw_conv2 = nn.Conv2d(dim,dim,3,1,1,groups=dim)
    
        self.dwconv3 = nn.Conv2d(dim,dim,3,1,1,groups=dim)
        self.pwconv = nn.Conv2d(dim,dim,1,1,0)
        
        self.gelu = nn.GELU()
        self.down_scale = 8
        
        self.pwconv_xa = nn.Conv2d(dim,dim,1,1,0)
        
    def forward(self, x):
        b,c,h,w = x.shape
        x = self.proj1(x)
        x_a, x_b = x.chunk(2, dim=1)
        
        x_a_avg = F.adaptive_avg_pool2d(x_a, (h//self.down_scale, w//self.down_scale))
        x_a_max = F.adaptive_max_pool2d(x_a, (h//self.down_scale, w//self.down_scale))
        x_a_comb = self.dw_conv1(x_a_avg) + self.dw_conv2(x_a_max)
        x_a_comb = self.gelu(self.pwconv_xa(x_a_comb))
        x_a_comb = x_a * F.interpolate(x_a_comb, (h,w), mode='nearest')

        x_b = self.gelu(self.pwconv(self.dwconv3(x_b)))
        
        return x_a_comb, x_b
        

class MoEBlock(nn.Module):
    def __init__(self, dim=48, num_experts=4, topk=2):
        super(MoEBlock, self).__init__()
        
        self.growth = lambda i: 2**(i+1)
        
        self.deep_feature = DeepFeature(dim)
        self.moe = MoELayer(experts=[Experts(dim=dim, squeezed_dim=self.growth(i)) for i in range(num_experts)], 
                            gate=Router(dim=dim, num_experts=num_experts), 
                            num_expert=topk,
                            )
        

        self.proj_xa = nn.Conv2d(dim,dim,1,1,0)
        self.proj_xb = nn.Conv2d(dim,dim,1,1,0)
        self.proj_out = nn.Conv2d(dim,dim,1,1,0)
        
    def forward(self, x):
        x_a, x_b = self.deep_feature(x)
        x_a = self.proj_xa(x_a)
        x_b = self.proj_xb(x_b)
        
        x_moe = self.moe(x_a, x_b)
        output = self.proj_out(x_moe)
        
        return output

class MoELayer(nn.Module):
    def __init__(self, experts: List[nn.Module], gate: nn.Module, num_expert: int = 2):
        super().__init__()
        
        assert len(experts) > 0
        self.experts = nn.ModuleList(experts)
        self.gate = gate
        self.num_expert = num_expert
        
    def forward(self, x: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
        gate = self.gate(x)
        logits = F.softmax(gate, dim=1)
        topk_weights, topk_experts_indices = torch.topk(logits, self.num_expert)
        
        output = x.clone()
        if self.training:
            experts_weights = torch.zeros_like(logits)
            experts_weights.scatter_(1, topk_experts_indices, logits.gather(1,topk_experts_indices))
            
            for idx, expert in enumerate(self.experts):
                output += expert(x, K) * experts_weights[:, idx:idx+1, None, None]
        else:
            selected_experts = [self.experts[idx] for idx in topk_experts_indices.squeeze(dim=0)]
            for i, exp in enumerate(selected_experts):
                output += exp(x, K) * topk_weights[:, i:i+1, None, None]

        return output

class Router(nn.Module):
    def __init__(self,
                 dim: int,
                 num_experts: int):
        super().__init__()
        
        self.body = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Rearrange('b c 1 1 -> b c'),
            nn.Linear(dim, num_experts, bias=False),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)
    
class Experts(nn.Module):
    def __init__(self,dim,squeezed_dim,expand_rate=1):
        super().__init__()
        self.expand_rate = expand_rate
        self.linear1 = nn.Conv2d(dim,squeezed_dim,1,1,0)
        self.linear2 = nn.Conv2d(dim,squeezed_dim,1,1,0)
        self.linear_final = nn.Conv2d(squeezed_dim * self.expand_rate,dim,1,1,0 )
    

    def forward(self,x_a,x_b):
        x = self.linear1(x_a)
        y = self.linear2(x_b)
        # add? multiply? 
        x_y = x * y
        final = self.linear_final(x_y) 
        
        return final
        
        
'''
Second Module
'''
class DepthFFN(nn.Module):
    def __init__(self, dim, kernel_size=3, expand_rate=2, depthwise=True):
        super(DepthFFN, self).__init__()
        
        self.expand_dim = int(dim * expand_rate)
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        
        
        self.conv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=(self.kernel_size,1), padding=(self.padding,0),  groups=dim if depthwise else 1),
            nn.Conv2d(dim, dim, kernel_size=(1,self.kernel_size), padding=(0,self.padding), groups=dim if depthwise else 1),
        )
        
        self.act = nn.GELU()
        self.proj1 = nn.Conv2d(dim, self.expand_dim, 1, 1, 0)
        self.proj2 = nn.Conv2d(dim, dim, 1, 1, 0)
        
        
    def forward(self,x):
        x = self.proj1(x)
        x = self.act(x)
        x_a, x_b = x.chunk(2, dim=1)
        x_a = self.act(self.conv(x_a))
        
        x_final = self.proj2(x_b * x_a)
        return x_final
    
    

class SpatialFocusBlock(nn.Module):
    def __init__(self, dim, num_experts=4, topk=2):
        super(SpatialFocusBlock, self).__init__()
        
        self.norm1 = LayerNorm(dim, data_format='channels_first')
        self.block = MoEBlock(dim=dim, num_experts=num_experts, topk=topk)
        
    def forward(self, x):
        x = self.block(self.norm1(x)) + x
        return x
    
    
class DepthFocusBlock(nn.Module):
    def __init__(self, dim, kernel_size=3):
        super(DepthFocusBlock, self).__init__()
        
        self.norm1 = LayerNorm(dim, data_format='channels_first')
        self.block1 = DepthFFN(dim, kernel_size=kernel_size, expand_rate=2, depthwise=True)
        

    def forward(self, x):
        x = self.block1(self.norm1(x)) + x
        return x
    
      
# Feature Representation Module
class FRM(nn.Module):
    def __init__(self, dim, num_experts=4, topk=2, kernel_size=3):
        super().__init__()
        
        self.spatial = SpatialFocusBlock(dim, num_experts=num_experts, topk=topk)
        self.depth = DepthFocusBlock(dim, kernel_size=kernel_size)
        
    def forward(self, x):
        x = self.spatial(x)
        x = self.depth(x)
        return x
    
class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x 
        

class Conv3XC(nn.Module):
    """Reparameterizable convolution that fuses multiple branches at inference time"""
    def __init__(self, c_in, c_out, gain1=1, gain2=0, s=1, bias=True, relu=False):
        super(Conv3XC, self).__init__()
        # Use the global training flag instead of checking at runtime
        self.deploy = not TRAINING_MODE
        
        self.c_in = c_in
        self.c_out = c_out
        self.stride = s
        self.has_relu = relu
        self.gain = gain1
        
        if not self.deploy:
            # Create training path components
            self.sk = nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=1, padding=0, stride=s, bias=bias)
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels=c_in, out_channels=c_in * self.gain, kernel_size=1, padding=0, bias=bias),
                nn.Conv2d(in_channels=c_in * self.gain, out_channels=c_out * self.gain, kernel_size=3, stride=s, padding=0, bias=bias),
                nn.Conv2d(in_channels=c_out * self.gain, out_channels=c_out, kernel_size=1, padding=0, bias=bias),
            )
            
            # In training mode, just register buffers for fused weights, NOT modules
            self.register_buffer('weight_fused', torch.zeros(c_out, c_in, 3, 3))
            self.register_buffer('bias_fused', torch.zeros(c_out))
            
            # Initialize fused parameters
            self._update_fused_params()
        else:
            # Just create the efficient inference path (direct 3x3 conv)
            self.eval_conv = nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=3, padding=1, stride=s, bias=bias)

    def _update_fused_params(self):
        """Internal method to update fused parameters without creating optimization targets"""
        if self.deploy:
            return  # No need to update if already in deploy mode
        
        w1 = self.conv[0].weight.data.clone().detach()
        b1 = self.conv[0].bias.data.clone().detach()
        w2 = self.conv[1].weight.data.clone().detach()
        b2 = self.conv[1].bias.data.clone().detach()
        w3 = self.conv[2].weight.data.clone().detach()
        b3 = self.conv[2].bias.data.clone().detach()

        w = F.conv2d(w1.flip(2, 3).permute(1, 0, 2, 3), w2, padding=2, stride=1).flip(2, 3).permute(1, 0, 2, 3)
        b = (w2 * b1.reshape(1, -1, 1, 1)).sum((1, 2, 3)) + b2

        weight_concat = F.conv2d(w.flip(2, 3).permute(1, 0, 2, 3), w3, padding=0, stride=1).flip(2, 3).permute(1, 0, 2, 3)
        bias_concat = (w3 * b.reshape(1, -1, 1, 1)).sum((1, 2, 3)) + b3

        sk_w = self.sk.weight.data.clone().detach()
        sk_b = self.sk.bias.data.clone().detach()
        target_kernel_size = 3

        H_pixels_to_pad = (target_kernel_size - 1) // 2
        W_pixels_to_pad = (target_kernel_size - 1) // 2
        sk_w = F.pad(sk_w, [H_pixels_to_pad, H_pixels_to_pad, W_pixels_to_pad, W_pixels_to_pad])

        self.weight_fused.copy_(weight_concat + sk_w)
        self.bias_fused.copy_(bias_concat + sk_b)

    def forward(self, x):
        if not self.deploy:
            # Training mode - use training path
            pad = 1
            x_pad = F.pad(x, (pad, pad, pad, pad), "constant", 0)
            out = self.conv(x_pad) + self.sk(x)
            
            # Update fused parameters for potential switch to deploy mode
            if not self.training:
                self._update_fused_params()
        else:
            # Inference mode - use pre-fused parameters
            out = F.conv2d(x, self.eval_conv.weight, self.eval_conv.bias, stride=self.stride, padding=1)

        if self.has_relu:
            out = F.leaky_relu(out, negative_slope=0.05)
        return out

    def switch_to_deploy(self):
        """Switch this module to deploy mode"""
        if self.deploy:
            return
            
        # Create the eval_conv module with our fused parameters
        self.eval_conv = nn.Conv2d(
            in_channels=self.c_in,
            out_channels=self.c_out,
            kernel_size=3,
            stride=self.stride,
            padding=1,
            bias=True
        )
        self.eval_conv.weight.data.copy_(self.weight_fused)
        self.eval_conv.bias.data.copy_(self.bias_fused)
        
        # Remove training components
        if hasattr(self, 'sk'):
            delattr(self, 'sk')
        if hasattr(self, 'conv'):
            delattr(self, 'conv')
        if hasattr(self, 'weight_fused'):
            delattr(self, 'weight_fused')
        if hasattr(self, 'bias_fused'):
            delattr(self, 'bias_fused')
            
        self.deploy = True


class EfficientChannelAttention(nn.Module):
    """Efficient Channel Attention module"""
    def __init__(self, dim, kernel_size=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Get channel attention weights
        attn = self.avg_pool(x)
        attn = self.conv(attn.squeeze(-1).transpose(-1, -2))
        attn = attn.transpose(-1, -2).unsqueeze(-1)
        attn = self.sigmoid(attn)
        
        # Apply attention
        return x * attn


class EfficientSpatialAttention(nn.Module):
    """Lightweight spatial attention module"""
    def __init__(self, dim, reduction=8):
        super().__init__()
        self.reduction = max(dim // reduction, 4)
        # Don't use kernel_size parameter with Conv3XC
        self.conv1 = Conv3XC(dim, self.reduction, gain1=1, s=1)
        self.act = nn.GELU()
        self.conv2 = Conv3XC(self.reduction, 1, gain1=1, s=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Get spatial attention weights
        attn = self.conv1(x)
        attn = self.act(attn)
        attn = self.conv2(attn)
        attn = self.sigmoid(attn)
        
        # Apply attention
        return x * attn


class ExpertBlock(nn.Module):
    """Simplified single expert in the mixture"""
    def __init__(self, dim, kernel_size=3):
        super().__init__()
        # Split channels for parallel processing (more expressive with same params)
        self.split_size = dim // 2
        
        # Two parallel paths with different receptive fields
        # Note: Conv3XC has fixed 3x3 kernel internally, so we don't pass kernel_size
        self.path1 = Conv3XC(self.split_size, self.split_size, gain1=1, s=1)
        self.path2 = Conv3XC(self.split_size, self.split_size, gain1=1, s=1)
        
        # Efficient activation
        self.act = nn.GELU()
        
        # Channel attention
        self.attention = EfficientChannelAttention(dim)
        
        # Optional spatial mixing (very lightweight)
        self.spatial_mix = Conv3XC(dim, dim, gain1=1, s=1)
        
    def forward(self, x):
        # Split channels for parallel processing
        x1, x2 = torch.split(x, [self.split_size, self.split_size], dim=1)
        
        # Process through parallel paths
        y1 = self.act(self.path1(x1))
        y2 = self.act(self.path2(x2))
        
        # Feature mixing - channel concat + lightweight projection
        out = torch.cat([y1, y2], dim=1)
        out = self.spatial_mix(out)
        out = self.attention(out)
        
        return out + x


class MixtureOfExpertsBlock(nn.Module):
    """Optimized block that uses multiple experts with a router"""
    def __init__(self, dim, num_experts=3, kernel_size=3, topk=1):
        # Set flag before creating Conv3XC modules
        global TRAINING_MODE
        
        super().__init__()
        # Add efficient layer norm
        self.norm = LayerNorm(dim, data_format='channels_first')
        
        # Create the router before experts to avoid any issues
        # Simpler routing mechanism
        self.router = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, num_experts, kernel_size=1)
        )
        
        # Create fewer experts
        self.experts = nn.ModuleList([
            ExpertBlock(dim, kernel_size) for _ in range(num_experts)
        ])
        
        self.topk = topk
        self.num_experts = num_experts
        # Track expert usage for load balancing
        self.register_buffer('expert_usage', torch.zeros(num_experts))
        self.total_samples = 0
        
    def forward(self, x):
        # Normalize input for stable routing
        x_norm = self.norm(x)
        
        # Use normalized features for routing
        router_scores = self.router(x_norm)
        
        # Efficient per-sample routing during inference
        if not self.training:
            b = x.shape[0]
            expert_indices = router_scores.argmax(dim=1)  # [B, 1, 1]
            out = x.clone()
            
            # Process each sample with its best expert (vectorized approach)
            for expert_idx in range(self.num_experts):
                # Create a mask for samples that should use this expert
                mask = (expert_indices[:, 0, 0] == expert_idx).float().view(b, 1, 1, 1)
                if mask.sum() > 0:  # If any samples use this expert
                    # Only process relevant samples (efficient)
                    expert_out = self.experts[expert_idx](x)
                    out = out * (1 - mask) + expert_out * mask
            
            return out
        else:
            # Original training path with load balancing tracking
            b, e, _, _ = router_scores.shape
            scores = router_scores.view(b, e)
            
            # Get topk experts
            topk_values, topk_indices = torch.topk(scores, self.topk, dim=1)
            topk_values = F.softmax(topk_values, dim=1)
            
            # Track expert usage for load balancing loss
            router_probs = F.softmax(scores, dim=1)
            if self.training:
                # Update expert usage statistics
                with torch.no_grad():
                    self.total_samples += b
                    for i in range(self.num_experts):
                        self.expert_usage[i] += router_probs[:, i].sum().item()
            
            out = torch.zeros_like(x)
            
            # Apply each selected expert and combine with weights
            for i in range(self.topk):
                expert_idx = topk_indices[:, i]
                weight = topk_values[:, i].view(b, 1, 1, 1)
                
                # Apply expert for each sample in batch
                for j in range(b):
                    out[j] += weight[j] * self.experts[expert_idx[j]](x[j].unsqueeze(0)).squeeze(0)
            
            return out
    
    def compute_load_balancing_loss(self):
        """Calculate load balancing loss to encourage uniform expert utilization"""
        if self.total_samples == 0:
            return torch.tensor(0.0, device=self.expert_usage.device)
        
        # Calculate average usage
        avg_usage = self.expert_usage / self.total_samples
        # Ideal distribution is uniform
        target = torch.ones_like(avg_usage) / self.num_experts
        # L2 distance to uniform distribution
        return F.mse_loss(avg_usage, target) * 0.1  # Small weight
    
    def reset_usage_stats(self):
        """Reset usage statistics at the end of an epoch"""
        self.expert_usage.zero_()
        self.total_samples = 0


# @ARCH_REGISTRY.register()
class MOAReparam(nn.Module):
    """Mixture of Attention with Reparameterization"""
    def __init__(self, num_in_ch=3, num_out_ch=3, dim=36, num_blocks=7, num_experts=3,
                kernel_size=3, topk=1, scale=4, upscale_mode='pixelshuffle', deploy=False):
        # Set the global training flag based on deploy parameter at the very beginning
        global TRAINING_MODE
        TRAINING_MODE = not deploy
        
        super(MOAReparam, self).__init__()
        self.dim = dim
        self.num_blocks = num_blocks
        self.scale = scale
        self.deploy = deploy
        self.topk = topk
        
        # Initial feature extraction with Conv3XC
        self.feat_extract = Conv3XC(num_in_ch, dim, gain1=2, s=1, relu=True)
        
        # Main body with mixture of experts blocks
        self.body = nn.ModuleList([
            MixtureOfExpertsBlock(dim, num_experts, min(5, kernel_size), topk) 
            for _ in range(num_blocks)
        ])
        
        # Add efficient block connections for better feature propagation
        self.connector = nn.ModuleList([
            Conv3XC(dim, dim, gain1=1, s=1)
            for _ in range(num_blocks-1)
        ])
        
        # Feature fusion with Conv3XC
        self.fusion = Conv3XC(dim * num_blocks, dim, gain1=2, s=1)
        
        # Upsampling
        if upscale_mode == 'pixelshuffle':
            # Simplified upsampling - single module for each scale
            if scale == 2:
                self.upsampling = nn.Sequential(
                    Conv3XC(dim, dim * 4, gain1=2, s=1),
                    nn.PixelShuffle(2),
                )
            elif scale == 3:
                self.upsampling = nn.Sequential(
                    Conv3XC(dim, dim * 9, gain1=2, s=1),
                    nn.PixelShuffle(3),
                )
            elif scale == 4:
                self.upsampling = nn.Sequential(
                    Conv3XC(dim, dim * 16, gain1=2, s=1),
                    nn.PixelShuffle(4),
                )
            else:
                raise NotImplementedError(f"Scale {scale} is not supported")
        else:
            # Simpler nearest + conv upsampling
            self.upsampling = nn.Sequential(
                nn.Upsample(scale_factor=scale, mode='nearest'),
                Conv3XC(dim, dim, gain1=2, s=1)
            )
        
        # Output layer
        self.output = Conv3XC(dim, num_out_ch, gain1=1, s=1)
    
    def get_load_balancing_loss(self):
        """Calculate total load balancing loss across all MoE blocks"""
        loss = 0
        for block in self.body:
            if hasattr(block, 'compute_load_balancing_loss'):
                loss += block.compute_load_balancing_loss()
        return loss
    
    def reset_expert_stats(self):
        """Reset expert usage statistics for all blocks"""
        for block in self.body:
            if hasattr(block, 'reset_usage_stats'):
                block.reset_usage_stats()
    
    def forward(self, x):
        # Feature extraction
        feat = self.feat_extract(x)
        
        # Store intermediate features for skip connection
        feat_list = []
        
        # Apply expert blocks with improved feature flow
        x_body = feat
        for i, block in enumerate(self.body):
            # Enhanced feature propagation
            if i > 0:
                # Mix with previous block's output for better information flow
                x_body = x_body + self.connector[i-1](feat_list[-1])
            
            x_body = block(x_body)
            feat_list.append(x_body)
        
        # Feature fusion with skip connection
        body_feat = torch.cat(feat_list, dim=1)
        body_out = self.fusion(body_feat)
        body_out = body_out + feat
        
        # Upsampling
        out = self.upsampling(body_out)
        
        # Final output
        out = self.output(out)
        
        return out
    
    def switch_to_deploy(self):
        """Convert model to efficient inference mode"""
        if self.deploy:
            return
            
        print("Switching model to deployment mode...")
        
        # Switch Conv3XC modules to deploy mode
        global TRAINING_MODE
        TRAINING_MODE = False
        
        # Switch all Conv3XC modules to deploy mode
        for m in self.modules():
            if isinstance(m, Conv3XC) and not m.deploy:
                m.switch_to_deploy()
        
        # Optional: Fuse experts in each MixtureOfExpertsBlock
        if hasattr(self, 'fuse_experts') and self.fuse_experts:
            for m in self.modules():
                if isinstance(m, MixtureOfExpertsBlock):
                    if hasattr(m, 'routing_stats'):
                        fused_expert = m.fuse_for_inference()
                        if fused_expert is not None:
                            m.fused_expert = fused_expert
                            m.use_fused_expert = True
        
        self.deploy = True
        print("Model switched to deployment mode.")
    
