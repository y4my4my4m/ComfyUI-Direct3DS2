# -*- coding: utf-8 -*-
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
from .unet3d import UNet3DModel
import trimesh
from tqdm import tqdm
from skimage import measure
from ...modules.utils import convert_module_to_f16, convert_module_to_f32


def adaptive_conv(inputs,weights):
    padding = (1, 1, 1, 1, 1, 1)
    padded_input = F.pad(inputs, padding, mode="constant", value=0)
    output = torch.zeros_like(inputs)
    size=inputs.shape[-1]
    for i in range(3):
        for j in range(3):
            for k in range(3):
                output=output+padded_input[:,:,i:i+size,j:j+size,k:k+size]*weights[:,i*9+j*3+k:i*9+j*3+k+1]
    return output

def adaptive_block(inputs,conv,weights_=None):
    if weights_ != None:
        weights = conv(weights_)
    else:
        weights = conv(inputs)
    weights = F.normalize(weights, dim=1, p=1)
    for i in range(3):
        inputs = adaptive_conv(inputs, weights)
    return inputs

class GeoDecoder(nn.Module):

    def __init__(self, 
                 n_features: int,
                 hidden_dim: int = 32, 
                 num_layers: int = 4, 
                 use_sdf: bool = False,
                 activation: nn.Module = nn.ReLU):
        super().__init__()
        self.use_sdf=use_sdf
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            activation(),
            *itertools.chain(*[[
                nn.Linear(hidden_dim, hidden_dim),
                activation(),
            ] for _ in range(num_layers - 2)]),
            nn.Linear(hidden_dim, 8),
        )

        # init all bias to zero
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.net(x)
        return x


class Voxel_RefinerXL(nn.Module):
    def __init__(self,
                in_channels: int = 1,
                out_channels: int = 1,
                layers_per_block: int = 2,
                layers_mid_block: int = 2,
                patch_size: int = 192,
                res: int = 512,
                use_checkpoint: bool=False,
                use_fp16: bool = False):

        super().__init__()

        self.unet3d1 = UNet3DModel(in_channels=16, out_channels=8, use_conv_out=False,
                                   layers_per_block=layers_per_block, layers_mid_block=layers_mid_block, 
                                   block_out_channels=(8, 32, 128,512), norm_num_groups=4, use_checkpoint=use_checkpoint)
        self.conv_in = nn.Conv3d(in_channels, 8, kernel_size=3, padding=1)
        self.latent_mlp = GeoDecoder(32)
        self.adaptive_conv1 = nn.Sequential(nn.Conv3d(8, 8, kernel_size=3, padding=1),
                                            nn.ReLU(),
                                            nn.Conv3d(8, 27, kernel_size=3, padding=1, bias=False))
        self.adaptive_conv2 = nn.Sequential(nn.Conv3d(8, 8, kernel_size=3, padding=1),
                                            nn.ReLU(),
                                            nn.Conv3d(8, 27, kernel_size=3, padding=1, bias=False))
        self.adaptive_conv3 = nn.Sequential(nn.Conv3d(8, 8, kernel_size=3, padding=1),
                                            nn.ReLU(),
                                            nn.Conv3d(8, 27, kernel_size=3, padding=1, bias=False))
        self.mid_conv = nn.Conv3d(8, 8, kernel_size=3, padding=1)
        self.conv_out = nn.Conv3d(8, out_channels, kernel_size=3, padding=1)
        self.patch_size = patch_size
        self.res = res

        self.use_fp16 = use_fp16
        self.dtype = torch.float16 if use_fp16 else torch.float32
        if use_fp16:
            self.convert_to_fp16()

    def convert_to_fp16(self) -> None:
        """
        Convert the torso of the model to float16.
        """
        # self.blocks.apply(convert_module_to_f16)
        self.apply(convert_module_to_f16)

    def run(self,
            reconst_x,
            feat, 
            mc_threshold=0,
        ):
        batch_size = int(reconst_x.coords[..., 0].max()) + 1
        sparse_sdf, sparse_index = reconst_x.feats, reconst_x.coords
        sparse_feat = feat.feats
        device = sparse_sdf.device
        dtype = sparse_sdf.dtype
        res = self.res

        sdfs = []
        for i in range(batch_size):
            idx = sparse_index[..., 0] == i
            sparse_sdf_i, sparse_index_i = sparse_sdf[idx].squeeze(-1),  sparse_index[idx][..., 1:]
            sdf = torch.ones((res, res, res)).to(device).to(dtype)
            sdf[sparse_index_i[..., 0], sparse_index_i[..., 1], sparse_index_i[..., 2]] = sparse_sdf_i
            sdfs.append(sdf.unsqueeze(0))

        sdfs = torch.stack(sdfs, dim=0)
        feats = torch.zeros((batch_size, sparse_feat.shape[-1], res, res, res), 
                            device=device, dtype=dtype)
        feats[sparse_index[...,0],:,sparse_index[...,1],sparse_index[...,2],sparse_index[...,3]] = sparse_feat
        
        N = sdfs.shape[0]
        outputs = torch.ones([N,1,res,res,res], dtype=dtype, device=device)
        stride = 160
        patch_size = self.patch_size
        step = 3
        sdfs = sdfs.to(dtype)
        feats = feats.to(dtype)
        patchs=[]
        for i in range(step):
            for j in range(step):
                for k in tqdm(range(step)):
                    sdf = sdfs[:, :, stride * i: stride * i + patch_size,
                               stride * j: stride * j + patch_size,
                               stride * k: stride * k + patch_size]
                    crop_feats = feats[:, :, stride * i: stride * i + patch_size, 
                                       stride * j: stride * j + patch_size, 
                                       stride * k: stride * k + patch_size]
                    inputs = self.conv_in(sdf)
                    crop_feats = self.latent_mlp(crop_feats.permute(0,2,3,4,1)).permute(0,4,1,2,3)
                    inputs = torch.cat([inputs, crop_feats],dim=1)
                    mid_feat = self.unet3d1(inputs)  
                    mid_feat = adaptive_block(mid_feat, self.adaptive_conv1)
                    mid_feat = self.mid_conv(mid_feat)
                    mid_feat = adaptive_block(mid_feat, self.adaptive_conv2)
                    final_feat = self.conv_out(mid_feat)
                    final_feat = adaptive_block(final_feat, self.adaptive_conv3, weights_=mid_feat)
                    output = F.tanh(final_feat)
                    patchs.append(output)
        weights = torch.linspace(0, 1, steps=32, device=device, dtype=dtype)
        lines=[]
        for i in range(9):
            out1 = patchs[i * 3]
            out2 = patchs[i * 3 + 1]
            out3 = patchs[i * 3 + 2]
            line = torch.ones([N, 1, 192, 192,res], dtype=dtype, device=device) * 2
            line[:, :, :, :, :160] = out1[:, :, :, :, :160]
            line[:, :, :, :, 192:320] = out2[:, :, :, :, 32:160]
            line[:, :, :, :, 352:] = out3[:, :, :, :, 32:]
            
            line[:,:,:,:,160:192] = out1[:,:,:,:,160:] * (1-weights.reshape(1,1,1,1,-1)) + out2[:,:,:,:,:32] * weights.reshape(1,1,1,1,-1)
            line[:,:,:,:,320:352] = out2[:,:,:,:,160:] * (1-weights.reshape(1,1,1,1,-1)) + out3[:,:,:,:,:32] * weights.reshape(1,1,1,1,-1)
            lines.append(line)
        layers=[]
        for i in range(3):
            line1 = lines[i*3]
            line2 = lines[i*3+1]
            line3 = lines[i*3+2]
            layer = torch.ones([N,1,192,res,res], device=device, dtype=dtype) * 2
            layer[:,:,:,:160] = line1[:,:,:,:160]
            layer[:,:,:,192:320] = line2[:,:,:,32:160]
            layer[:,:,:,352:] = line3[:,:,:,32:]
            layer[:,:,:,160:192] = line1[:,:,:,160:]*(1-weights.reshape(1,1,1,-1,1))+line2[:,:,:,:32]*weights.reshape(1,1,1,-1,1)
            layer[:,:,:,320:352] = line2[:,:,:,160:]*(1-weights.reshape(1,1,1,-1,1))+line3[:,:,:,:32]*weights.reshape(1,1,1,-1,1)
            layers.append(layer)
        outputs[:,:,:160] = layers[0][:,:,:160]
        outputs[:,:,192:320] = layers[1][:,:,32:160]
        outputs[:,:,352:] = layers[2][:,:,32:]
        outputs[:,:,160:192] = layers[0][:,:,160:]*(1-weights.reshape(1,1,-1,1,1))+layers[1][:,:,:32]*weights.reshape(1,1,-1,1,1)
        outputs[:,:,320:352] = layers[1][:,:,160:]*(1-weights.reshape(1,1,-1,1,1))+layers[2][:,:,:32]*weights.reshape(1,1,-1,1,1)
        # outputs = -outputs

        meshes = []
        for i in range(outputs.shape[0]):
            vertices, faces, _, _ = measure.marching_cubes(outputs[i, 0].cpu().numpy(), level=mc_threshold, method='lewiner')
            vertices = vertices / res * 2 - 1
            meshes.append(trimesh.Trimesh(vertices, faces))
        
        return meshes


