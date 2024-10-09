# Copyright 2022 The Nerfstudio Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Model for InstructNeRF2NeRF
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Type

import torch
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from nerfstudio.model_components.losses import (
    L1Loss,
    MSELoss,
    interlevel_loss,
)
from nerfstudio.models.nerfacto import NerfactoModel, NerfactoModelConfig

import pdb
import os
from torchvision import transforms
import torch.nn.functional as F

import random


@dataclass
class InstructNeRF2NeRFModelConfig(NerfactoModelConfig):
    """Configuration for the InstructNeRF2NeRFModel."""
    _target: Type = field(default_factory=lambda: InstructNeRF2NeRFModel)
    use_lpips: bool = True
    """Whether to use LPIPS loss"""
    use_l1: bool = True
    """Whether to use L1 loss"""
    patch_size: int = 32
    """Patch size to use for LPIPS loss."""
    lpips_loss_mult: float = 1.0
    """Multiplier for LPIPS loss."""
    

class InstructNeRF2NeRFModel(NerfactoModel):  # InstructNeRF2NeRFModel -> NerfactoModel -> Model  
    """Model for InstructNeRF2NeRF."""

    config: InstructNeRF2NeRFModelConfig

    def populate_modules(self):
        """Required to use L1 Loss."""
        super().populate_modules()

        if self.config.use_l1:
            self.rgb_loss = L1Loss()
        else:
            self.rgb_loss = MSELoss()
        self.lpips = LearnedPerceptualImagePatchSimilarity()


    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        """
        outputs.keys(): 'rgb' [16384, 3], 'accumulation' [16384, 1], 'depth' [16384, 1], 'weights_list' len=3, [16384, 256, 1], 'ray_samples_list' # len=3, [16384, 256], 
                        'prop_depth_0' [16384, 1], 'prop_depth_1' [16384, 1]
        batch.keys(): 'image' [16384, 3], 'indices' [16384, 3]
        metrics_dict.keys(): 'psnr' 29.4065, 'distortion' 0.0047
        """
        loss_dict = {}
        image = batch["image"].to(self.device) 
        
        # reconstruction loss
        loss_dict["rgb_loss"] = self.rgb_loss(image, outputs["rgb"])  
        if self.config.use_lpips:
            out_patches = (outputs["rgb"].view(-1, self.config.patch_size,self.config.patch_size, 3).permute(0, 3, 1, 2) * 2 - 1).clamp(-1, 1)
            gt_patches = (image.view(-1, self.config.patch_size,self.config.patch_size, 3).permute(0, 3, 1, 2) * 2 - 1).clamp(-1, 1)
            loss_dict["lpips_loss"] = self.config.lpips_loss_mult * self.lpips(out_patches, gt_patches)
        if self.training:
            loss_dict["interlevel_loss"] = self.config.interlevel_loss_mult * interlevel_loss(
                outputs["weights_list"], outputs["ray_samples_list"]
            )
            assert metrics_dict is not None and "distortion" in metrics_dict
            loss_dict["distortion_loss"] = self.config.distortion_loss_mult * metrics_dict["distortion"]
            if self.config.predict_normals:
                # orientation loss for computed normals
                loss_dict["orientation_loss"] = self.config.orientation_loss_mult * torch.mean(
                    outputs["rendered_orientation_loss"]
                )
                # ground truth supervision for normals
                loss_dict["pred_normal_loss"] = self.config.pred_normal_loss_mult * torch.mean(
                    outputs["rendered_pred_normal_loss"]
                )
                
        # discrinimator regularization for novel views, GenN2Ns
        if self.config.use_style and self.config.use_dis:
            if "predict_real" in outputs.keys():
                self.discriminator_loss = self.discriminator_loss.to(self.device)  
                discrim_loss = self.discriminator_loss(outputs["predict_real"], outputs["predict_fake"])  
                loss_dict['discrim_loss'] = 0.1 * discrim_loss  # self.config.discrim_weight      
                
                self.generator_loss = self.generator_loss.to(self.device)
                generator_loss = self.generator_loss(outputs["predict_fake"])  
                loss_dict['generator_loss'] = 0.1 * generator_loss
                
        # novel_views_style_loss from edited nerf rendering results, GenN2N        
        if self.config.use_style and self.config.use_generative:
            # different viewpoint share similar style
            if 'style_code' in outputs.keys():
                style_code = outputs["style_code"]
                style_code2 = style_code.clone()
                style_code2 = torch.cat([style_code2[-1][None], style_code2], dim=0)[:-1]  # [59, 64]  # https://blog.csdn.net/qq_43391414/article/details/123440294
                self.pdist = self.pdist.to(self.device)
                novel_views_style_loss = self.pdist(style_code, style_code2) 
                loss_dict['style_loss'] = 0.1 * novel_views_style_loss.mean()  

            # kl loss
            if 'edited_style_code' in outputs.keys():
                self.kl_loss = self.kl_loss.to(self.device)  # https://pytorch.org/docs/stable/generated/torch.nn.KLDivLoss.html
                edited_style_code = outputs["edited_style_code"]  # s0

                input = F.log_softmax(edited_style_code, dim=1).to(self.device)
                target = F.softmax(torch.randn(*edited_style_code.shape), dim=1).to(self.device)
                
                kl_loss = self.kl_loss(input, target)
                loss_dict['kl_loss'] = kl_loss


        # use_contrastive_style on style z (c0,ci; ci,si)  
        if self.config.use_style and self.config.use_contrastive_style:
            self.contrastive_style_loss = self.contrastive_style_loss.to(self.device)
            if 'style_code_si' in outputs.keys() and 'style_code' in outputs.keys():
                style_si = outputs['style_code_si']  
                style_c0 = outputs['style_code_c0']  
                ci_id = random.randint(0, outputs["style_code"].shape[0]-1)              
                style_ci = outputs["style_code"][ci_id].clone() 
                contrastive_style_loss = self.contrastive_style_loss(style_c0, style_ci, style_si) 
                loss_dict['contrastive_style_loss'] = contrastive_style_loss * 0.1
                
        if self.config.use_style and self.config.use_depth: 
            if 'edited_img_depth' in outputs.keys():
                depth_s0 = outputs['edited_img_depth'][None]
                depth_c0 = outputs['rendered_depth'][None]

                depth_s0_normalized = (depth_s0 - depth_s0.min()) / (depth_s0.max() - depth_s0.min())
                depth_c0_normalized = (depth_c0 - depth_c0.min()) / (depth_c0.max() - depth_c0.min())

                depth_mse_loss = torch.mean((depth_s0_normalized - depth_c0_normalized)**2)  
                loss_dict['depth_mse_loss'] = depth_mse_loss

        return loss_dict
