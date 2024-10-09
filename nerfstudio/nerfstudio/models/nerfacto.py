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
NeRF implementation that combines many recent advancements.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Type

import numpy as np
import torch
from torch.nn import Parameter
from torchmetrics import PeakSignalNoiseRatio
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from typing_extensions import Literal

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.engine.callbacks import (
    TrainingCallback,
    TrainingCallbackAttributes,
    TrainingCallbackLocation,
)
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.spatial_distortions import SceneContraction
from nerfstudio.fields.density_fields import HashMLPDensityField
from nerfstudio.fields.nerfacto_field import TCNNNerfactoField
from nerfstudio.model_components.losses import (
    MSELoss,
    distortion_loss,
    interlevel_loss,
    orientation_loss,
    pred_normal_loss,
)
from nerfstudio.model_components.ray_samplers import (
    ProposalNetworkSampler,
    UniformSampler,
)
from nerfstudio.model_components.renderers import (
    AccumulationRenderer,
    DepthRenderer,
    NormalsRenderer,
    RGBRenderer,
    VAE_encoder
)
from nerfstudio.model_components.scene_colliders import NearFarCollider
from nerfstudio.model_components.shaders import NormalsShader
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils import colormaps

# lxy
from nerfstudio.models.thmodel import Discriminator, DiscriminatorLoss, ContrastiveLoss, GeneratorLoss
import torch.nn.functional as F
import torch.nn as nn

@dataclass
class NerfactoModelConfig(ModelConfig):
    """Nerfacto Model Config"""

    _target: Type = field(default_factory=lambda: NerfactoModel)
    near_plane: float = 0.05
    """How far along the ray to start sampling."""
    far_plane: float = 1000.0
    """How far along the ray to stop sampling."""
    background_color: Literal["random", "last_sample", "black", "white"] = "last_sample"
    """Whether to randomize the background color."""
    hidden_dim: int = 64
    """Dimension of hidden layers"""
    hidden_dim_color: int = 64
    """Dimension of hidden layers for color network"""
    hidden_dim_transient: int = 64
    """Dimension of hidden layers for transient network"""
    num_levels: int = 16
    """Number of levels of the hashmap for the base mlp."""
    max_res: int = 2048
    """Maximum resolution of the hashmap for the base mlp."""
    log2_hashmap_size: int = 19
    """Size of the hashmap for the base mlp"""
    num_proposal_samples_per_ray: Tuple[int, ...] = (256, 96)
    """Number of samples per ray for each proposal network."""
    num_nerf_samples_per_ray: int = 48
    """Number of samples per ray for the nerf network."""
    proposal_update_every: int = 5
    """Sample every n steps after the warmup"""
    proposal_warmup: int = 5000
    """Scales n from 1 to proposal_update_every over this many steps"""
    num_proposal_iterations: int = 2
    """Number of proposal network iterations."""
    use_same_proposal_network: bool = False
    """Use the same proposal network. Otherwise use different ones."""
    proposal_net_args_list: List[Dict] = field(
        default_factory=lambda: [
            {"hidden_dim": 16, "log2_hashmap_size": 17, "num_levels": 5, "max_res": 128, "use_linear": False},
            {"hidden_dim": 16, "log2_hashmap_size": 17, "num_levels": 5, "max_res": 256, "use_linear": False},
        ]
    )
    """Arguments for the proposal density fields."""
    proposal_initial_sampler: Literal["piecewise", "uniform"] = "piecewise"
    """Initial sampler for the proposal network. Piecewise is preferred for unbounded scenes."""
    interlevel_loss_mult: float = 1.0
    """Proposal loss multiplier."""
    distortion_loss_mult: float = 0.002
    """Distortion loss multiplier."""
    orientation_loss_mult: float = 0.0001
    """Orientation loss multiplier on computed normals."""
    pred_normal_loss_mult: float = 0.001
    """Predicted normal loss multiplier."""
    use_proposal_weight_anneal: bool = True
    """Whether to use proposal weight annealing."""
    use_average_appearance_embedding: bool = True
    """Whether to use average appearance embedding or zeros for inference."""
    proposal_weights_anneal_slope: float = 10.0
    """Slope of the annealing function for the proposal weights."""
    proposal_weights_anneal_max_num_iters: int = 1000
    """Max num iterations for the annealing function."""
    use_single_jitter: bool = True
    """Whether use single jitter or not for the proposal networks."""
    predict_normals: bool = False
    """Whether to predict normals or not."""
    disable_scene_contraction: bool = False
    """Whether to disable scene contraction or not."""
    
    # bellows are added by # lxy
    use_style: bool = True
    "Whether to use Style latent code to edit NeRF scene"    
    use_debug: bool = False
    "Whether to use debugging mode"  
    use_dis: bool = True
    "Whether to use discriminator to add regularization between novel views"  
    use_inv: bool = False
    "Whether to use inverse prompt for forward and backward cycle style transforms"  # default False
    use_generative: bool = True
    "Whether to make pipeline into generative fashion"  
    load_editimgs: bool = True
    "Whether to use pre edited imgs for fastern speed"
    use_deltaxyz: bool = False
    # use_deltaxyz: bool = True
    "Whether to use residual nerf method by z -> delta xyz + xyz -> nerf"  # default False
    use_contrastive_style: bool = True
    "Whether to use contrastive learing on style z, c0,ci close; ci, si far"  # 2023-05-24
    use_depth: bool = False
    "Whether to use inferenced depth from LeReS"  # 2023-05-24  # default Falses
    
    # # bellows are origin in2n 
    # use_style: bool = False
    # "Whether to use Style latent code to edit NeRF scene"    
    # use_debug: bool = False
    # "Whether to use debugging mode"  
    # use_dis: bool = False
    # "Whether to use discriminator to add regularization between novel views"  
    # use_inv: bool = False
    # "Whether to use inverse prompt for forward and backward cycle style transforms"  # default False
    # use_generative: bool = False
    # "Whether to make pipeline into generative fashion"  
    # load_editimgs: bool = False
    # "Whether to use pre edited imgs for fastern speed"
    # use_deltaxyz: bool = False
    # "Whether to use residual nerf method by z -> delta xyz + xyz -> nerf"  # default False
    # use_contrastive_style: bool = False
    # "Whether to use contrastive learing on style z, c0,ci close; ci, si far"  # 2023-05-24
    # use_depth: bool = False
    # "Whether to use inferenced depth from LeReS"  # 2023-05-24  # default Falses
    


class NerfactoModel(Model):
    """Nerfacto model

    Args:
        config: Nerfacto configuration to instantiate model
    """

    config: NerfactoModelConfig

    def populate_modules(self):
        """Set the fields and modules."""
        super().populate_modules()

        if self.config.disable_scene_contraction:
            scene_contraction = None
        else:
            scene_contraction = SceneContraction(order=float("inf"))

        # import pdb; pdb.set_trace()
        # Fields  # lxy add use_style
        self.field = TCNNNerfactoField(
            self.scene_box.aabb,
            hidden_dim=self.config.hidden_dim,
            num_levels=self.config.num_levels,
            max_res=self.config.max_res,
            log2_hashmap_size=self.config.log2_hashmap_size,
            hidden_dim_color=self.config.hidden_dim_color,
            hidden_dim_transient=self.config.hidden_dim_transient,
            spatial_distortion=scene_contraction,
            num_images=self.num_train_data,
            use_pred_normals=self.config.predict_normals,
            use_average_appearance_embedding=self.config.use_average_appearance_embedding,
        )
        # import pdb; pdb.set_trace()

        self.density_fns = []
        num_prop_nets = self.config.num_proposal_iterations
        # Build the proposal network(s)
        # import pdb; pdb.set_trace()  # mark 
        
        self.proposal_networks = torch.nn.ModuleList()
        if self.config.use_same_proposal_network:
            assert len(self.config.proposal_net_args_list) == 1, "Only one proposal network is allowed."
            prop_net_args = self.config.proposal_net_args_list[0]
            network = HashMLPDensityField(self.scene_box.aabb, spatial_distortion=scene_contraction, **prop_net_args)
            self.proposal_networks.append(network)
            self.density_fns.extend([network.density_fn for _ in range(num_prop_nets)])
        else:
            for i in range(num_prop_nets):
                prop_net_args = self.config.proposal_net_args_list[min(i, len(self.config.proposal_net_args_list) - 1)]
                network = HashMLPDensityField(
                    self.scene_box.aabb,
                    spatial_distortion=scene_contraction,
                    **prop_net_args,
                )
                self.proposal_networks.append(network)
            self.density_fns.extend([network.density_fn for network in self.proposal_networks])

        # Samplers
        update_schedule = lambda step: np.clip(
            np.interp(step, [0, self.config.proposal_warmup], [0, self.config.proposal_update_every]),
            1,
            self.config.proposal_update_every,
        )

        # Change proposal network initial sampler if uniform
        initial_sampler = None  # None is for piecewise as default (see ProposalNetworkSampler)
        if self.config.proposal_initial_sampler == "uniform":
            initial_sampler = UniformSampler(single_jitter=self.config.use_single_jitter)

        self.proposal_sampler = ProposalNetworkSampler(
            num_nerf_samples_per_ray=self.config.num_nerf_samples_per_ray,
            num_proposal_samples_per_ray=self.config.num_proposal_samples_per_ray,
            num_proposal_network_iterations=self.config.num_proposal_iterations,
            single_jitter=self.config.use_single_jitter,
            update_sched=update_schedule,
            initial_sampler=initial_sampler,
        )

        # Collider
        self.collider = NearFarCollider(near_plane=self.config.near_plane, far_plane=self.config.far_plane)

        # renderers
        self.renderer_rgb = RGBRenderer(background_color=self.config.background_color)  # volume rendering accumlation
        self.renderer_accumulation = AccumulationRenderer()
        # self.renderer_depth = DepthRenderer()
        self.renderer_depth = DepthRenderer('expected')  # 2023-05-24
        self.renderer_normals = NormalsRenderer()

        # shaders
        self.normals_shader = NormalsShader()

        # losses
        self.rgb_loss = MSELoss()

        # metrics
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = structural_similarity_index_measure
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)
        
        # novel views 2023-05-17  # lxy 
        if self.config.use_dis:
            self.discriminator = Discriminator()
            self.discriminator_loss = DiscriminatorLoss()
            self.generator_loss = GeneratorLoss()  # 20230608
        
        # forward backward transform  2023-05-18 # lxy    
        if self.config.use_inv:
            self.contrastive_loss = ContrastiveLoss(margin=1.8)  # for face
            
        # generative pipeline     
        if self.config.use_generative:
            self.kl_loss = nn.KLDivLoss(reduction="batchmean")
            self.pdist = torch.nn.PairwiseDistance(p=2)#p=2就是计算欧氏距离，p=1就是曼哈顿距离，例如上面的例子，距离是1.
            
        # contrastive loss on style     
        if self.config.use_contrastive_style:
            self.contrastive_style_loss = ContrastiveLoss(margin=1.8)  # for face # ??
            
        # if self.config.use_depth:
        #     self.depth_loss = DepthLoss()
            
            

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = {}
        param_groups["proposal_networks"] = list(self.proposal_networks.parameters())
        param_groups["fields"] = list(self.field.parameters())
        return param_groups

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        callbacks = []
        if self.config.use_proposal_weight_anneal:
            # anneal the weights of the proposal network before doing PDF sampling
            N = self.config.proposal_weights_anneal_max_num_iters

            def set_anneal(step):
                # https://arxiv.org/pdf/2111.12077.pdf eq. 18
                train_frac = np.clip(step / N, 0, 1)
                bias = lambda x, b: (b * x) / ((b - 1) * x + 1)
                anneal = bias(train_frac, self.config.proposal_weights_anneal_slope)
                self.proposal_sampler.set_anneal(anneal)

            callbacks.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                    update_every_num_iters=1,
                    func=set_anneal,
                )
            )
            callbacks.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                    update_every_num_iters=1,
                    func=self.proposal_sampler.step_cb,
                )
            )
        return callbacks

    def get_outputs(self, ray_bundle: RayBundle):
        ray_samples, weights_list, ray_samples_list = self.proposal_sampler(ray_bundle, density_fns=self.density_fns)
        
        # if 'styles' in ray_bundle.metadata.keys():
        #     import pdb; pdb.set_trace()
        field_outputs = self.field(ray_samples, compute_normals=self.config.predict_normals)  # mark predict density+rgb  # change inside  # 2023-5-14
        weights = ray_samples.get_weights(field_outputs[FieldHeadNames.DENSITY])
        weights_list.append(weights)
        ray_samples_list.append(ray_samples)

        rgb = self.renderer_rgb(rgb=field_outputs[FieldHeadNames.RGB], weights=weights)  # use volume rendering accumlation
        depth = self.renderer_depth(weights=weights, ray_samples=ray_samples)
        accumulation = self.renderer_accumulation(weights=weights)

        outputs = {
            "rgb": rgb,
            "accumulation": accumulation,
            "depth": depth,
        }

        if self.config.predict_normals:
            normals = self.renderer_normals(normals=field_outputs[FieldHeadNames.NORMALS], weights=weights)
            pred_normals = self.renderer_normals(field_outputs[FieldHeadNames.PRED_NORMALS], weights=weights)
            outputs["normals"] = self.normals_shader(normals)
            outputs["pred_normals"] = self.normals_shader(pred_normals)
        # These use a lot of GPU memory, so we avoid storing them for eval.
        if self.training:
            outputs["weights_list"] = weights_list
            outputs["ray_samples_list"] = ray_samples_list

        if self.training and self.config.predict_normals:
            outputs["rendered_orientation_loss"] = orientation_loss(
                weights.detach(), field_outputs[FieldHeadNames.NORMALS], ray_bundle.directions
            )

            outputs["rendered_pred_normal_loss"] = pred_normal_loss(
                weights.detach(),
                field_outputs[FieldHeadNames.NORMALS].detach(),
                field_outputs[FieldHeadNames.PRED_NORMALS],
            )

        for i in range(self.config.num_proposal_iterations):
            outputs[f"prop_depth_{i}"] = self.renderer_depth(weights=weights_list[i], ray_samples=ray_samples_list[i])
            
        # # 2023-05-17    
        # # remove to outside in2n_pipeline.py
        # if self.config.use_style and self.config.use_dis:
        #     if 'novel_edited_image' in ray_bundle.metadata.keys():
        #         import pdb; pdb.set_trace()
        #         H = int(ray_bundle.metadata['H'][-1])
        #         W = int(ray_bundle.metadata['W'][-1])
                
        #         novel_edited_image = ray_bundle.metadata['novel_edited_image'].view(1, 3, H, W).to(self.device)  # mark hard code
        #         input = torch.cat([novel_edited_image, novel_edited_image - novel_edited_image], dim=1)
        #         predict_real = self.discriminator(input)
                
        #         # render img on novel view # with grad
        #         camera_outputs = self.model.get_outputs_for_camera_ray_bundle(novel_ray_bundle) 
        #         # outputs["rgb"] # [16384, 3]
        
        # # forward and backward transform  # 2023-05-18 # lxy
        # if self.config.use_inv:
        #     if 'styles_ori' in ray_bundle.metadata.keys():
        #         styles_edit = ray_bundle.metadata['styles']  # [1,64]
        #         styles_origin = ray_bundle.metadata['styles_ori'] 
        #         styles_origin_noisy = ray_bundle.metadata['styles_inv'] 
        #         self.contrastive_loss = self.contrastive_loss.to(self.device)
        #         contrastive_loss = self.contrastive_loss(styles_origin, styles_origin_noisy, styles_edit)
                
        #         outputs['contrastive_loss'] = contrastive_loss
                
        # if self.config.use_generative:
        #     self.kl_loss = self.kl_loss.to(self.device)
        #     import pdb; pdb.set_trace()
        #     styles_origin = ray_bundle.metadata['styles_ori'] 
        #     styles_origin_noisy = ray_bundle.metadata['styles_inv'] 
        #     kl_loss = self.kl_loss(styles_origin, styles_origin_noisy, styles_edit)
        #     outputs['kl_loss'] = kl_loss
            
        #     styles_edit = ray_bundle.metadata['styles'].clone()  # [1,64]
        #     input = F.log_softmax(styles_edit, dim=1)
        #     target = F.softmax(torch.randn(1, 64), dim=1)
            
        #     log_target = F.log_softmax(torch.rand(3, 5), dim=1)
        #     kl_loss = kl_loss(input, log_target)
        #     outputs['kl_loss'] = kl_loss
            
        
            

        return outputs

    def get_metrics_dict(self, outputs, batch):
        metrics_dict = {}
        image = batch["image"].to(self.device)
        metrics_dict["psnr"] = self.psnr(outputs["rgb"], image)
        if self.training:
            metrics_dict["distortion"] = distortion_loss(outputs["weights_list"], outputs["ray_samples_list"])
        return metrics_dict

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        loss_dict = {}
        image = batch["image"].to(self.device)
        loss_dict["rgb_loss"] = self.rgb_loss(image, outputs["rgb"])
    
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
        return loss_dict

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        image = batch["image"].to(self.device)
        rgb = outputs["rgb"]
        acc = colormaps.apply_colormap(outputs["accumulation"])
        depth = colormaps.apply_depth_colormap(
            outputs["depth"],
            accumulation=outputs["accumulation"],
        )

        combined_rgb = torch.cat([image, rgb], dim=1)
        combined_acc = torch.cat([acc], dim=1)
        combined_depth = torch.cat([depth], dim=1)

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        image = torch.moveaxis(image, -1, 0)[None, ...]
        rgb = torch.moveaxis(rgb, -1, 0)[None, ...]

        psnr = self.psnr(image, rgb)
        ssim = self.ssim(image, rgb)
        lpips = self.lpips(image, rgb)

        # all of these metrics will be logged as scalars
        metrics_dict = {"psnr": float(psnr.item()), "ssim": float(ssim)}  # type: ignore
        metrics_dict["lpips"] = float(lpips)

        images_dict = {"img": combined_rgb, "accumulation": combined_acc, "depth": combined_depth}

        for i in range(self.config.num_proposal_iterations):
            key = f"prop_depth_{i}"
            prop_depth_i = colormaps.apply_depth_colormap(
                outputs[key],
                accumulation=outputs["accumulation"],
            )
            images_dict[key] = prop_depth_i

        return metrics_dict, images_dict
