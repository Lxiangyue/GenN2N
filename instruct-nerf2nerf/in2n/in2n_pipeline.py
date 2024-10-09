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

"""InstructPix2Pix Pipeline and trainer"""

from dataclasses import dataclass, field
from itertools import cycle
from typing import Optional, Type
import torch
from typing_extensions import Literal
from nerfstudio.pipelines.base_pipeline import VanillaPipeline, VanillaPipelineConfig
from nerfstudio.viewer.server.viewer_elements import ViewerNumber, ViewerText

from in2n.in2n_datamanager import (
    InstructNeRF2NeRFDataManagerConfig,
)
from in2n.ip2p import InstructPix2Pix

import pdb

import random
import cv2
import os

from in2n.save_video import save_vid
import glob


def save_img(img, name):
    # img: [3,h,w], name: 'debug/name'
    import os
    from torchvision import transforms
    if not os.path.isdir('debug1'): os.mkdir('debug1')
    toPIL = transforms.ToPILImage() 
    pic = toPIL(img); pic.save('debug1/' + name)


def save_img_absname(img, absname): 
    # img: [3,h,w], name: 'debug/name'
    import os
    from torchvision import transforms
    toPIL = transforms.ToPILImage() 
    pic = toPIL(img); pic.save(absname)


@dataclass
class InstructNeRF2NeRFPipelineConfig(VanillaPipelineConfig):
    """Configuration for pipeline instantiation"""

    _target: Type = field(default_factory=lambda: InstructNeRF2NeRFPipeline)
    """target class to instantiate"""
    datamanager: InstructNeRF2NeRFDataManagerConfig = InstructNeRF2NeRFDataManagerConfig()
    """specifies the datamanager config"""
    prompt: str = "don't change the image"
    """prompt for InstructPix2Pix"""
    guidance_scale: float = 7.5
    """(text) guidance scale for InstructPix2Pix"""
    image_guidance_scale: float = 1.5
    """image guidance scale for InstructPix2Pix"""
    edit_rate: int = 10
    """how many NeRF steps before image edit"""
    edit_count: int = 1
    """how many images to edit per NeRF step"""
    diffusion_steps: int = 20
    """Number of diffusion steps to take for InstructPix2Pix"""
    lower_bound: float = 0.02
    """Lower bound for diffusion timesteps to use for image editing"""
    upper_bound: float = 0.98
    """Upper bound for diffusion timesteps to use for image editing"""
    ip2p_device: Optional[str] = None
    """Second device to place InstructPix2Pix on. If None, will use the same device as the pipeline"""
    ip2p_use_full_precision: bool = True
    """Whether to use full precision for InstructPix2Pix"""
    prompt_inv: Optional[str] = None   
    """inverse prompt for forward-backward cycle transforms, GenN2N"""  
    edited_imgs_path: Optional[str] = None    
    """for pre load, GenN2N"""  
    edited_imgs_path_type: Optional[str] = None   
    """for pre load, GenN2N"""  
    edited_imgs_depth_path: Optional[str] = None   
    """for pre load, GenN2N"""  
    inference_num_per_ckpt: Optional[int] = 1
    """inference numbers of per trained model, GenN2N"""  
    
    
import time    
class Timer:
    def __init__(self):
        self.items = {}
        self.time_scale = 1000.0  # ms
        self.time_unit = "ms"

    def start(self, name: str) -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.items[name] = time.time()
        print(f"{name} ...")
        # logging.info(f"{name} ...")

    def end(self, name: str) -> None:
        if name not in self.items:
            return
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start_time = self.items.pop(name)
        delta = time.time() - start_time
        t = delta * self.time_scale
        print(f"{name} finished in {t:.2f}{self.time_unit}.")
        # logging.info(f"{name} finished in {t:.2f}{self.time_unit}.")
    
Timer = Timer()
    

class InstructNeRF2NeRFPipeline(VanillaPipeline):
    """InstructNeRF2NeRF pipeline"""

    config: InstructNeRF2NeRFPipelineConfig

    def __init__(
        self,
        config: InstructNeRF2NeRFPipelineConfig,
        device: str,
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
    ):
        super().__init__(config, device, test_mode, world_size, local_rank)

        # select device for InstructPix2Pix
        self.ip2p_device = (
            torch.device(device)
            if self.config.ip2p_device is None
            else torch.device(self.config.ip2p_device)
        )

        self.ip2p = InstructPix2Pix(self.ip2p_device, ip2p_use_full_precision=self.config.ip2p_use_full_precision)

        # load base text embedding using classifier free guidance
        self.text_embedding = self.ip2p.pipe._encode_prompt(
            self.config.prompt, device=self.ip2p_device, num_images_per_prompt=1, do_classifier_free_guidance=True, negative_prompt=""
        )
        
        # self.text_embedding_inv = self.ip2p.pipe._encode_prompt(
        #     self.config.prompt_inv, device=self.ip2p_device, num_images_per_prompt=1, do_classifier_free_guidance=True, negative_prompt=""
        # )
        self.text_embedding_inv = self.ip2p.pipe._encode_prompt(
            "don't change the image", device=self.ip2p_device, num_images_per_prompt=1, do_classifier_free_guidance=True, negative_prompt=""
        )

        # keep track of spot in dataset
        if self.datamanager.config.train_num_images_to_sample_from == -1:
            self.train_indices_order = cycle(range(len(self.datamanager.train_dataparser_outputs.image_filenames)))
        else:
            self.train_indices_order = cycle(range(self.datamanager.config.train_num_images_to_sample_from))

        # viewer elements
        self.prompt_box = ViewerText(name="Prompt", default_value=self.config.prompt, cb_hook=self.prompt_callback)
        self.guidance_scale_box = ViewerNumber(name="Text Guidance Scale", default_value=self.config.guidance_scale, cb_hook=self.guidance_scale_callback)
        self.image_guidance_scale_box = ViewerNumber(name="Image Guidance Scale", default_value=self.config.image_guidance_scale, cb_hook=self.image_guidance_scale_callback)


        if self.model.config.load_editimgs:
            # load all ore edited images, GenN2N  
            
            import glob
            from PIL import Image
            import numpy as np
            
            pre_edited_imgs = []
            pre_edited_imgs_one_style = []

            for i,path in enumerate(sorted(glob.glob(self.config.edited_imgs_path))):  # style  
                for j,name in enumerate(sorted(glob.glob('%s/*.%s'%(path, self.config.edited_imgs_path_type)))):  # view  
                    pil_image = Image.open(name) 
                    image = np.array(pil_image, dtype="uint8") 
                    pre_edited_imgs_one_style.append(torch.Tensor(image))
                pre_edited_imgs_one_style_stack = torch.stack(pre_edited_imgs_one_style).to(self.device)  
                pre_edited_imgs_one_style = []
                pre_edited_imgs.append(pre_edited_imgs_one_style_stack) 
                del pre_edited_imgs_one_style_stack
            pre_edited_imgs = torch.stack(pre_edited_imgs)  
            self.pre_edited_imgs = pre_edited_imgs 
            del pre_edited_imgs

            # load edited imgs's depth as gt, GenN2N  
            if self.model.config.use_depth:
                assert self.config.edited_imgs_depth_path is not None, 'Please give depth path'
                pre_edited_imgs_depth = []
                pre_edited_imgs_one_depth = []
                for i,path in enumerate(sorted(glob.glob(self.config.edited_imgs_depth_path))):  # style   
                    for j,name in enumerate(sorted(glob.glob('%s/*depth_raw.%s'%(path, self.config.edited_imgs_path_type)))):  # view  
                        depth_img = cv2.imread(name, cv2.IMREAD_UNCHANGED) 
                        pre_edited_imgs_one_depth.append(torch.Tensor(depth_img/1.0))
                    pre_edited_imgs_one_depth_stack = torch.stack(pre_edited_imgs_one_depth).to(self.device) 
                    pre_edited_imgs_one_depth = []
                    pre_edited_imgs_depth.append(pre_edited_imgs_one_depth_stack) 
                    del pre_edited_imgs_one_depth_stack
                pre_edited_imgs_depth = torch.stack(pre_edited_imgs_depth)  
                self.pre_edited_imgs_depth = pre_edited_imgs_depth 
                del pre_edited_imgs_depth


    def guidance_scale_callback(self, handle: ViewerText) -> None:
        """Callback for guidance scale slider"""
        self.config.guidance_scale = handle.value

    def image_guidance_scale_callback(self, handle: ViewerText) -> None:
        """Callback for text guidance scale slider"""
        self.config.image_guidance_scale = handle.value

    def prompt_callback(self, handle: ViewerText) -> None:
        """Callback for prompt box, change prompt in config and update text embedding"""
        self.config.prompt = handle.value
        
        self.text_embedding = self.ip2p.pipe._encode_prompt(
            self.config.prompt, device=self.ip2p_device, num_images_per_prompt=1, do_classifier_free_guidance=True, negative_prompt=""
        )
        

    def get_train_loss_dict(self, step: int, checkpoint_dir=None, max_num_iterations=None, load_dir=None):  
        """This function gets your training loss dict and performs image editing.
        Args:
            step: current iteration step to update sampler if using DDP (distributed)
        """

        if self.model.config.use_style:  # GenN2N      
            
            # too slow if edit an image every iteration, alternatively we load edited image as input
            ray_bundle, batch = self.datamanager.next_train(step)     
            
            # render imgs the same view with the input all train datas 
            inference_num = self.config.inference_num_per_ckpt    # inference how many times, default = 1

            # start inference
            if step == 30000 or step % 2500 == 0 or max_num_iterations == 1:   # using step = 30000 is render the original NeRF  # max_num_iterations=1 means only load the model and render imgs and then break
                with torch.no_grad():
                    save_path = str(checkpoint_dir).split('nerfstudio_models')[0] + 'inference_step%06d'%step    
                    if max_num_iterations == 1:
                        save_path = str(load_dir).split('nerfstudio_models')[0] + 'inference_step%06d'%(step-1)
                        
                    if not os.path.isdir(save_path): os.mkdir(save_path)
                    save_path_styles = save_path + '/sample_styles'
                    if not os.path.isdir(save_path_styles): os.mkdir(save_path_styles)
                    save_video_path = save_path + '/inference_videos'
                    if not os.path.isdir(save_video_path): os.mkdir(save_video_path)

                    if step == 30000: 
                        inference_num = 1   # render original NeRF only once
                        
                    # inference_interploate = True  
                    inference_interploate = False
                    if inference_interploate:    
                        # interploate z 
                        z_dir = '/apdcephfs/share_1330077/emmafxliu/Codes_instruct/instruct-nerf2nerf/outputs/fangzhou-small/in2n/2023-11-27_024729_Elf_use_interplote/old/inference_step035000_5/sample_styles'
                        z_interploate_dir = '/apdcephfs/share_1330077/emmafxliu/Codes_instruct/instruct-nerf2nerf/outputs/fangzhou-small/in2n/2023-11-27_024729_Elf_use_interplote/old/inference_step035000_5/sample_styles_interploate'
                        if not os.path.isdir(z_interploate_dir): os.mkdir(z_interploate_dir)
                        z0 = torch.load(z_dir + '/inference_002_styles.pt')
                        z1 = torch.load(z_dir + '/inference_004_styles.pt')
                        
                        save_path = save_path + '_interploate'
                        if not os.path.isdir(save_path): os.mkdir(save_path)
                        save_path_styles = save_path + '/sample_styles'
                        if not os.path.isdir(save_path_styles): os.mkdir(save_path_styles)
                        save_video_path = save_path + '/inference_videos'
                        if not os.path.isdir(save_video_path): os.mkdir(save_video_path)
                        
                        for id in range(11): 
                            alpha = id * 0.1
                            beta = 1 - alpha
                            z_interploate = z0 * alpha + z1 * beta
                            sample_style = z_interploate.to(self.device)
                            torch.save(sample_style.clone().detach().cpu(), save_path_styles +'/inference_styles_alpha_%02f_beta_%02f.pt'%(alpha, beta))
                            
                            save_path_inference = save_path + '/inference_styles_alpha_%02f_beta_%02f'%(alpha, beta)
                            if not os.path.isdir(save_path_inference): os.mkdir(save_path_inference)  

                            img_list = []
                            for iter in range(len(self.datamanager.train_dataparser_outputs.image_filenames)):   
                                
                                original_image = self.datamanager.original_image_batch["image"][iter].to(self.device)  
                                num = self.datamanager.original_image_batch['image_filename_num'][iter].to(self.device) 
                                
                                current_index = self.datamanager.image_batch["image_idx"][iter]  

                                camera_transforms = self.datamanager.train_camera_optimizer(current_index.unsqueeze(dim=0))   # (3,4)
                                current_camera = self.datamanager.train_dataparser_outputs.cameras[current_index].to(self.device)  # Cameras
                                current_ray_bundle = current_camera.generate_rays(torch.tensor(list(range(1))).unsqueeze(-1), camera_opt_to_camera=camera_transforms) 

                                if step == 30000: # render the edited NeRF imgs
                                    pass  
                                else: # render the origin imgs from NeRF  
                                    current_ray_bundle.metadata['styles'] = sample_style

                                # Timer.start(name="inference")
                                camera_outputs = self.model.get_outputs_for_camera_ray_bundle(current_ray_bundle)
                                # Timer.end(name="inference")
                                
                                rendered_image = camera_outputs["rgb"].unsqueeze(dim=0).permute(0, 3, 1, 2)   

                                img = rendered_image.clone().detach().cpu()
                                save_img_absname(img.squeeze(), save_path_inference + '/frame_%05d.jpg'%num)
            
                            for i,path in enumerate(sorted(glob.glob(save_path_inference + '/frame_*.*'))): 
                                img_list.append(torch.Tensor(cv2.imread(path)[:,:,::-1].copy()))
                            save_vid(save_video_path+'/inference_styles_alpha_%02f_beta_%02f'%(alpha, beta), torch.stack(img_list), suffix='.mp4',upsample_frame=0)  # random order in video
                        
                    else:          

                        for numm in range(inference_num):
                            save_path_inference = save_path + '/inference_%03d'%numm
                            if not os.path.isdir(save_path_inference): os.mkdir(save_path_inference)   

                            sample_style = torch.randn((1, 64)).to(self.device)
                            torch.save(sample_style.clone().detach().cpu(), save_path_styles+'/inference_%03d_styles.pt'%numm)

                            img_list = []
                            for iter in range(len(self.datamanager.train_dataparser_outputs.image_filenames)):   
                                
                                original_image = self.datamanager.original_image_batch["image"][iter].to(self.device)  
                                num = self.datamanager.original_image_batch['image_filename_num'][iter].to(self.device) # original_image and frame_%05d(%num).jpg is the same img
                                
                                current_index = self.datamanager.image_batch["image_idx"][iter]  

                                camera_transforms = self.datamanager.train_camera_optimizer(current_index.unsqueeze(dim=0))   
                                current_camera = self.datamanager.train_dataparser_outputs.cameras[current_index].to(self.device)  
                                current_ray_bundle = current_camera.generate_rays(torch.tensor(list(range(1))).unsqueeze(-1), camera_opt_to_camera=camera_transforms) 

                                if step == 30000: # render the edited NeRF imgs
                                    pass  
                                else: # render the origin imgs from NeRF  
                                    current_ray_bundle.metadata['styles'] = sample_style

                                camera_outputs = self.model.get_outputs_for_camera_ray_bundle(current_ray_bundle)
                                rendered_image = camera_outputs["rgb"].unsqueeze(dim=0).permute(0, 3, 1, 2)  

                                img = rendered_image.clone().detach().cpu()
                                save_img_absname(img.squeeze(), save_path_inference + '/frame_%05d.jpg'%num)
            
                            print('inference num %03d done'%numm, 'saving in the %s'%save_path_inference)
                            
                            for i,path in enumerate(sorted(glob.glob(save_path_inference + '/frame_*.*'))): 
                                img_list.append(torch.Tensor(cv2.imread(path)[:,:,::-1].copy()))
                    
            if max_num_iterations == 1:
                print('RENDERING ALL DONE! Could break now.')
                # return model_outputs, loss_dict, metrics_dict
                return None, None, None
            
            # start training
            current_spot = next(self.train_indices_order)
            
            # get original image from dataset
            original_image = self.datamanager.original_image_batch["image"][current_spot].to(self.device) 
            num = self.datamanager.original_image_batch['image_filename_num'][current_spot].to(self.device) # original_image和frame_%05d(%num).jpg is the same img
            
            # get current index in datamanger
            current_index = self.datamanager.image_batch["image_idx"][current_spot]  

            # get current camera, include camera transforms from original optimizer
            camera_transforms = self.datamanager.train_camera_optimizer(current_index.unsqueeze(dim=0))   
            current_camera = self.datamanager.train_dataparser_outputs.cameras[current_index].to(self.device)  
            current_ray_bundle = current_camera.generate_rays(torch.tensor(list(range(1))).unsqueeze(-1), camera_opt_to_camera=camera_transforms)  

            # too slow if edit an image every iteration, alternatively we edit images before training and load them during training
            if self.model.config.load_editimgs:  
                style_id = random.randint(0, self.pre_edited_imgs.shape[0]-1)  
                view_id = num - 1  # frame_%05d.jpg is naming from 1 to 65
                pre_edited_image = (self.pre_edited_imgs[style_id, view_id, :,:,:]/255.0).permute(2, 0, 1)[None] 
                # save_img(pre_edited_image.clone().detach().cpu().squeeze(), 'pre_edited_image.jpg')
                # save_img(original_image.clone().detach().cpu().squeeze(), 'original_image.jpg')
                edited_image = pre_edited_image
                del pre_edited_image
                
                if self.model.config.use_depth:  
                    pre_edited_img_depth = (self.pre_edited_imgs_depth[style_id, view_id, :,:])  # s0
                
            else:
                edited_image = self.ip2p.edit_image(
                            self.text_embedding.to(self.ip2p_device),
                            original_image.to(self.ip2p_device),
                            original_image.to(self.ip2p_device),
                            guidance_scale=self.config.guidance_scale,
                            image_guidance_scale=self.config.image_guidance_scale,
                            diffusion_steps=self.config.diffusion_steps,
                            lower_bound=self.config.lower_bound,
                            upper_bound=self.config.upper_bound,
                        )

            # write edited image to dataloader
            tmp = edited_image.squeeze().permute(1,2,0)
            if tmp.shape != original_image.shape:
                edited_image = torch.nn.functional.interpolate(edited_image, size=original_image.size()[:2], mode='bilinear') 
                self.datamanager.image_batch["image"][current_spot] = edited_image.squeeze().permute(1,2,0)
            else:
                self.datamanager.image_batch["image"][current_spot] = tmp


            # encoder to style code z
            with torch.no_grad():
                latents = self.ip2p.imgs_to_latent(edited_image)
                # decoded_img = self.ip2p.latents_to_img(latents)
                # save_img(decoded_img.clone().detach().cpu().squeeze(), 'decoded_img.jpg')  # exactly the same with edited_image

            current_style = self.ip2p.latent_to_styles(latents)  # [1, 64]

            current_ray_bundle.metadata['styles'] = current_style.clone()

            # get current render of nerf  # no grad, mainly for visualization and debugging
            if self.model.config.use_depth:               
                camera_outputs = self.model.get_outputs_for_camera_ray_bundle(current_ray_bundle, return_depth_grad=True)    
                rendered_depth = camera_outputs['depth'].squeeze()  # c0 # has grad
            else:
                camera_outputs = self.model.get_outputs_for_camera_ray_bundle(current_ray_bundle)
            rendered_image = camera_outputs["rgb"].unsqueeze(dim=0).permute(0, 3, 1, 2)  

            # resize to original image size (often not necessary)
            if (edited_image.size() != rendered_image.size()):
                edited_image = torch.nn.functional.interpolate(edited_image, size=rendered_image.size()[2:], mode='bilinear') 

            original_image = original_image.unsqueeze(dim=0).permute(0, 3, 1, 2)   
            if self.model.config.use_debug:
                save_img(edited_image.clone().detach().cpu().squeeze(), 'edited_image.jpg')
                save_img(original_image.clone().detach().cpu().squeeze(), 'original_image.jpg')

            if self.model.config.use_debug:
                save_img(rendered_image.clone().detach().cpu().squeeze(), 'rendered_image.jpg')                

            # delete to free up memory
            del camera_outputs
            del current_camera
            del current_ray_bundle
            del camera_transforms
            torch.cuda.empty_cache()

            ray_bundle.metadata['styles'] = current_style 
            if self.model.config.use_deltaxyz:
                ray_bundle.metadata['use_deltaxyz'] = torch.tensor(int(self.model.config.use_deltaxyz))[None].repeat(1,64)  
            
            # all training-related value save in model_outputs
            model_outputs = self.model(ray_bundle) # for reconstruction loss
            metrics_dict = self.model.get_metrics_dict(model_outputs, batch)
            
            # novel view regularization (Discriminatar)
            if self.model.config.use_dis: 
                novel_spot = current_spot+1
                if current_spot == (self.datamanager.original_image_batch['image'].shape[0]-1):  # avoid cross border
                    novel_spot = current_spot-1
                    
                novel_img = self.datamanager.original_image_batch['image'][novel_spot].to(self.device) 

                # generate novel index in datamanger
                novel_index = self.datamanager.image_batch["image_idx"][novel_spot]  

                # get novel camera, include camera transforms from original optimizer
                camera_transforms = self.datamanager.train_camera_optimizer(novel_index.unsqueeze(dim=0))  
                novel_camera = self.datamanager.train_dataparser_outputs.cameras[novel_index].to(self.device)  
                novel_ray_bundle = novel_camera.generate_rays(torch.tensor(list(range(1))).unsqueeze(-1), camera_opt_to_camera=camera_transforms) 

                novel_ray_bundle.metadata['styles'] = current_style.clone()

                # get novel render of nerf  
                novel_img = novel_img.unsqueeze(dim=0).permute(0, 3, 1, 2)   
                camera_outputs = self.model.get_outputs_for_camera_ray_bundle(novel_ray_bundle)  
                novel_image = camera_outputs["rgb"].unsqueeze(dim=0).permute(0, 3, 1, 2)  

                # delete to free up memory
                del camera_outputs
                del novel_camera
                del novel_ray_bundle
                del camera_transforms
                torch.cuda.empty_cache()

                # render novel view for GAN loss
                if self.model.config.load_editimgs:  
                    novel_num = self.datamanager.original_image_batch['image_filename_num'][novel_spot].to(self.device) 
                    style_id_two = random.sample(range(0, self.pre_edited_imgs.shape[0]-1), 2) 
                    novel_view_id = novel_num - 1  # frame_%05d.jpg is naming from 1 to 65
                    try:
                        novel_edited_image =   (self.pre_edited_imgs[style_id_two[0], novel_view_id, :,:,:]/255.0).permute(2, 0, 1)[None] 
                        novel_edited_image_2 = (self.pre_edited_imgs[style_id_two[1], novel_view_id, :,:,:]/255.0).permute(2, 0, 1)[None] 
                    except:
                        import pdb; pdb.set_trace()
                    if (novel_edited_image.size() != novel_image.size()):
                        novel_edited_image = torch.nn.functional.interpolate(novel_edited_image, size=novel_image.size()[2:], mode='bilinear')  
                        novel_edited_image_2 = torch.nn.functional.interpolate(novel_edited_image_2, size=novel_image.size()[2:], mode='bilinear')                              
                else:
                    novel_edited_image = self.ip2p.edit_image(
                                self.text_embedding.to(self.ip2p_device),
                                novel_image.to(self.ip2p_device),
                                novel_img.to(self.ip2p_device),
                                guidance_scale=self.config.guidance_scale,
                                image_guidance_scale=self.config.image_guidance_scale,
                                diffusion_steps=self.config.diffusion_steps,
                                lower_bound=self.config.lower_bound,
                                upper_bound=self.config.upper_bound,
                            )
                    if (novel_edited_image.size() != novel_image.size()):
                        novel_edited_image = torch.nn.functional.interpolate(novel_edited_image, size=novel_image.size()[2:], mode='bilinear')  
                    assert (novel_edited_image.dim() == 4) and (novel_image.dim() == 4), "Check input"
                    novel_edited_image_2 = self.ip2p.edit_image(
                                self.text_embedding.to(self.ip2p_device),
                                novel_image.to(self.ip2p_device),
                                novel_img.to(self.ip2p_device),
                                guidance_scale=self.config.guidance_scale,
                                image_guidance_scale=self.config.image_guidance_scale,
                                diffusion_steps=self.config.diffusion_steps,
                                lower_bound=self.config.lower_bound,
                                upper_bound=self.config.upper_bound,
                            )
                
                input = torch.cat([novel_edited_image, novel_edited_image_2 - novel_edited_image], dim=1)   
                self.model.discriminator = self.model.discriminator.to(self.device)
                predict_real = self.model.discriminator(input)

                input = torch.cat([novel_edited_image, novel_image - novel_edited_image], dim=1)
                predict_fake = self.model.discriminator(input)      
                
                model_outputs['predict_real'] = predict_real
                model_outputs['predict_fake'] = predict_fake
    
            
            # KL loss  compute all views' origin imgs' edited imgs
            if self.config.model.load_editimgs:  
                all_styles_edited_image = (self.pre_edited_imgs[:5, view_id, :, :, :]/255.0).permute(0,3,1,2)  # [5, 3, 384, 512] # avoid OOM      
                all_styles_edited_image = torch.nn.functional.interpolate(all_styles_edited_image, size=novel_image.size()[2:], mode='bilinear') 
                    
                torch.cuda.empty_cache()
                # encoder to style code z
                with torch.no_grad(): 
                    latents = self.ip2p.imgs_to_latent(all_styles_edited_image).squeeze()   
                edited_style_code = self.ip2p.latent_to_styles(latents)
                model_outputs['edited_style_code'] = edited_style_code   
                
            else:
                if (step % 10 == 0) and (self.model.config.use_generative):  # do KL loss using gap of 10 iterations
                    latent_list = []
                    # for i in range(len(self.datamanager.image_batch["image_idx"])):   # 59  # too slow!
                    for i in range(5):   
                        current_spot = self.datamanager.image_batch["image_idx"][i]
                        original_image = self.datamanager.original_image_batch["image"][current_spot].to(self.device)  
                        original_image = original_image.unsqueeze(dim=0).permute(0, 3, 1, 2)   
                        edited_image = self.ip2p.edit_image(
                                    self.text_embedding.to(self.ip2p_device),
                                    original_image.to(self.ip2p_device),
                                    original_image.to(self.ip2p_device),
                                    guidance_scale=self.config.guidance_scale,
                                    image_guidance_scale=self.config.image_guidance_scale,
                                    diffusion_steps=self.config.diffusion_steps,
                                    lower_bound=self.config.lower_bound,
                                    upper_bound=self.config.upper_bound,
                                )
                        if (edited_image.size() != rendered_image.size()):
                            edited_image = torch.nn.functional.interpolate(edited_image, size=rendered_image.size()[2:], mode='bilinear')
                        # encoder to style code z
                        with torch.no_grad():
                            latents = self.ip2p.imgs_to_latent(edited_image).squeeze()  
                        latent_list.append(latents)
                    latents = torch.stack(latent_list)  
                    edited_style_code = self.ip2p.latent_to_styles(latents)  # [59, 64]
                    model_outputs['edited_style_code'] = edited_style_code   
                    
             
            # loss between novel views' sytle latent code 
            if (step > 35000) and (self.model.config.use_generative):  
               
                # render multi views from edited nerf  
                latent_list = []
                for i in range(len(self.datamanager.image_batch["image_idx"])):   
                    current_spot = i
                    
                    # generate current index in datamanger
                    current_index = self.datamanager.image_batch["image_idx"][current_spot]  
                    
                    # get current camera, include camera transforms from original optimizer
                    camera_transforms = self.datamanager.train_camera_optimizer(current_index.unsqueeze(dim=0)) 
                    current_camera = self.datamanager.train_dataparser_outputs.cameras[current_index].to(self.device) 
                    current_ray_bundle = current_camera.generate_rays(torch.tensor(list(range(1))).unsqueeze(-1), camera_opt_to_camera=camera_transforms) 

                    current_ray_bundle.metadata['styles'] = current_style.clone()

                    # get current render of nerf  
                    camera_outputs = self.model.get_outputs_for_camera_ray_bundle(current_ray_bundle) 
                    rendered_image_tmp = camera_outputs["rgb"].unsqueeze(dim=0).permute(0, 3, 1, 2)   
                    
                    with torch.no_grad(): 
                        latent = self.ip2p.imgs_to_latent(rendered_image_tmp).squeeze()    
                    latent_list.append(latent)
                
                latents = torch.stack(latent_list)  
                style_code = self.ip2p.latent_to_styles(latents) 
                model_outputs['style_code'] = style_code                     
            
            if self.model.config.use_contrastive_style:
                with torch.no_grad(): 
                    latents = self.ip2p.imgs_to_latent(novel_edited_image) 
                    latents_c0 = self.ip2p.imgs_to_latent(rendered_image)
                edited_style_code = self.ip2p.latent_to_styles(latents)
                style_code_c0 = self.ip2p.latent_to_styles(latents_c0)
    
                model_outputs['style_code_si'] = edited_style_code  
                model_outputs['style_code_c0'] = style_code_c0  
                
            if self.model.config.use_depth:
                model_outputs['rendered_depth'] = rendered_depth
                model_outputs['edited_img_depth'] = pre_edited_img_depth

            loss_dict = self.model.get_loss_dict(model_outputs, batch, metrics_dict)
        
        
        else:
            #---------------------------------------bellow is the origin in2n------------------------------------------------------#
            ray_bundle, batch = self.datamanager.next_train(step)

            model_outputs = self.model(ray_bundle)   
            metrics_dict = self.model.get_metrics_dict(model_outputs, batch)
            
            # edit an image every ``edit_rate`` steps
            if (step % self.config.edit_rate == 0):  

                # edit ``edit_count`` images in a row
                for i in range(self.config.edit_count):  

                    # iterate through "spot in dataset"
                    current_spot = next(self.train_indices_order)
                    
                    # get original image from dataset
                    original_image = self.datamanager.original_image_batch["image"][current_spot].to(self.device)  
                    # generate current index in datamanger
                    current_index = self.datamanager.image_batch["image_idx"][current_spot]  

                    # get current camera, include camera transforms from original optimizer
                    camera_transforms = self.datamanager.train_camera_optimizer(current_index.unsqueeze(dim=0))   
                    current_camera = self.datamanager.train_dataparser_outputs.cameras[current_index].to(self.device)  
                    current_ray_bundle = current_camera.generate_rays(torch.tensor(list(range(1))).unsqueeze(-1), camera_opt_to_camera=camera_transforms)  

                    # get current render of nerf  # no grad
                    original_image = original_image.unsqueeze(dim=0).permute(0, 3, 1, 2)   
                    camera_outputs = self.model.get_outputs_for_camera_ray_bundle(current_ray_bundle) 
                    rendered_image = camera_outputs["rgb"].unsqueeze(dim=0).permute(0, 3, 1, 2)   

                    # delete to free up memory
                    del camera_outputs
                    del current_camera
                    del current_ray_bundle
                    del camera_transforms
                    torch.cuda.empty_cache()

                    edited_image = self.ip2p.edit_image(
                                self.text_embedding.to(self.ip2p_device),
                                rendered_image.to(self.ip2p_device),
                                original_image.to(self.ip2p_device),
                                guidance_scale=self.config.guidance_scale,
                                image_guidance_scale=self.config.image_guidance_scale,
                                diffusion_steps=self.config.diffusion_steps,
                                lower_bound=self.config.lower_bound,
                                upper_bound=self.config.upper_bound,
                            )

                    # resize to original image size (often not necessary)
                    if (edited_image.size() != rendered_image.size()):
                        edited_image = torch.nn.functional.interpolate(edited_image, size=rendered_image.size()[2:], mode='bilinear')

                    # write edited image to dataloader
                    self.datamanager.image_batch["image"][current_spot] = edited_image.squeeze().permute(1,2,0)

            loss_dict = self.model.get_loss_dict(model_outputs, batch, metrics_dict)

        return model_outputs, loss_dict, metrics_dict

    def forward(self):
        """Not implemented since we only want the parameter saving of the nn module, but not forward()"""
        raise NotImplementedError
