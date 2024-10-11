# GenN2N: Generative NeRF2NeRF Translation

###  [Paper](https://arxiv.org/abs/2404.02788) | [Project Page](https://xiangyueliu.github.io/GenN2N/) 
> [Xiangyue Liu](https://xiangyueliu.github.io/), [Han Xue](https://axian12138.github.io/assests/HanXue_resume.pdf), [Kunming Luo](https://coolbeam.github.io/), [Ping Tan](https://ece.hkust.edu.hk/pingtan)<sup>†</sup>, [Li Yi](https://ericyi.github.io/)<sup>†</sup>
>
> CVPR 2024
>
<div align=center>
<img src="assets/teaser.gif" width="100%"/>
</div>

## Abstract
We present GenN2N, a unified NeRF-to-NeRF translation framework for various NeRF translation tasks such as text-driven NeRF editing, colorization, super-resolution, inpainting, etc. Unlike previous methods designed for individual translation tasks with task-specific schemes, GenN2N achieves all these NeRF editing tasks by employing a plug-and-play image-to-image translator to perform editing in the 2D domain and lifting 2D edits into the 3D NeRF space. Since the 3D consistency of 2D edits may not be assured, we propose to model the distribution of the underlying 3D edits through a generative model that can cover all possible edited NeRFs. To model the distribution of 3D edited NeRFs from 2D edited images, we carefully design a VAE-GAN that encodes images while decoding NeRFs. The latent space is trained to align with a Gaussian distribution and the NeRFs are supervised through an adversarial loss on its renderings. To ensure the latent code does not depend on 2D viewpoints but truly reflects the 3D edits, we also regularize the latent code through a contrastive learning scheme. Extensive experiments on various editing tasks show GenN2N, as a universal framework, performs as well or better than task-specific specialists while possessing flexible generative power. 

## Setup

### Environment

* Clone this repo
    ```shell
    git clone https://github.com/Lxiangyue/GenN2N.git
    cd GenN2N
    ```
* Install dependencies to setup a conda environment:
    ```shell
    conda create -n genn2n python=3.10
    conda activate genn2n
    pip install -r requirements.txt
    ```
    We made our modifications to nerfstudio and instruct-nerf2nerf, so clone our version and install (compulsory):
    `rm -r ...` maybe not needed for the first time
    ```shell
    cd nerfstudio && rm -r nerfstudio.egg-info && pip install -e . && cd ..
    cd instruct-nerf2nerf && rm -r in2n.egg-info && pip install -e . && cd ..
    cd instruct-nerf2nerf
    ```
    
### Data
* Create 'Data' folder, and download the following datas to 'Data':
    * Download [fangzhou-small](https://drive.google.com/drive/folders/18-XLZt3pRwuOalOoS-YAgyVV-nthURPi?usp=sharing) and [fangzhou-small-editing](https://drive.google.com/drive/folders/1Ijhn2AmE6l17ZZ0LFKbPgF_ff1BpafhB?usp=sharing) for __Text-driven Editing__ task.
    * Download [redroof_palace](https://drive.google.com/drive/folders/16O4inI2nZ6FlQ7qqunTq0_J6vyqcSVwm?usp=sharing) and [redroof_palace_color](https://drive.google.com/drive/folders/17958oW6qd96zhn_rshkZad_IiU1quLbb?usp=sharing) for __Colorization__ task.
    * Download [trex_lr](https://drive.google.com/drive/folders/13UD0EOeEdzDhqpwZaz-vaQP9dgTAjqEV?usp=sharing) and [trex_SR](https://drive.google.com/drive/folders/1zpDOdyDytr68ldd84WHrB50qooeXP-gs?usp=sharing) for __Super-resolution__ task.
    * Download [statue_order](https://drive.google.com/drive/folders/1vB4vr-jp9738muop410hX2BufifQpdBh?usp=sharing) and [statue_repeat_removal](https://drive.google.com/drive/folders/183UXXIBJ_NsmgYBSAxg6xQi7hpQcRbTI?usp=sharing) for __Inpainting__ task.
    * Path organized as:
        ```shell
        /Data
            /fangzhou-small
            /fangzhou-small-editing
            /redroof_palace
            /redroof_palace_colo
            /statue_order
            /statue_repeat_removal
            /trex_lr
            /trex_SR
        /instruct-nerf2nerf
        /nerfstudio
        ```

### Model
* For inference the follwing inference examples, download and put our trained models to 'instruct-nerf2nerf/outputs':
    * Download [fangzhou-small](https://drive.google.com/drive/folders/14LTo-i2JX3bytSFYFMjR5z_io8S9MBJN?usp=sharing) for __Text-driven Editing__ task.
    * Download [redroof_palace](https://drive.google.com/drive/folders/1-bZDRSekXpTL6y3qKPmBh6QzT5-ite8I?usp=sharing) for __Colorization__ task.
    * Download [trex_lr](https://drive.google.com/drive/folders/1wj2fnZpMbzo8gxLT0ggvi74rhqzN7aF2?usp=sharing) for __Super-resolution__ task.
    * Download [statue_order](https://drive.google.com/drive/folders/1w7Q_maU6DoBnSPxL44-Mrcov8BbhGvUG?usp=sharing) for __Inpainting__ task.
    * Path organized as:
        ```shell
        /instruct-nerf2nerf
            /outputs
                /fangzhou-small
                /redroof_palace
                /statue_order
                /trex_lr
        ```

## Reproducing Experiments
### Text-driven Editing
For example, __inference__ the editing prompt *"Turn him into the Tolkien Elf"* on *fangzhou-small* with our trained model:
```shell
ns-train in2n --data ../Data/fangzhou-small/ --load-dir outputs/fangzhou-small/in2n/2023-11-27_024729/nerfstudio_models --pipeline.prompt "Turn him into the Tolkien Elf" --pipeline.edited_imgs_path "../Data/fangzhou-small-editing/fangzhou_small_elf_1.25_6.5/fangzhou*" --pipeline.edited_imgs_path_type "png"  --pipeline.inference_num_per_ckpt 30  --max_num_iterations 1 --load_step 35000
```
For example, __training__ the editing prompt *"Turn him into the Tolkien Elf"* on *fangzhou-small*:
```shell
ns-train in2n --data ../Data/fangzhou-small/ --load-dir outputs/fangzhou-small/nerfacto/2023-11-14_204006/nerfstudio_models --pipeline.prompt "Turn him into the Tolkien Elf" --pipeline.edited_imgs_path "../Data/fangzhou-small-editing/fangzhou_small_elf_1.25_6.5/fangzhou*" --pipeline.edited_imgs_path_type "png" --pipeline.inference_num_per_ckpt 30
```
### Colorization
For example, __inference__ the colorization on *redroof_palace* with our trained model:
```shell
ns-train in2n --data ../Data/redroof_palace/ --load-dir outputs/redroof_palace/in2n/2023-11-15_135805/nerfstudio_models --pipeline.prompt "Make it colorful" --pipeline.edited_imgs_path "../Data/redroof_palace_color/redroof_palace*" --pipeline.edited_imgs_path_type "jpg"  --pipeline.inference_num_per_ckpt 30  --max_num_iterations 1 --load_step 35750
```
For example, __training__ the colorization on *redroof_palace*:
```shell
ns-train in2n --data ../Data/redroof_palace/ --load-dir outputs/redroof_palace/nerfacto/2023-11-15_120546/nerfstudio_models --pipeline.prompt "Make it colorful" --pipeline.edited_imgs_path "../Data/redroof_palace_color/redroof_palace*" --pipeline.edited_imgs_path_type "jpg"  --pipeline.inference_num_per_ckpt 30  
```


### Super-resolution
For example, __inference__ the super-resolution on *trex_lr* with our trained model:
```shell
ns-train in2n --data ../Data/trex_lr/ --load-dir outputs/trex_lr/in2n/2023-11-17_040719/nerfstudio_models --pipeline.prompt "Make it colorful" --pipeline.edited_imgs_path "../Data/trex_SR/trex_*" --pipeline.edited_imgs_path_type "png" --max_num_iterations 1 --load_step 36250 
```
For example, __training__ the super-resolution on *trex_lr*:
```shell
ns-train in2n --data ../Data/trex_lr --load-dir outputs/trex_lr/nerfacto/2023-11-14_204456/nerfstudio_models --pipeline.prompt "Make the image higher resolution" --pipeline.edited_imgs_path "../Data/trex_SR/trex_*" --pipeline.edited_imgs_path_type "png" 
```
### Inpainting 
For example, __inference__ the inpainting on *statue* with our trained model:
```shell
ns-train in2n --data ../Data/statue_order/ --load-dir outputs/statue_order/in2n/2023-11-17_024339/nerfstudio_models --pipeline.prompt "remove the statue" --pipeline.edited_imgs_path "../Data/statue_repeat_removal/statue*" --pipeline.edited_imgs_path_type "png" --max_num_iterations 1 
```
For example, __training__ the inpainting on *statue*:
```shell
ns-train in2n --data ../Data/statue_order/ --load-dir outputs/statue_order/nerfacto/2024-10-08_203621/nerfstudio_models --pipeline.prompt "remove the statue" --pipeline.edited_imgs_path "../Data/statue_repeat_removal/statue*" --pipeline.edited_imgs_path_type "png" 
```

## Citation
If you find our work useful in your research, please cite:
```
@inproceedings{liu2024genn2n,
  title={GenN2N: Generative NeRF2NeRF Translation},
  author={Liu, Xiangyue and Xue, Han and Luo, Kunming and Tan, Ping and Yi, Li},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={5105--5114},
  year={2024}
}
```

## Acknowledgements
The implementation of GenN2N are based on [Instruct-NeRF2NeRF](https://github.com/ayaanzhaque/instruct-nerf2nerf). Thanks to these authors for releasing the code.
