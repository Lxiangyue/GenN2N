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
Code to train model, only needed in order to not save InstructPix2Pix checkpoints
"""
from dataclasses import dataclass, field
from typing import Type
import torch
from nerfstudio.engine.trainer import Trainer, TrainerConfig
from nerfstudio.utils.decorators import check_main_thread

@dataclass
class InstructNeRF2NeRFTrainerConfig(TrainerConfig):
    """Configuration for the InstructNeRF2NeRFTrainer."""
    _target: Type = field(default_factory=lambda: InstructNeRF2NeRFTrainer)


class InstructNeRF2NeRFTrainer(Trainer):
    """Trainer for InstructNeRF2NeRF (only difference is that it doesn't save InstructPix2Pix checkpoints)"""

    @check_main_thread
    def save_checkpoint(self, step: int) -> None:
        """Save the model and optimizers
        Args:
            step: number of steps in training for given checkpoint
        """
        # possibly make the checkpoint directory
        if not self.checkpoint_dir.exists():
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        # save the checkpoint
        import os
        with open(os.path.join(self.checkpoint_dir, '..','models.txt'), 'w') as f:  # xh
            f.write(str(self.pipeline))
        ckpt_path = self.checkpoint_dir / f"step-{step:09d}.ckpt"
        pipeline_state_dict = {k: v for k, v in self.pipeline.state_dict().items() if ("ip2p." not in k)or('ip2p.style_body' in k)}
        torch.save(
            {
                "step": step,
                "pipeline": self.pipeline.module.state_dict()  # type: ignore
                if hasattr(self.pipeline, "module")
                else pipeline_state_dict,
                "optimizers": {k: v.state_dict() for (k, v) in self.optimizers.optimizers.items()},
                "scalers": self.grad_scaler.state_dict(),
            },
            ckpt_path,
        )
        
        # backup ckpt
        if step % 2500 == 0:
            ckpt_path_back = self.checkpoint_dir / f"{step:09d}.ckpt"
            torch.save(
                {
                    "step": step,
                    "pipeline": self.pipeline.module.state_dict()  # type: ignore
                    if hasattr(self.pipeline, "module")
                    else pipeline_state_dict,
                    "optimizers": {k: v.state_dict() for (k, v) in self.optimizers.optimizers.items()},
                    "scalers": self.grad_scaler.state_dict(),
                },
                ckpt_path_back,
            )
        
        # possibly delete old checkpoints
        if self.config.save_only_latest_checkpoint:
            # delete everything else in the checkpoint folder
            for f in self.checkpoint_dir.glob("step*"): # xh
                if f != ckpt_path:
                    f.unlink()

