import copy
import torch
import os
import shutil
import torch.distributed as dist
from pathlib import Path
from torch.optim import Adam, AdamW
from torch.utils.tensorboard import SummaryWriter

from .utils import cycle, exists
from torch.optim.lr_scheduler import LambdaLR


class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


# trainer class
class Trainer(object):
    def __init__(
        self,
        rank,
        diffusion_model,
        train_dl,
        val_dl,
        sample_num_of_frame,
        init_num_of_frame,
        scheduler_function,
        ema_decay=0.995,
        train_lr=1e-4,
        train_num_steps=1000000,
        scheduler_checkpoint_step=100000,
        step_start_ema=2000,
        update_ema_every=10,
        save_and_sample_every=1000,
        results_folder="./results",
        tensorboard_dir="./tensorboard_logs/diffusion-video/",
        model_name="model",
        val_num_of_batch=2,
        optimizer="adam",
    ):
        super().__init__()
        self.model = diffusion_model
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.model)
        self.update_ema_every = update_ema_every
        self.sample_num_of_frame = sample_num_of_frame
        self.val_num_of_batch = val_num_of_batch

        self.step_start_ema = step_start_ema
        self.save_and_sample_every = save_and_sample_every

        self.train_num_steps = train_num_steps

        self.train_dl_class = train_dl
        self.val_dl_class = val_dl
        self.train_dl = cycle(train_dl)
        self.val_dl = cycle(val_dl)
        if optimizer == "adam":
            self.opt = Adam(diffusion_model.parameters(), lr=train_lr)
        elif optimizer == "adamw":
            self.opt = AdamW(diffusion_model.parameters(), lr=train_lr)
        self.scheduler = LambdaLR(self.opt, lr_lambda=scheduler_function)

        self.step = 0
        self.device = rank
        self.init_num_of_frame = init_num_of_frame
        self.scheduler_checkpoint_step = scheduler_checkpoint_step

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok=True)
        self.model_name = model_name

        if os.path.isdir(tensorboard_dir):
            shutil.rmtree(tensorboard_dir)
        self.writer = SummaryWriter(tensorboard_dir)

        self.reset_parameters()

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
        else:
            self.ema.update_model_average(self.ema_model, self.model)

    def save(self):
        if self.device == 0:
            data = {
                "step": self.step,
                "model": self.model.module.state_dict(),
                "ema": self.ema_model.module.state_dict(),
            }
            idx = (self.step // self.save_and_sample_every) % 3
            torch.save(data, str(self.results_folder / f"{self.model_name}_{idx}.pt"))

    def load(self, idx=0, load_step=True):
        data = torch.load(
            str(self.results_folder / f"{self.model_name}_{idx}.pt"),
            map_location=lambda storage, loc: storage,
        )

        if load_step:
            self.step = data["step"]
        self.model.module.load_state_dict(data["model"])
        self.ema_model.module.load_state_dict(data["ema"])

    def train(self):

        while self.step < self.train_num_steps:
            if (self.step >= self.scheduler_checkpoint_step) and (self.step != 0):
                self.scheduler.step()
            data = next(self.train_dl).to(self.device)
            loss = self.model(data * 2.0 - 1.0)
            loss.backward()
            if self.device == 0:
                self.writer.add_scalar("sequence_length", data.shape[0], self.step)
                self.writer.add_scalar("loss", loss.item(), self.step)
            dist.barrier()

            self.opt.step()
            self.opt.zero_grad()

            if (self.update_ema_every > 0) and (self.step % self.update_ema_every == 0):
                self.step_ema()

            if (self.step % self.save_and_sample_every == 0) and (self.step != 0):
                # milestone = self.step // self.save_and_sample_every
                if exists(self.model.module.transform_fn) and len(self.model.module.otherlogs["predict"]) > 0:
                    self.writer.add_video(
                        f"predicted/device{self.device}",
                        (self.model.module.otherlogs["predict"].transpose(0, 1) + 1) * 0.5,
                        self.step // self.save_and_sample_every,
                    )
                for i, batch in enumerate(self.val_dl):
                    if i >= self.val_num_of_batch:
                        break
                    videos = self.ema_model.module.sample(
                        batch[: self.init_num_of_frame].to(self.device) * 2.0 - 1.0,
                        self.sample_num_of_frame,
                    )
                    videos = (videos + 1.0) * 0.5
                    self.writer.add_video(
                        f"samples_device/device{self.device}/num{i}",
                        videos.clamp(0.0, 1.0).transpose(0, 1),
                        self.step // self.save_and_sample_every,
                    )
                    self.writer.add_video(
                        f"true_frames/device{self.device}/num{i}",
                        batch.transpose(0, 1),
                        self.step // self.save_and_sample_every,
                    )
                if self.device == 0:
                    self.save()
                dist.barrier()

            self.step += 1
        if self.device == 0:
            self.save()
        print("training completed")

