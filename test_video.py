from data import load_data
import argparse
import config_test as config
import os
import torch

from modules.denoising_diffusion import GaussianDiffusion
from modules.unet import Unet
from modules.temporal_models import HistoryNet, CondNet
from torchvision.utils import save_image
from torchvision.io import write_video
from joblib import Parallel, delayed

# import torchvision.transforms.functional as VF

parser = argparse.ArgumentParser(description="values from bash script")
parser.add_argument("--device", type=int, required=True, help="cuda device")
args = parser.parse_args()


def get_dim(data_config):
    return 48 if data_config["img_size"] == 64 else 64


def get_transform_mults(data_config):
    return (1, 2, 3, 4) if data_config["img_size"] in [128, 256] else (1, 2, 2, 4)


def get_main_mults(data_config):
    return (1, 1, 2, 2, 4, 4) if data_config["img_size"] in [128, 256] else (1, 2, 4, 8)


for data_config in config.data_configs:
    train_data, val_data = load_data(
        data_config, config.BATCH_SIZE, pin_memory=False, num_workers=8, distributed=False
    )
    for pred_mode in config.pred_modes:
        for transform_mode in config.transform_modes:
            model_name = f"{config.backbone}-{config.optimizer}-{pred_mode}-l1-{data_config['dataset_name']}-d{get_dim(data_config)}-t{config.iteration_step}-{transform_mode}-al{config.aux_loss}{config.additional_note}"
            results_folder = os.path.join(config.result_root, f"{model_name}")
            loaded_param = torch.load(
                str(f"{results_folder}/{model_name}_{0}.pt"),
                map_location=lambda storage, loc: storage,
            )

            denoise_model = Unet(
                dim=get_dim(data_config),
                context_dim_factor=config.context_dim_factor,
                channels=data_config["img_channel"],
                dim_mults=get_main_mults(data_config),
            )

            context_model = CondNet(
                dim=int(config.context_dim_factor * get_dim(data_config)),
                channels=data_config["img_channel"],
                backbone=config.backbone,
                dim_mults=get_main_mults(data_config),
            )

            transform_model = (
                HistoryNet(
                    dim=int(config.transform_dim_factor * get_dim(data_config)),
                    channels=data_config["img_channel"],
                    context_mode=transform_mode,
                    backbone=config.backbone,
                    dim_mults=get_transform_mults(data_config),
                )
                if transform_mode in ["residual", "transform", "flow", "ll_transform"]
                else None
            )

            model = GaussianDiffusion(
                denoise_fn=denoise_model,
                history_fn=context_model,
                transform_fn=transform_model,
                pred_mode=pred_mode,
                clip_noise=config.clip_noise,
                timesteps=config.iteration_step,
                loss_type=config.loss_type,
                aux_loss=config.aux_loss,
            )

            if 'kth' in data_config["dataset_name"]:
                N_SAMPLED = 12
            else:
                N_SAMPLED = 16

            model.load_state_dict(loaded_param["model"])
            print("loaded!")
            model.eval()
            model.to(args.device)
            for k, batch in enumerate(val_data):
                if k >= config.N_BATCH:
                    break
                for i, b in enumerate(
                    batch[config.N_CONTEXT : config.N_CONTEXT + N_SAMPLED].transpose(0, 1)
                ):
                    if not os.path.isdir(f"evaluate/truth/{model_name}"):
                        os.mkdir(f"evaluate/truth/{model_name}")
                    Parallel(n_jobs=4)(
                        delayed(save_image)(f, f"evaluate/truth/{model_name}/{k}-{i}-{j}.png")
                        for j, f in enumerate(b.cpu())
                    )
                    # write_video(
                    #     f"evaluate/truth/{model_name}/{k}-{i}.mp4",
                    #     torch.round(255 * b.permute(0, 2, 3, 1)).expand(-1,-1,-1,3),
                    #     fps=4,
                    # )
                batch = (batch - 0.5) * 2.0
                batch = batch.to(args.device)
                sampled = model.sample(
                    init_frames=batch[: config.N_CONTEXT], num_of_frames=N_SAMPLED
                ).transpose(
                    0, 1
                )  # N T C H W
                sampled = (sampled + 1.0) / 2.0
                for i, b in enumerate(sampled.clamp(0, 1)):
                    if not os.path.isdir(f"evaluate/generated/{model_name}"):
                        os.mkdir(f"evaluate/generated/{model_name}")
                    Parallel(n_jobs=4)(
                        delayed(save_image)(f, f"evaluate/generated/{model_name}/{k}-{i}-{j}.png")
                        for j, f in enumerate(b.cpu())
                    )
                    # write_video(
                    #     f"evaluate/generated/{model_name}/{k}-{i}.mp4",
                    #     torch.round(255 * b.permute(0, 2, 3, 1)).expand(-1,-1,-1,3).cpu(),
                    #     fps=4,
                    # )
