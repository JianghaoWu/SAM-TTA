from box import Box
from configs.base_config import base_config


config = {
    "gpu_ids": "1",
    "batch_size": 1,
    "val_batchsize": 1,
    "num_workers": 4,
    "num_iters": 40000,
    "max_nums": 40,
    "num_points": 5,
    "eval_interval": 1,
    "dataset": "ISIC",
    "out_dir": "output/mt_tta/ISIC", # ISIC

    "prompt": "point",      # Prompt type: box, point, coarse
    "name": "tent",
    "augment": True,
    "corrupt": None,
    "visual": False,
    "opt": {
        "learning_rate": 1e-4,
    },
    "model": {
        "type": "vit_b",
    },
}

cfg = Box(base_config)
cfg.merge_update(config)