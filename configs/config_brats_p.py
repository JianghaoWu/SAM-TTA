from box import Box
from configs.base_config import base_config


config = {
    "gpu_ids": "0",
    "batch_size": 1,
    "val_batchsize": 1,
    "num_workers": 4,
    "num_iters": 40000,
    "max_nums": 40,
    "num_points": 5,
    "eval_interval": 1,
    "dataset": "BraTS_SSA_t2w_2D",
    "out_dir": "output/tent_tta/BraTS_SSA_t2w_2D", # BraTS_SSA_t2w_2D

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