base_config = {
    "eval_interval": 1,
    "ema_rate": 0.9999,
    "get_prompt": False,
    "split": True,
    "csv_keys": ["Name", "Prompt", "Mean Dice", "iters"],
    "opt": {
        "learning_rate": 1e-4,
        "weight_decay": 1e-4,
        "decay_factor": 10,
        "steps": [60000, 86666],
        "warmup_steps": 250,
    },
    "corruptions": [
        "gaussian_noise",
        "shot_noise",
        "impulse_noise",
        "defocus_blur",
        "glass_blur",
        "motion_blur",
        "zoom_blur",
        "snow",
        "frost",
        "fog",
        "brightness",
        "contrast",
        "elastic_transform",
        "pixelate",
        "jpeg_compression",
    ],
    "model": {
        "type": "vit_b",
        "checkpoint": "./checkpoints/",
        "ckpt": "",
        "freeze": {
            "image_encoder": True,
            "prompt_encoder": True,
            "mask_decoder": True,
        },
    },
    "datasets": {
        "PraNet": {
            "root_dir": "./data/PraNet/",
            "all_list": "./data/PraNet/train.csv",
            "train_list": "./data/PraNet/train.csv",
            "val_list": "./data/PraNet/val.csv",
            "test_list": "./data/PraNet/test.csv"
        },
        "fundus": {
            "root_dir": "./data/Fundus-doFE/",
            "all_list": "./data/Fundus-doFE/train.csv",
            "train_list": "./data/Fundus-doFE/train.csv",
            "val_list": "./data/Fundus-doFE/test.csv",
            "test_list": "./data/Fundus-doFE/test.csv"
        },
        "Pancreas": {
            "root_dir": "./data/Pancreas/",
            "all_list": "./data/Pancreas/train.csv",
            "train_list": "./data/Pancreas/train.csv",
            "val_list": "./data/Pancreas/val.csv",
            "test_list": "./data/Pancreas/test.csv"
        },
        "city": {
            "root_dir": "./data/city/",
            "all_list": "./data/city/train.csv",
            "train_list": "./data/city/train.csv",
            "val_list": "./data/city/val.csv",
            "test_list": "./data/city/test.csv"
        },
        "CVC": {
            "root_dir": "./data/CVC/",
            "all_list": "./data/CVC/train.csv",
            "train_list": "./data/CVC/train.csv",
            "val_list": "./data/CVC/val.csv",
            "test_list": "./data/CVC/test.csv"
        },
        "Kvasir": {
            "root_dir": "./data/Kvasir/",
            "all_list": "./data/Kvasir/train.csv",
            "train_list": "./data/Kvasir/train.csv",
            "val_list": "./data/Kvasir/val.csv",
            "test_list": "./data/Kvasir/test.csv"
        },
        "mbh": {
            "root_dir": "./data/mbh/",
            "all_list": "./data/mbh/train.csv",
            "train_list": "./data/mbh/train.csv",
            "val_list": "./data/mbh/val.csv",
            "test_list": "./data/mbh/test.csv"
        },
        "ISIC": {
            "root_dir": "./data/ISIC/",
            "all_list": "./data/ISIC/train.csv",
            "train_list": "./data/ISIC/train.csv",
            "test_list": "./data/ISIC/test.csv",
            "val_list": "./data/ISIC/val.csv",
        },
        "BraTS_PED_t2f_2D": {
            "root_dir": "./data/BraTS_PED_t2f_2D/",
            "all_list": "./data/BraTS_PED_t2f_2D/train.csv",
            "train_list": "./data/BraTS_PED_t2f_2D/train.csv",
            "test_list": "./data/BraTS_PED_t2f_2D/test.csv",
            "val_list": "./data/BraTS_PED_t2f_2D/val.csv",
        },
        "BraTS_PED_t2w_2D": {
            "root_dir": "./data/BraTS_PED_t2w_2D/",
            "all_list": "./data/BraTS_PED_t2w_2D/train.csv",
            "train_list": "./data/BraTS_PED_t2w_2D/train.csv",
            "test_list": "./data/BraTS_PED_t2w_2D/test.csv",
            "val_list": "./data/BraTS_PED_t2w_2D/val.csv",
        },
        "BraTS_SSA_t2w_2D": {
            "root_dir": "./data/BraTS_SSA_t2w_2D",
            "all_list": "./data/BraTS_SSA_t2w_2D/train.csv",
            "train_list": "./data/BraTS_SSA_t2w_2D/train.csv",
            "test_list": "./data/BraTS_SSA_t2w_2D/test.csv",
            "val_list": "./data/BraTS_SSA_t2w_2D/val.csv",
        },
        "BraTS_SSA_t2f_2D": {
            "root_dir": "./data/BraTS_SSA_t2f_2D",
            "all_list": "./data/BraTS_SSA_t2f_2D/train.csv",
            "train_list": "./data/BraTS_SSA_t2f_2D/train.csv",
            "test_list": "./data/BraTS_SSA_t2f_2D/test.csv",
            "val_list": "./data/BraTS_SSA_t2f_2D/val.csv",
        }
        },
    }

