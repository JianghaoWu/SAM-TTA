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
        "type": "medsam",
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
        "abd3": {
            "root_dir": "./data/MSWAL/",
            "all_list": "./data/MSWAL/train3.csv",
            "train_list": "./data/MSWAL/train3.csv",
            "val_list": "./data/MSWAL/val3.csv",
            "test_list": "./data/MSWAL/test3.csv"
        },
        "abd5": {
            "root_dir": "./data/MSWAL/",
            "all_list": "./data/MSWAL/train5.csv",
            "train_list": "./data/MSWAL/train5.csv",
            "val_list": "./data/MSWAL/val5.csv",
            "test_list": "./data/MSWAL/test5.csv"
        },
        "CVC": {
            "root_dir": "./data/CVC/",
            "all_list": "./data/CVC/train.csv",
            "train_list": "./data/CVC/train.csv",
            "val_list": "./data/CVC/val.csv",
            "test_list": "./data/CVC/test.csv"
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
        "fundus": {
            "root_dir": "./data/Fundus-doFE/",
            "all_list": "./data/Fundus-doFE/train.csv",
            "train_list": "./data/Fundus-doFE/train.csv",
            "val_list": "./data/Fundus-doFE/test.csv",
            "test_list": "./data/Fundus-doFE/test.csv"
        },
        "abd": {
            "root_dir": "./data/abd/",
            "all_list": "./data/abd/train44.csv",
            "train_list": "./data/abd/train44.csv",
            "val_list": "./data/abd/val.csv",
            "test_list": "./data/abd/test.csv"
        },
        "BraTS_SSA_t2w_2D": {
            "root_dir": "./data/BraTS_SSA_t2w_2D/",
            "all_list": "./data/BraTS_SSA_t2w_2D/train.csv",
            "train_list": "./data/BraTS_SSA_t2w_2D/train.csv",
            "test_list": "./data/BraTS_SSA_t2w_2D/test.csv",
            "val_list": "./data/BraTS_SSA_t2w_2D/val.csv",
        },
        "BraTS_SSA_t2f_2D": {
            "root_dir": "./data/BraTS_SSA_t2f_2D/",
            "all_list": "./data/BraTS_SSA_t2f_2D/train.csv",
            "train_list": "./data/BraTS_SSA_t2f_2D/train.csv",
            "test_list": "./data/BraTS_SSA_t2f_2D/test.csv",
            "val_list": "./data/BraTS_SSA_t2f_2D/val.csv",
        },
        "VST2": {
            "root_dir": "./data/VST2/",
            "train_list": "./data/VST2/train.csv",
            "test_list": "./data/VST2/test.csv",
            "val_list": "./data/VST2/val.csv",
        },
        "coco": {
            "root_dir": "./data/coco2017/val2017",            
            "annotation_file": "./data/coco2017/annotations/instances_val2017.json",
        },
        "coconut": {
            "root_dir": "./data/coconut/val2017",
            "annotation_file": "./data/coconut/coconut_dataset/annotations/annotations/relabeled_instances_val.json",
        },
        "PascalVOC": {
            "root_dir": "./data/VOC2012/",
        },
        "sa": {
            "root_dir": "./data/SA-1B",
        },
        "Polyp":{
            "root_dir": "./data/Polyp/Kvasir-SEG",
            "annotation_file": "./data/Polyp/Kvasir-SEG/kavsir_bboxes.json"
        },
        "ISTD": {
            "train": "./data/ISTD/train/train_A",
            "test": "./data/ISTD/test/test_A",
        },
        "MSD": {
            "train": "./data/MSD/train/image",
            "test": "./data/MSD/test/image",
        },
        "GDD": {
            "train": "./data/GDD/train/image",
            "test": "./data/GDD/test/image",
        },
        "CAMO":{
            "GT": "./data/CAMO-V.1.0-CVIU2019/GT",
            "train": "./data/CAMO-V.1.0-CVIU2019/Images/Train",
            "test": "./data/CAMO-V.1.0-CVIU2019/Images/Test",
        },
        "abd5": {
            "root_dir": "./data/MSWAL/",
            "all_list": "./data/MSWAL/train5.csv",
            "train_list": "./data/MSWAL/train5.csv",
            "val_list": "./data/MSWAL/val5.csv",
            "test_list": "./data/MSWAL/test5.csv"
        },
        "COD10K":{
            "GT": "./data/COD10K-v2/Test/GT_Object",
            "test": "./data/COD10K-v2/Test/Image",
        },
        "Kvasir": {
            "root_dir": "./data/Kvasir/",
            "all_list": "./data/Kvasir/train.csv",
            "train_list": "./data/Kvasir/train.csv",
            "val_list": "./data/Kvasir/val.csv",
            "test_list": "./data/Kvasir/test.csv"
        },
        "robot": {
            "OCID": "./data/OCID-dataset",
            "OSD": "./data/OSD-0.2-depth"
        },
    },
}
