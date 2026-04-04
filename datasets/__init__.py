import importlib


def call_load_dataset(cfg):
    """
    Resolve the appropriate dataset loader based on dataset name and output directory.

    For TTA (test-time adaptation) runs:
      - Medical 3D datasets (BraTS, Pancreas, mbh) → NII_test loader
      - 2D natural image datasets (PraNet, ISIC, etc.) → ISIC_test loader

    For standard training runs:
      - 2D natural image datasets → ISIC loader
      - 3D medical datasets → NII loader
    """
    name = cfg.dataset
    key = name.split("-")[0]

    NATURAL_2D = {'PraNet', 'ISIC', 'Kvasir', 'CVC', 'city', 'fundus'}
    MEDICAL_3D = {'VST1', 'VST2', 'mbh', 'Pancreas'}
    BRATS_PREFIXES = ('BraTS',)

    is_tta = 'tta' in cfg.out_dir

    if is_tta:
        if key.startswith(BRATS_PREFIXES) or key in MEDICAL_3D:
            loader_key = 'NII_test'
        elif key in NATURAL_2D:
            loader_key = 'ISIC_test'
        else:
            loader_key = 'NII_test'
    else:
        if key in NATURAL_2D:
            loader_key = 'ISIC'
        elif key in MEDICAL_3D:
            loader_key = 'NII'
        else:
            loader_key = 'NII_test'

    function_name = "load_datasets"
    if cfg.visual:
        function_name += "_visual"
    if cfg.prompt == "coarse":
        function_name += "_coarse"

    module = importlib.import_module(f"datasets.{loader_key}")
    return getattr(module, function_name)


def call_load_dataset_prompt(cfg):
    name = cfg.dataset
    key = name.split("-")[0]
    module = importlib.import_module(f"datasets.{key}")
    return getattr(module, "load_datasets_prompt")


def call_load_dataset_val(cfg):
    name = cfg.dataset
    key = name.split("-")[0]
    module = importlib.import_module(f"datasets.{key}")
    return getattr(module, "load_datasets_val")
