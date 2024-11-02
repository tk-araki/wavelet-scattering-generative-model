from pathlib import Path
import yaml

import torch
def load_config(config_file):  # config_file = 'config_wsgm_v2.yaml'; config = load_config(config_file)

    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    config['common']['root_dir'] = Path(config['common']['root_dir'])
    for param_key,param_config in config.items():
        if param_key == 'common':
            continue

        for key, value in param_config.items():
            if key.endswith(("dir","file")):
                param_config[key] = config['common']['root_dir'] / value

    return config

def fix_torch_seed(seed):
    # random.seed(seed)
    # np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
