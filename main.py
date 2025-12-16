from sys import argv

from follower.training_config import Experiment
from follower.training_utils import run, create_sf_config

from ccp.training_config import Experiment as ExperimentCCP
from ccp.training_utils import run as run_ccp
from ccp.training_utils import create_sf_config as create_sf_config_ccp


def recursive_update(experiment: dict, key, value):
    if key in experiment:
        experiment[key] = value
        return True
    else:
        for k, v in experiment.items():
            if isinstance(v, dict):
                if recursive_update(v, key, value):
                    return True
        return False


def update_dict(target_dict, keys, values):
    for key, value in zip(keys, values):
        if recursive_update(target_dict, key, value):
            print(f'Updated {key} to {value}')
        else:
            raise KeyError(f'Could not find {key} in experiment')


def parse_args_to_items(argv_):
    keys = []
    values = []

    for arg in argv_[1:]:
        key, value = arg.split('=')
        key = key.replace('--', '')

        keys.append(key)
        values.append(value)

    return keys, values


def main():
    experiment = Experiment() # 创建实验
    experiment = create_sf_config(experiment).__dict__
    keys, values = parse_args_to_items(list(argv))

    # check all args and replace them in experiment recursively
    update_dict(experiment, keys, values)
    run(config=experiment)

def main_ccp():
    experiment = ExperimentCCP() # 创建实验
    experiment = create_sf_config(experiment).__dict__
    keys, values = parse_args_to_items(list(argv))

    # check all args and replace them in experiment recursively
    update_dict(experiment, keys, values)
    run_ccp(config=experiment)


if __name__ == '__main__':
    # main()
    main_ccp()