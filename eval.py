from env.create_eval_env import create_env_base, Environment
from pogema_toolbox.evaluator import evaluation
from pogema import BatchAStarAgent

from pogema_toolbox.eval_utils import initialize_wandb, save_evaluation_results

from pathlib import Path
import wandb

import yaml

from pogema_toolbox.registry import ToolboxRegistry

from ccp.inference import CCPInference, CCPInferenceConfig
from ccp.preprocessing import ccp_preprocessor

PROJECT_NAME = 'pogema-toolbox'
BASE_PATH = Path('experiments')


def main(disable_wandb=False):
    ToolboxRegistry.register_env('Pogema-v0', create_env_base, Environment)
    ToolboxRegistry.register_algorithm('CCP', CCPInference, CCPInferenceConfig,
                                        ccp_preprocessor)

    test_maps_path = Path("env/test-maps.yaml")

    with open(test_maps_path, 'r') as f:
        maps_to_register = yaml.safe_load(f)
    ToolboxRegistry.register_maps(maps_to_register)


    folder_names = [
        '01-random-20x20',
        '02-mazes',
        '03-den520d',
        '04-Paris_1',
        '05-warehouse',
        '06-movingai',
        # '07-pathfinding',
        '08-puzzles',
    ]

    for folder in folder_names:
        maps_path = BASE_PATH / folder / "maps.yaml"
        if maps_path.exists():
            with open(maps_path, 'r') as f:
                maps_to_register_set = yaml.safe_load(f)
            ToolboxRegistry.register_maps(maps_to_register_set)

        config_path = BASE_PATH / folder / f"{Path(folder).name}.yaml"
        eval_dir = BASE_PATH / folder

        with open(config_path) as f:
            evaluation_config = yaml.safe_load(f)
        if folder == 'eval-fast':
            disable_wandb = True

        # initialize_wandb(evaluation_config, eval_dir, disable_wandb, PROJECT_NAME)
        evaluation(evaluation_config, eval_dir=eval_dir)
        # save_evaluation_results(eval_dir)
        wandb.finish()


if __name__ == '__main__':
    main()
