"""1"""
import warnings

warnings.filterwarnings("ignore")
import sys
from pathlib import Path
import os
import torch

ROOT = os.path.dirname(os.path.abspath("."))
sys.path.append(ROOT)
from trademaster.utils import plot
import argparse
import os.path as osp
from mmcv import Config
from trademaster.utils import replace_cfg_vals
from trademaster.datasets.builder import build_dataset
from trademaster.trainers.builder import build_trainer
from trademaster.utils import set_seed
set_seed(2023)

"""2"""
parser = argparse.ArgumentParser(description='Download Alpaca Datasets')
parser.add_argument("--config", default=osp.join(ROOT, "configs", "portfolio_management", "portfolio_management_dj30_sarl_sarl_adam_mse.py"),
                    help="download datasets config file path")
parser.add_argument("--task_name", type=str, default="train")
args, _ = parser.parse_known_args()

cfg = Config.fromfile(args.config)
task_name = args.task_name
cfg = replace_cfg_vals(cfg)

"""3"""
print("cfg", cfg)

"""4"""
dataset = build_dataset(cfg)

"""5"""
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
work_dir = os.path.join(ROOT, cfg.trainer.work_dir)

if not os.path.exists(work_dir):
    os.makedirs(work_dir)
cfg.dump(osp.join(work_dir, osp.basename(args.config)))

trainer = build_trainer(cfg, default_args=dict(dataset=dataset, device = device))

"""6"""
trainer.train_and_valid()

"""7"""
import ray
from ray.tune.registry import register_env
from trademaster.environments.portfolio_management.sarl_environment import PortfolioManagementSARLEnvironment
def env_creator(env_name):
    if env_name == 'portfolio_management_sarl':
        env = PortfolioManagementSARLEnvironment
    else:
        raise NotImplementedError
    return env
ray.init(ignore_reinit_error=True)
register_env("portfolio_management_sarl", lambda config: env_creator("portfolio_management_sarl")(config))
trainer.test();

"""8"""
plot(trainer.test_environment.save_asset_memory(),alg="SARL")

print("=======================OK============================")

# cfg Config (path: C:\Users\DOCTOR SUN\Desktop\DRL论文综合\TradeMaster-1.0.0\TradeMaster-1.0.0\configs\portfolio_management\portfolio_management_dj30_sarl_sarl_adam_mse.py):
# {'data': {'type': 'PortfolioManagementDataset',
#           'data_path': 'data/portfolio_management/dj30',
#           'train_path': 'data/portfolio_management/dj30/train.csv',
#           'valid_path': 'data/portfolio_management/dj30/valid.csv',
#           'test_path': 'data/portfolio_management/dj30/test.csv',
#           'tech_indicator_list': ['high', 'low', 'open', 'close', 'adjcp', 'zopen', 'zhigh', 'zlow', 'zadjcp', 'zclose', 'zd_5', 'zd_10', 'zd_15', 'zd_20', 'zd_25', 'zd_30'],
#           'length_day': 5,
#           'initial_amount': 10000,
#           'transaction_cost_pct': 0.001,
#           'test_dynamic_path': 'data/portfolio_management/dj30/test_with_label.csv'},
#           'environment': {'type': 'PortfolioManagementSARLEnvironment'},
#           'trainer': {'type': 'PortfolioManagementSARLTrainer',
#           'agent_name': 'ddpg',
#           'if_remove': False,
#           'configs': {'framework': 'tf2', 'num_workers': 0},
#           'work_dir': 'work_dir/portfolio_management_dj30_sarl_sarl_adam_mse', 'epochs': 2},
#           'loss': {'type': 'MSELoss'},
#           'optimizer': {'type': 'Adam', 'lr': 0.001},
#           'task_name': 'portfolio_management',
#           'dataset_name': 'dj30', 'net_name': 'sarl',
#           'agent_name': 'sarl',
#           'optimizer_name': 'adam',
#           'loss_name': 'mse',
#           'work_dir': 'work_dir/portfolio_management_dj30_sarl_sarl_adam_mse'}