# -*- coding:utf-8 -*-
"""
作者: xyj
日期: 2023-01-23
说明: 
"""

import datetime
import torch
from config import Param
from utils import setup_seed
from frame.my_frame import Frame
import os


def modify_args(_args):
    _args.data_name = 'FewRel'  # 'FewRel' or 'TACRED'
    _args.task_name = _args.data_name
    _args.rel_per_task = 8 if _args.data_name == 'FewRel' else 4
    _args.batch_size = 16 # 16
    _args.total_round = 1 # 5
    _args.step1_epochs = 5 # 5
    _args.step2_epochs = 5 # 10 
    _args.data_augmentation = False # True
    _args.num_protos = 20 # 10
    _args.seed = 2021 # 2021
    _args.data_augmentation_type = [0,1,2]  # 0: 置换; 1: 聚焦; 2: 翻转;
    log_name = f'{datetime.date.today()}_{_args.data_name}_da{_args.data_augmentation}_3_np{_args.num_protos}_ep1_{_args.step1_epochs}_' \
               f'ep2_{_args.step2_epochs}_seed{_args.seed}_daType{_args.data_augmentation_type}.txt'
    _args.log_path = os.path.join(_args.result_path, log_name)
    return _args


def run(_args):
    setup_seed(_args.seed)
    frame = Frame(_args)
    frame.train(_args)


if __name__ == '__main__':
    param = Param()  # There are detailed hyper-parameter configurations.
    args = param.get_args()

    args.device = torch.device(args.device)
    args.n_gpu = torch.cuda.device_count()

    args = modify_args(args)

    torch.cuda.set_device(args.gpu)
    run(args)
