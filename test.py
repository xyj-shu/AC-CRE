# -*- coding:utf-8 -*-
"""
作者: xyj
日期: 2023-01-24
说明: 
"""
import torch

cardinality_per_samples = torch.Tensor([31, 32, 40, 27, 32, 22, 34, 28, 22, 25, 32, 32, 33, 14, 33, 0])

print(cardinality_per_samples)

for i, val in enumerate(cardinality_per_samples):
    if val == 0:
        cardinality_per_samples[i] = 1

print(cardinality_per_samples)
