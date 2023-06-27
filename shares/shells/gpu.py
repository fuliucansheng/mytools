# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import time
import torch

tensor = torch.rand(10240, 10240).cuda()

while True:
    _ = torch.matmul(tensor, tensor)

