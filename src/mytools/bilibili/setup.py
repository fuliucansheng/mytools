# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import os

ubuntu_commands = [
    "sudo apt update -y",
    "sudo apt install -y ffmpeg",
    "pip3 install --upgrade bilibili-api-python",
]

for command in ubuntu_commands:
    os.system(command)
