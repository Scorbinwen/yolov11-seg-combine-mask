# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import os
import shutil
import socket
import sys
import tempfile

from . import USER_CONFIG_DIR
from .torch_utils import TORCH_1_9


def find_free_network_port() -> int:
    """
    Finds a free port on localhost.

    It is useful in single-node training when we don't want to connect to a real main node but have to set the
    `MASTER_PORT` environment variable.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]  # port


def generate_ddp_file(trainer):
    """Generates a DDP file and returns its file name."""
    module, name = f"{trainer.__class__.__module__}.{trainer.__class__.__name__}".rsplit(".", 1)
    content = f"""
# Ultralytics Multi-GPU training temp file (should be automatically deleted after use)

import signal
from pathlib import Path
import sys
def handler(signum, frame):
    # 终止整个进程组
    print(f'[SIGINT] Terminating process group...')
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()  # 释放分布式资源
    os.killpg(os.getpgid(os.getpid()), signal.SIGKILL)  # 杀死整个进程组
    sys.exit(1)

signal.signal(signal.SIGINT, handler)

overrides = {vars(trainer.args)}
import os
import argparse  # 新增argparse模块
# ... 原有路径设置保持不变 ...
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))
from {module} import {name}
from ultralytics.utils import DEFAULT_CFG_DICT
if __name__ == "__main__":
    try:
        cfg = DEFAULT_CFG_DICT.copy()
        cfg.update(save_dir='')   # handle the extra key 'save_dir'
        trainer = {name}(cfg=cfg, overrides=overrides)
        trainer.args.model = "{getattr(trainer.hub_session, "model_url", trainer.args.model)}"
        results = trainer.train()
    finally:
        # 确保释放分布式资源
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
"""
    with tempfile.NamedTemporaryFile(
        prefix="_temp_",
        suffix=f"{id(trainer)}.py",
        mode="w+",
        encoding="utf-8",
        dir=".",
        delete=False,
    ) as file:
        file.write(content)
        if hasattr(os, 'register_at_fork'):
            os.register_at_fork(after_in_child=lambda: os.remove(file.name))
    return file.name


def generate_ddp_command(world_size, trainer):
    """Generates and returns command for distributed training."""
    import __main__  # noqa local import to avoid https://github.com/Lightning-AI/lightning/issues/15218

    if not trainer.resume:
        shutil.rmtree(trainer.save_dir)  # remove the save_dir
    file = generate_ddp_file(trainer)
    dist_cmd = "torch.distributed.run" if TORCH_1_9 else "torch.distributed.launch"
    port = find_free_network_port()
    cmd = [sys.executable,
           "-u",  # 无缓冲输出
           "-m", dist_cmd,
           "--nproc_per_node",  f"{world_size}",
           "--master_port",  f"{port}",
           file
    ]
    return cmd, file

def kill_related_processes(file: str):
    """跨平台终止与临时文件相关的进程"""
    if sys.platform == "win32":
        # Windows 使用 tasklist 和 taskkill
        import subprocess
        try:
            # 查找包含文件名的进程
            result = subprocess.run(
                f'tasklist /FI "IMAGENAME eq python*" /FO CSV /NH',
                capture_output=True,
                text=True,
                shell=True,
                check=True)
            # 解析进程PID
            pids = [line.split(',')[1].strip('"')
                   for line in result.stdout.splitlines()
                   if file in line]
            # 终止进程
            for pid in pids:
                subprocess.run(f"taskkill /F /PID {pid}", shell=True, check=True)
        except subprocess.CalledProcessError:
            pass
    else:
        # Linux/Mac 使用 pgrep/pkill
        import subprocess
        try:
            subprocess.run(f"pkill -f {file}", shell=True, check=True)
        except subprocess.CalledProcessError:
            pass

def ddp_cleanup(trainer, file):
    """Delete temp file if created."""
    # 先终止相关进程
    kill_related_processes(file)
    # 再删除临时文件
    if f"{id(trainer)}.py" in file:  # if temp_file suffix in file
        os.remove(file)
