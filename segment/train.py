from pathlib import Path
import sys
import os
import argparse  # 新增argparse模块

# ... 原有路径设置保持不变 ...
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

from ultralytics import YOLO
os.environ['MKL_SERVICE_FORCE_INTEL'] = 'True'

# 新增参数解析函数
def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="../ultralytics/cfg/models/11/yolo11-seg.yaml",
                        help='path to model config file')
    parser.add_argument('--cfg', type=str, default="../ultralytics/cfg/hyps/hyps.scratch-low.yaml",
                        help='path to hyperparameter config file')
    parser.add_argument('--device', type=str, default="0, 1",
                        help='comma-separated list of cuda device(s) e.g. 0 or 0,1,2,3')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_opt()  # 解析命令行参数

    # 加载模型（使用参数代替硬编码路径）
    model = YOLO(args.model)

    # 训练模型（使用参数代替硬编码配置）
    train_results = model.train(
        cfg=args.cfg,
        device=args.device
    )