from pathlib import Path
import sys
import os
import yaml
import argparse
import tempfile

# 设置环境变量和路径
os.environ['MKL_SERVICE_FORCE_INTEL'] = 'True'
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

from ultralytics import YOLO


def parse_opt():
    # 基础参数解析
    base_parser = argparse.ArgumentParser()
    base_parser.add_argument('--model', required=True, help='模型权重路径（如yolov11s-seg.pt）')
    base_parser.add_argument('--cfg', default="../ultralytics/cfg/hyps/hyps.scratch-low.yaml",
                             help='超参数配置文件路径')
    base_args, unknown = base_parser.parse_known_args()

    # 加载YAML配置
    with open(base_args.cfg) as f:
        hyp = yaml.safe_load(f)

    # 校验未知参数
    valid_params = hyp.keys()
    for arg in unknown:
        if arg.startswith('--'):
            param_name = arg[2:]
            if param_name not in valid_params:
                raise ValueError(f"非法参数: '{param_name}' 不在配置文件中")

    # 创建完整参数解析器
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--cfg', default=base_args.cfg)

    # 动态添加参数
    existing_flags = [a.option_strings for a in parser._actions]
    for param, value in hyp.items():
        param_type = type(value)
        if [f'--{param}'] not in existing_flags:
            parser.add_argument(f'--{param}',
                                type=param_type,
                                default=value,
                                help=f'(默认来自YAML) {param_type.__name__} 类型参数')

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_opt()

    # 加载最终配置
    with open(args.cfg) as f:
        cfg_params = yaml.safe_load(f)

    # 合并命令行覆盖值
    command_overrides = {k: v for k, v in vars(args).items()
                         if k not in ['model', 'cfg'] and v != cfg_params.get(k)}
    cfg_params.update(command_overrides)

    # 加载模型
    model = YOLO(args.model)
    # 执行验证
    model.val(**cfg_params)