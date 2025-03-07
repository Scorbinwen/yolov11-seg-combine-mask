from pathlib import Path
import sys
import os
import yaml
import argparse
import tempfile
os.environ['MKL_SERVICE_FORCE_INTEL'] = 'True'
# ... 保留原有路径设置 ...
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))
from ultralytics import YOLO

def parse_opt():
    # 基础解析器只处理model和cfg
    base_parser = argparse.ArgumentParser()
    base_parser.add_argument('--model', default="../ultralytics/cfg/models/11/yolo11-seg.yaml", help='模型配置文件路径')
    base_parser.add_argument('--cfg', default="../ultralytics/cfg/hyps/hyps.scratch-low.yaml", help='超参数配置文件路径')
    base_args, unknown = base_parser.parse_known_args()

    # 加载YAML配置
    with open(base_args.cfg) as f:
        hyp = yaml.safe_load(f)

    # 校验未知参数合法性
    valid_params = hyp.keys()
    for arg in unknown:
        if arg.startswith('--'):
            param_name = arg[2:]
            if param_name not in valid_params:
                raise ValueError(f"非法参数: '{param_name}' 不在配置文件中")

    # 创建包含所有合法参数的完整解析器
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default=base_args.model)
    parser.add_argument('--cfg', default=base_args.cfg)
    existing_flags = [a.option_strings for a in parser._actions]
    # 动态添加YAML中的参数
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

    model = YOLO(args.model)

    with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.yaml',
            dir="../ultralytics/cfg/hyps/",
            delete=True) as temp_file:
        yaml.dump(cfg_params, temp_file)
        temp_file.flush()  # 确保写入磁盘
        # 执行训练
        model.train(
            cfg=temp_file.name  # 使用临时配置文件
        )