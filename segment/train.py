from pathlib import Path
import sys
import os
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from ultralytics import YOLO

# Load a model
model = YOLO("../weights/yolo11l-seg.pt")

# Train the model
train_results = model.train(
    cfg="../ultralytics/cfg/hyps/hyps.scratch-low.yaml",  # path to dataset YAML
    device="1",  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
)
