# encoding: utf-8
'''
@author: kang
@contact: 1021573448@qq.com
@file: labelme2yolov5.py
@time: 2022/11/17 14:08
@desc:
'''

import json
import os.path
import shutil
import sys
import argparse
from pathlib import Path
import numpy as np
import glob
import PIL.Image
import pycocotools.mask as maskUtils
from tqdm import tqdm
import yaml
from labelmap import *
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
from ultralytics.data.utils import IMG_FORMATS
import re
import PIL.Image as Image
def check_image_integrity(file_path):
    with Image.open(file_path) as img:
        img.verify()

def parse_polygon(points, shape_type):
    if shape_type == "rectangle":
        (x1, y1), (x2, y2) = points
        x1, x2 = sorted([x1, x2])
        y1, y2 = sorted([y1, y2])
        points = np.array([x1, y1, x2, y1, x2, y2, x1, y2], dtype=float)
    elif shape_type == "circle":
        (x1, y1), (x2, y2) = points
        r = np.linalg.norm([x2 - x1, y2 - y1])
        # r(1-cos(a/2))<x, a=2*pi/N => N>pi/arccos(1-x/r)
        # x: tolerance of the gap between the arc and the line segment
        n_points_circle = max(int(np.pi / np.arccos(1 - 1 / r)), 12)
        i = np.arange(n_points_circle)
        x = x1 + r * np.sin(2 * np.pi / n_points_circle * i)
        y = y1 + r * np.cos(2 * np.pi / n_points_circle * i)
        points = np.stack((x, y), axis=1).flatten().astype(float)
    elif shape_type == "polygon":
        if len(points) <= 2:
            return None
        points = np.asarray(points).flatten().astype(float)
    else:
        return None
    return points.tolist() if len(points) >= 6 else None

def main(opt):
    sys.path.append('../../')
    # load IMPORT_CLS from cfg yaml file
    with open(opt.cfg) as stream:
        try:
            cfg_dict = (yaml.safe_load(stream))
            IMPORT_CLS = list(cfg_dict["names"].values())

        except yaml.YAMLError as exc:
            print(exc)

    anno_pair_dict, change_dict, _ = mapping_cfg_dict[opt.mapping_version]

    if not os.path.exists(opt.move_path):
        os.makedirs(opt.move_path)

    im_files = glob.glob(os.path.join(opt.data_dir, "**/*.*"), recursive=True)

    im_files = [x for x in im_files if x.rsplit('.', 1)[-1].lower() in IMG_FORMATS]
    im_files = [x for x in im_files if not x.rsplit('.', 1)[0].endswith('_gt')]

    for img_path in tqdm(im_files):
        parent, filename = os.path.split(img_path)
        im_name, im_ext = os.path.splitext(filename)
        gt_path = os.path.join(parent, f'{im_name}_gt{im_ext}')
        json_path = os.path.join(parent, f'{im_name}.json')
        out_txt = os.path.join(parent, f'{im_name}.txt')

        try:
            with open(json_path, 'r', encoding='UTF-8') as fp:
                data = json.load(fp)
            assert data['imagePath'] == filename, "imagePath mismatch with filename"

            has_gt = os.path.exists(gt_path)
            if has_gt:
                check_image_integrity(gt_path)
            check_image_integrity(img_path)

            w, h = data['imageWidth'], data['imageHeight']
            if not has_gt and os.path.exists(img_path):
                with PIL.Image.open(img_path) as img:
                    h, w = img.height, img.width
                    has_gt = h * 2 == w

            out_lines = []
            while len(data['shapes']) > 0:
                anno = data['shapes'][0]
                del data['shapes'][0]
                ori_label = anno['label']
                if re.match(anno_pair_dict['match_reg_name'], ori_label):
                    label = change_dict.get(anno_pair_dict['change_dict_name'], None)
                    anno_pair = [anno]
                    idxes = []
                    for i, a in enumerate(data['shapes']):
                        if a['label'] == ori_label:
                            anno_pair.append(a)
                        else:
                            idxes.append(i)
                    data['shapes'] = [data['shapes'][i] for i in idxes]
                    assert len(anno_pair) == 2, f'anno_pair has {len(anno_pair)} elements'
                    points = [parse_polygon(a["points"], a.get("shape_type", "")) for a in anno_pair]
                else:
                    label = change_dict.get(ori_label, None)
                    points = [parse_polygon(anno["points"], anno.get("shape_type", ""))]

                if not all(points):
                    raise TypeError("points should not be None")

                try:
                    label_ind = IMPORT_CLS.index(label)
                except:
                    print(f"{label} not in IMPORT_CLS ori_label is {ori_label}")
                    raise

                rles = maskUtils.frPyObjects(points, h, h)
                rle = maskUtils.merge(rles)
                out_line = ' '.join([str(label_ind), rle["counts"].decode("utf-8")]) + '\n'
                out_lines.append(out_line)

            assert len(out_lines) > 0, "defect list should be non empty"
            assert os.path.exists(img_path), "fail to find image"
            assert has_gt, "fail to find gerber image"

            with open(out_txt, 'w') as f:
                f.writelines(out_lines)

        except Exception as e:
            print(f'file in [json_path, gt_path, img_path, out_txt] has error: {e}')
            for src_file in [json_path, gt_path, img_path, out_txt]:
                if os.path.exists(src_file):
                    dst_path = os.path.join(opt.move_path, os.path.relpath(src_file, opt.data_dir))
                    Path(dst_path).parent.mkdir(parents=True, exist_ok=True)
                    try:
                        shutil.move(src_file, dst_path)
                    except Exception as e:
                        print(f'{e}')

def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--move_path', type=str, default="/home/wxb/Data/move_path", help='path to save error json file')
    parser.add_argument('--data_dir', type=str, default="/home/wxb/Data/nei_xin/gt_train", help='path to process data')
    parser.add_argument('--cfg', type=str, default='../data/neiceng_40cls.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--mapping_version', type=str, default="v2", help='match rule version for maobian anno pair')
    return parser.parse_known_args()[0] if known else parser.parse_args()

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
