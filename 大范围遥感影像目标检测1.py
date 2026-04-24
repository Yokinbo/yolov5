import os
import numpy as np
import torch
import rasterio
from rasterio.windows import Window
from tqdm import tqdm
from pathlib import Path
from PIL import Image
from sahi.models import Yolov5Model  # 使用SAHI的 Yolov5Model
from sahi.utils import read_image
from torchvision.ops import nms  # 使用torchvision中的nms
from utils.general import scale_boxes
import fiona
from rasterio.transform import from_origin

# ===================== 你只改这里 =====================
INPUT_TIF = "/mnt/f/金三角原始下载影像/简单测试可删/矩形区域/Level16/矩形区域.tif"  # 输入GeoTIFF文件路径
WEIGHTS = "/mnt/e/YOLO/yolov5/runs/train/exp91/weights/best.pt"  # YOLOv5模型权重路径
OUT_TIF = "/mnt/f/金三角原始下载影像//简单测试可删/矩形区域/Level16/推理结果/16pred.tif"  # 输出结果路径

TILE = 1024  # 动态切片大小，可以更改为 512, 640 或其他尺寸
OVERLAP = 128  # 重叠区域大小
# =====================================================

# SAHI模型加载和推理
class SAHIYOLOv5:
    def __init__(self, weight_path, device="cuda", imgsz=1024, conf=0.25, iou=0.45):
        self.device = device
        self.model = Yolov5Model(weight_path, device=device)  # 使用SAHI的模型接口
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.names = self.model.names

    def infer(self, img):
        # 使用SAHI提供的推理方法，进行切片推理
        detections = self.model.detect(img, conf_thres=self.conf, iou_thres=self.iou)
        return detections


# 计算滑窗的起始位置
def make_starts(length: int, tile: int, stride: int):
    """生成滑窗起点列表，保证覆盖到最后一个像素。"""
    if length <= tile:
        return [0]
    starts = list(range(0, length - tile + 1, stride))
    last = length - tile
    if starts[-1] != last:
        starts.append(last)
    return starts

# 像素坐标 -> 地理坐标
def pixel_to_geo(x, y, gt):
    geo_x = gt[0] + x * gt[1] + y * gt[2]
    geo_y = gt[3] + x * gt[4] + y * gt[5]
    return geo_x, geo_y

# 全局 NMS
def global_nms(all_predictions, iou_thres=0.5):
    # 提取出所有的框和得分
    boxes = []
    scores = []
    for prediction in all_predictions:
        boxes.append(prediction['boxes'])
        scores.append(prediction['scores'])

    # 执行NMS
    boxes = torch.tensor(boxes, dtype=torch.float32)
    scores = torch.tensor(scores)
    keep = nms(boxes, scores, iou_thres)  # 使用 torchvision 中的 NMS 函数
    return keep.tolist()


def main():
    stride = TILE - OVERLAP
    center_margin = OVERLAP // 2

    if not os.path.exists(INPUT_TIF):
        raise SystemExit(f"[错误] 找不到输入影像：{INPUT_TIF}")

    # 使用 rasterio 读取影像
    with rasterio.open(INPUT_TIF) as src:
        W, H = src.width, src.height
        count = src.count
        crs = src.crs
        transform = src.transform
        dtype = src.dtypes[0]

        if count < 3:
            raise SystemExit(f"[错误] 输入影像波段数={count}，少于3（需要RGB三波段）。")

        xs = make_starts(W, TILE, stride)
        ys = make_starts(H, TILE, stride)
        total = len(xs) * len(ys)

        print("========== 推理参数 ==========")
        print(f"INPUT_TIF     : {INPUT_TIF}")
        print(f"OUT_TIF       : {OUT_TIF}")
        print(f"W,H           : {W},{H}")
        print(f"TILE/OVERLAP  : {TILE}/{OVERLAP}")
        print(f"stride        : {stride}")
        print(f"center_margin : {center_margin}")
        print(f"tiles total   : {total}")
        print("==============================")

        # 输出：单波段类别ID图（uint8）
        profile = src.profile.copy()
        profile.update(
            driver="GTiff",
            count=1,
            dtype="uint8",
            compress="lzw",
            tiled=True,
            blockxsize=512,
            blockysize=512,
            nodata=0,          # 0 作为背景/无效（按你数据习惯）
        )

        os.makedirs(os.path.dirname(OUT_TIF), exist_ok=True)

        # 可选：跳过“几乎全黑”的tile（加速、也避免黑边处误检）
        SKIP_BLACK_RATIO = 0.98
        BLACK_TH = 3  # <=3 当作黑

        all_predictions = []  # 保存所有切片的目标框

        with rasterio.open(OUT_TIF, "w", **profile) as dst:
            pbar = tqdm(total=total, ncols=100)
            skipped = 0

            for y0 in ys:
                for x0 in xs:
                    # 读一个tile（boundless=True：边缘自动补0）
                    win = Window(x0, y0, TILE, TILE)
                    patch = src.read(
                        indexes=[1, 2, 3],
                        window=win,
                        boundless=True,
                        fill_value=0
                    )  # (3, TILE, TILE)

                    # 转到 (H,W,C)
                    patch = np.transpose(patch, (1, 2, 0))

                    # dtype 兜底（避免某些tif是uint16/float）
                    if patch.dtype != np.uint8:
                        patch = np.clip(patch, 0, 255).astype(np.uint8)

                    # 黑块跳过
                    if SKIP_BLACK_RATIO <= 1.0:
                        black_ratio = np.mean(np.all(patch <= BLACK_TH, axis=2))
                        if black_ratio >= SKIP_BLACK_RATIO:
                            skipped += 1
                            pbar.update(1)
                            continue

                    # 调模型：输出是 PIL 的 “类别ID图”
                    patch_pil = Image.fromarray(patch)
                    results = model.infer(np.array(patch_pil))  # 使用 SAHI YOLOv5 模型进行推理
                    pred = results.pandas().xywh

                    # 将预测结果保存到 all_predictions 中
                    all_predictions.append({
                        'boxes': pred['boxes'].cpu().numpy(),  # 转换为 numpy 数组
                        'scores': pred['scores'].cpu().numpy()  # 转换为 numpy 数组
                    })

                    # 只取中心区域拼接（边缘块自动放宽到贴边，保证全覆盖）
                    left_in = center_margin
                    right_in = TILE - center_margin
                    top_in = center_margin
                    bot_in = TILE - center_margin

                    # 输出整图写入范围（对应的全局坐标）
                    left_out = x0 + left_in
                    right_out = x0 + right_in
                    top_out = y0 + top_in
                    bot_out = y0 + bot_in

                    if x0 == 0:
                        left_in = 0
                        left_out = 0
                    if x0 == xs[-1]:
                        right_in = TILE
                        right_out = W
                    if y0 == 0:
                        top_in = 0
                        top_out = 0
                    if y0 == ys[-1]:
                        bot_in = TILE
                        bot_out = H

                    left_out = max(0, left_out)
                    top_out = max(0, top_out)
                    right_out = min(W, right_out)
                    bot_out = min(H, bot_out)

                    # 对应裁剪后的中心块
                    cut = pred[top_in:top_in + (bot_out - top_out),
                               left_in:left_in + (right_out - left_out)]

                    # 写入输出（单波段，写 window）
                    out_win = Window(left_out, top_out, right_out - left_out, bot_out - top_out)
                    dst.write(cut[np.newaxis, :, :], window=out_win)

                    pbar.update(1)

            pbar.close()

        # 使用 NMS 对预测框进行合并
        final_predictions = global_nms(all_predictions, iou_thres=0.5)
        print(f"[完成] 输出：{OUT_TIF}")
        if SKIP_BLACK_RATIO <= 1.0:
            print(f"[提示] 黑块跳过：{skipped} / {total}（阈值={SKIP_BLACK_RATIO}）")
        print("输出是“类别ID栅格”（单波段uint8）。在QGIS/ArcGIS里用“唯一值/调色板”渲染即可。")

if __name__ == "__main__":
    main()
