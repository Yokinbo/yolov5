# yolov5_big_tif_infer.py
# ============================================================
# YOLOv5 大幅面 GeoTIFF 推理：滑窗切片 + 重叠 +（跨切片）全局NMS
# 输出：GeoJSON + Shapefile（同时输出） + 像素坐标JSON
# 已加入 tqdm 进度条
# ============================================================

import os
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import torch
import rasterio
from rasterio.windows import Window

import cv2

from shapely.geometry import box
import geopandas as gpd
from tqdm import tqdm


# ============================================================
# ✅ 你只需要改这里：默认路径与默认参数
# ============================================================
DEFAULT_YOLOV5_DIR = "/mnt/e/YOLO/yolov5"  # yolov5 本地仓库根目录
DEFAULT_WEIGHTS    = "/mnt/e/YOLO/yolov5/runs/train/exp91/weights/best.pt"
DEFAULT_SOURCE_TIF = "/mnt/f/金三角原始下载影像/简单测试可删/矩形区域/Level17/矩形区域.tif"
DEFAULT_OUTDIR     = "/mnt/f/金三角原始下载影像/简单测试可删/矩形区域/Level17/17级推理结果"

# 默认推理参数（按需改）
DEFAULT_IMGSZ      = 1024
DEFAULT_TILE       = 1024
DEFAULT_OVERLAP    = 0.30
DEFAULT_CONF       = 0.25
DEFAULT_IOU        = 0.45
DEFAULT_DEVICE     = "0"       # "0" / "0,1" / "cpu"
DEFAULT_CLASSES    = ""        # 例如 "0,2,3"；留空表示不过滤
DEFAULT_MAX_DET    = 300
DEFAULT_HALF       = False     # True 用 fp16（显卡支持时更快）
DEFAULT_AGNOSTIC   = False     # True 类无关 NMS
DEFAULT_BANDS      = "1,2,3"   # RGB 波段（1-based）
# ============================================================


def parse_args():
    p = argparse.ArgumentParser("YOLOv5 Large GeoTIFF Inference (tiling + global NMS + GeoJSON+SHP)")

    p.add_argument("--yolov5-dir", type=str, default=DEFAULT_YOLOV5_DIR,
                   help="Path to local yolov5 repo root")
    p.add_argument("--weights", type=str, default=DEFAULT_WEIGHTS,
                   help="Path to trained .pt weights")
    p.add_argument("--source", type=str, default=DEFAULT_SOURCE_TIF,
                   help="Path to input GeoTIFF (.tif)")
    p.add_argument("--outdir", type=str, default=DEFAULT_OUTDIR,
                   help="Output directory")

    p.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ,
                   help="Inference size for each tile (square).")
    p.add_argument("--tile", type=int, default=DEFAULT_TILE,
                   help="Sliding window tile size on original image (pixels).")
    p.add_argument("--overlap", type=float, default=DEFAULT_OVERLAP,
                   help="Tile overlap ratio, e.g. 0.25 means 25% overlap.")
    p.add_argument("--conf-thres", type=float, default=DEFAULT_CONF,
                   help="Confidence threshold")
    p.add_argument("--iou-thres", type=float, default=DEFAULT_IOU,
                   help="IOU threshold for NMS")
    p.add_argument("--device", type=str, default=DEFAULT_DEVICE,
                   help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    p.add_argument("--classes", type=str, default=DEFAULT_CLASSES,
                   help="Filter by class: e.g. '0,2,3' (optional)")
    p.add_argument("--agnostic-nms", action="store_true", default=DEFAULT_AGNOSTIC,
                   help="class-agnostic NMS (if set)")
    p.add_argument("--half", action="store_true", default=DEFAULT_HALF,
                   help="use FP16 if supported (if set)")
    p.add_argument("--max-det", type=int, default=DEFAULT_MAX_DET,
                   help="max detections per tile before merge")

    p.add_argument("--band-index", type=str, default=DEFAULT_BANDS,
                   help="Bands to use for RGB, 1-based index, e.g. '1,2,3' or '4,3,2'")

    # 可选：黑块跳过（加速）
    p.add_argument("--skip-black-mean", type=float, default=1.0,
                   help="Skip tiles if mean intensity < this value. Set <=0 to disable.")
    return p.parse_args()


def xyxy_iou(box1, box2):
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])
    inter_w = max(0.0, xB - xA)
    inter_h = max(0.0, yB - yA)
    inter = inter_w * inter_h
    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union = area1 + area2 - inter + 1e-9
    return inter / union


def global_nms(dets, iou_thres=0.5, class_aware=True):
    """
    dets: list of dict: {xyxy, conf, cls}
    """
    if not dets:
        return dets

    dets = sorted(dets, key=lambda d: float(d["conf"]), reverse=True)
    kept = []

    while dets:
        best = dets.pop(0)
        kept.append(best)

        remain = []
        for d in dets:
            if class_aware and d["cls"] != best["cls"]:
                remain.append(d)
                continue
            if xyxy_iou(best["xyxy"], d["xyxy"]) <= iou_thres:
                remain.append(d)
        dets = remain

    return kept


def ensure_dir(d):
    os.makedirs(d, exist_ok=True)
    return d


def add_yolov5_to_path(yolov5_dir: str):
    yolov5_dir = os.path.abspath(yolov5_dir)
    if yolov5_dir not in sys.path:
        sys.path.insert(0, yolov5_dir)


def load_model(weights, device="", half=False):
    from models.common import DetectMultiBackend
    from utils.torch_utils import select_device

    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=False, data=None, fp16=half)
    stride = int(model.stride)
    names = model.names
    pt = model.pt
    return model, device, stride, names, pt


def letterbox(im, new_shape=640, stride=32, auto=False, scaleFill=False, scaleup=True, color=(114, 114, 114)):
    shape = im.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)

    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))  # (w,h)
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    if auto:
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    elif scaleFill:
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        r = (new_shape[1] / shape[1], new_shape[0] / shape[0])

    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    return im, r, (dw, dh)


def read_tile_as_rgb(src, window: Window, band_ids):
    arr = src.read(band_ids, window=window, boundless=True, fill_value=0)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    # 若 tif 是 uint8 RGB：几乎不改变数据；若是 uint16/float：自动拉伸到 0~255
    rgb = []
    for i in range(arr.shape[0]):
        band = arr[i].astype(np.float32)
        lo, hi = np.percentile(band, (2, 98))
        if hi <= lo:
            band = np.clip(band, 0, 255)
        else:
            band = (band - lo) / (hi - lo) * 255.0
            band = np.clip(band, 0, 255)
        rgb.append(band.astype(np.uint8))

    rgb = np.stack(rgb, axis=2)  # HWC
    return rgb


def run_tile_infer(model, device, img, imgsz, conf_thres, iou_thres, classes, agnostic_nms, max_det):
    from utils.general import non_max_suppression

    img_lb, r, (dw, dh) = letterbox(img, new_shape=imgsz, stride=int(model.stride), auto=False)
    img_lb = img_lb.transpose((2, 0, 1))  # CHW
    img_lb = np.ascontiguousarray(img_lb)

    im = torch.from_numpy(img_lb).to(device)
    im = im.float() / 255.0
    if len(im.shape) == 3:
        im = im.unsqueeze(0)

    pred = model(im, augment=False, visualize=False)

    det = non_max_suppression(
        pred,
        conf_thres=conf_thres,
        iou_thres=iou_thres,
        classes=classes,
        agnostic=agnostic_nms,
        max_det=max_det
    )[0]

    return det, r, dw, dh


def main():
    args = parse_args()

    # 友好检查
    if not os.path.isdir(args.yolov5_dir):
        raise SystemExit(f"[错误] YOLOV5_DIR 不存在：{args.yolov5_dir}")
    if not os.path.isfile(args.weights):
        raise SystemExit(f"[错误] WEIGHTS 不存在：{args.weights}")
    if not os.path.isfile(args.source):
        raise SystemExit(f"[错误] SOURCE tif 不存在：{args.source}")

    ensure_dir(args.outdir)
    add_yolov5_to_path(args.yolov5_dir)

    # classes
    classes = None
    if args.classes.strip():
        classes = [int(x) for x in args.classes.split(",") if x.strip() != ""]

    band_ids = [int(x) for x in args.band_index.split(",")]
    if len(band_ids) != 3:
        raise ValueError("--band-index 必须是 3 个波段，例如 '1,2,3' 或 '4,3,2'")

    model, device, stride, names, pt = load_model(args.weights, device=args.device, half=args.half)

    tile = int(args.tile)
    overlap = float(args.overlap)
    if not (0 <= overlap < 1.0):
        raise ValueError("--overlap must be in [0,1)")
    step = max(1, int(tile * (1.0 - overlap)))

    all_dets = []
    skipped_black = 0

    with rasterio.open(args.source) as src:
        width, height = src.width, src.height
        transform = src.transform
        crs = src.crs

        xs = list(range(0, width, step))
        ys = list(range(0, height, step))
        total_tiles = len(xs) * len(ys)

        print("========== 参数 ==========")
        print("YOLOV5_DIR :", args.yolov5_dir)
        print("WEIGHTS    :", args.weights)
        print("SOURCE     :", args.source)
        print("OUTDIR     :", args.outdir)
        print(f"W,H        : {width},{height}")
        print(f"TILE/OVERL : {tile}/{overlap}  step={step}")
        print(f"IMGSZ      : {args.imgsz}")
        print(f"CONF/IOU   : {args.conf_thres}/{args.iou_thres}")
        print(f"SKIP_BLACK : mean < {args.skip_black_mean} (<=0 关闭)")
        print("==========================")

        pbar = tqdm(total=total_tiles, desc="YOLOv5 tiled inference", ncols=110)

        tile_count = 0
        for y0 in ys:
            for x0 in xs:
                tile_count += 1
                window = Window(x0, y0, tile, tile)
                tile_rgb = read_tile_as_rgb(src, window, band_ids)

                # 黑块跳过（可关）
                if args.skip_black_mean > 0 and tile_rgb.mean() < args.skip_black_mean:
                    skipped_black += 1
                    pbar.update(1)
                    continue

                det, r, dw, dh = run_tile_infer(
                    model=model,
                    device=device,
                    img=tile_rgb,
                    imgsz=args.imgsz,
                    conf_thres=args.conf_thres,
                    iou_thres=args.iou_thres,
                    classes=classes,
                    agnostic_nms=args.agnostic_nms,
                    max_det=args.max_det
                )

                if det is not None and len(det) > 0:
                    det = det.detach().cpu().numpy()

                    for *xyxy, conf, cls in det:
                        x1, y1, x2, y2 = xyxy

                        # letterbox 坐标 -> tile 坐标
                        x1 = (x1 - dw) / r
                        x2 = (x2 - dw) / r
                        y1 = (y1 - dh) / r
                        y2 = (y2 - dh) / r

                        # clip to tile
                        x1 = float(np.clip(x1, 0, tile - 1))
                        x2 = float(np.clip(x2, 0, tile - 1))
                        y1 = float(np.clip(y1, 0, tile - 1))
                        y2 = float(np.clip(y2, 0, tile - 1))

                        # tile -> global pixel
                        gx1 = float(np.clip(x1 + x0, 0, width - 1))
                        gx2 = float(np.clip(x2 + x0, 0, width - 1))
                        gy1 = float(np.clip(y1 + y0, 0, height - 1))
                        gy2 = float(np.clip(y2 + y0, 0, height - 1))

                        # filter tiny boxes
                        if (gx2 - gx1) < 2 or (gy2 - gy1) < 2:
                            continue

                        all_dets.append({"xyxy": [gx1, gy1, gx2, gy2], "conf": float(conf), "cls": int(cls)})

                if tile_count % 10 == 0:
                    pbar.set_postfix({"dets": len(all_dets), "skipped": skipped_black})
                pbar.update(1)

        pbar.close()

    # 全局 NMS（跨 tile）
    class_aware = not args.agnostic_nms
    all_dets_nms = global_nms(all_dets, iou_thres=args.iou_thres, class_aware=class_aware)
    print(f"Raw dets: {len(all_dets)} -> After global NMS: {len(all_dets_nms)}")
    print(f"Skipped black tiles: {skipped_black}")

    # GeoDataFrame
    geoms, recs = [], []

    with rasterio.open(args.source) as src:
        transform = src.transform
        crs = src.crs

        def cname(cid):
            if isinstance(names, dict):
                return names.get(cid, str(cid))
            if isinstance(names, (list, tuple)) and cid < len(names):
                return names[cid]
            return str(cid)

        for d in all_dets_nms:
            x1, y1, x2, y2 = d["xyxy"]

            # pixel -> geo
            X1, Y1 = transform * (x1, y1)
            X2, Y2 = transform * (x2, y2)

            minx, maxx = (min(X1, X2), max(X1, X2))
            miny, maxy = (min(Y1, Y2), max(Y1, Y2))

            geoms.append(box(minx, miny, maxx, maxy))
            recs.append({
                "class_id": d["cls"],
                "class_name": cname(d["cls"]),
                "confidence": d["conf"],
                "px_x1": x1, "px_y1": y1, "px_x2": x2, "px_y2": y2
            })

    gdf = gpd.GeoDataFrame(recs, geometry=geoms, crs=crs)

    # ============================================================
    # ✅ 同时输出 GeoJSON + Shapefile（你要的）
    # ============================================================

    # 1) 输出 GeoJSON（论文/复现）
    out_geojson = os.path.join(args.outdir, f"{Path(args.source).stem}_yolov5.geojson")
    gdf.to_file(out_geojson, driver="GeoJSON", encoding="utf-8")
    print("Saved:", out_geojson)

    # 2) 输出 Shapefile（ArcMap/工程）
    # 注意：这里不要写 .shp 后缀，库会自动生成一套 shp/dbf/shx/prj/cpg
    out_shp = os.path.join(args.outdir, f"{Path(args.source).stem}_yolov5_detections")
    gdf.to_file(out_shp, driver="ESRI Shapefile", encoding="utf-8")
    print("Saved Shapefile:", out_shp)

    # 3) 同时保存像素坐标 json（可选：排查/复现）
    out_json = os.path.join(args.outdir, f"{Path(args.source).stem}_yolov5_px.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(all_dets_nms, f, ensure_ascii=False, indent=2)
    print("Saved:", out_json)


if __name__ == "__main__":
    main()
