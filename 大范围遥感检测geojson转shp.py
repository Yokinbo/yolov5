# geojson_to_shp.py
# =========================================
# 将 GeoJSON (yolov5.geojson) 转为 Shapefile (.shp)
# 依赖: geopandas (conda-forge 安装最稳)
# =========================================

import os
from pathlib import Path

import geopandas as gpd


# ==========================
# ✅ 你只改这里
# ==========================
GEOJSON_PATH = "/mnt/f/金三角原始下载影像/简单测试可删/矩形区域/Level16/推理结果/yolov5.geojson"
OUT_DIR     = "/mnt/f/金三角原始下载影像/简单测试可删/矩形区域/Level16/推理结果/geojson转shp"
OUT_NAME    = "yolov5_detections"   # 输出 shp 文件名（不带后缀）
# ==========================


def main():
    geojson_path = Path(GEOJSON_PATH)
    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not geojson_path.exists():
        raise SystemExit(f"[错误] 找不到输入文件：{geojson_path}")

    out_shp = out_dir / f"{OUT_NAME}.shp"

    print("Input :", geojson_path)
    print("Output:", out_shp)

    # 读取 GeoJSON
    gdf = gpd.read_file(geojson_path)

    if gdf.empty:
        raise SystemExit("[警告] GeoJSON 读取成功，但没有任何要素（empty）。")

    # 可选：修复无效几何（有时会遇到自交等）
    # gdf["geometry"] = gdf["geometry"].buffer(0)

    # 写 Shapefile
    # 注意：Shapefile 字段名会被截断到 10 个字符，这是格式限制（不影响几何）
    gdf.to_file(out_shp, driver="ESRI Shapefile", encoding="utf-8")

    print(f"[完成] 已输出 Shapefile：{out_shp}")
    print("提示：Shapefile 会生成同名的 .shx/.dbf/.prj 等文件，请保持它们在同一目录。")


if __name__ == "__main__":
    main()
