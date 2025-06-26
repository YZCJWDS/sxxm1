# 数据目录说明

## 目录结构
- `raw/`: 原始数据
  - `images/`: 原始图片
  - `original_annotations/`: 原始标注文件
  - `yolo_staged_labels/`: YOLO格式标注文件
- `train/`: 训练数据
- `val/`: 验证数据  
- `test/`: 测试数据

## 使用说明
1. 将原始图片放入 `raw/images/` 目录
2. 将原始标注文件放入 `raw/original_annotations/` 目录
3. 运行数据转换脚本生成YOLO格式标注
4. 运行数据分割脚本将数据分配到train/val/test目录
