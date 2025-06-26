# COCO 转换使用说明

## 概述

本项目提供了完整的 COCO 格式到 YOLO 格式的转换功能，兼容 Ultralytics 的 `convert_coco` 函数接口。

## 功能特点

- ✅ 支持 COCO JSON 格式到 YOLO TXT 格式转换
- ✅ 自动处理类别映射和边界框坐标转换
- ✅ 支持图像文件复制
- ✅ 生成类别信息文件 (classes.txt)
- ✅ 详细的日志记录和错误处理
- ⚠️ 分割标注转换（暂不支持，将在后续版本中添加）

## 使用方法

### 1. 直接调用函数

```python
from yoloserver.utils.data_converter import convert_coco_ultralytics_style

# 基本边界框转换
success = convert_coco_ultralytics_style(
    labels_dir=r"C:\path\to\coco\annotations",
    save_dir="output_directory",
    use_segments=False,  # 边界框模式
    copy_images=True     # 复制图像文件
)

# 分割标注转换（暂不支持）
success = convert_coco_ultralytics_style(
    labels_dir=r"C:\path\to\coco\annotations",
    save_dir="output_directory",
    use_segments=True,   # 分割模式（暂不支持）
    copy_images=True
)
```

### 2. 使用命令行脚本

```bash
# 进入项目目录
cd BTD/yoloserver

# 使用data_processing.py脚本
python scripts/data_processing.py coco \
    --labels-dir "C:\path\to\coco\annotations" \
    --save-dir "output_directory" \
    --copy-images

# 如果需要分割标注（暂不支持）
python scripts/data_processing.py coco \
    --labels-dir "C:\path\to\coco\annotations" \
    --save-dir "output_directory" \
    --use-segments \
    --copy-images
```

### 3. 运行示例脚本

```bash
# 运行示例脚本
python scripts/coco_convert_example.py
```

## 输入要求

### 目录结构
```
输入目录/
├── annotations.json    # COCO格式标注文件
├── instances.json      # 或其他COCO JSON文件
└── images/            # 图像文件（可选，用于复制）
    ├── image1.jpg
    ├── image2.jpg
    └── ...
```

### COCO JSON 格式要求
- 标准的COCO格式JSON文件
- 包含 `images`, `annotations`, `categories` 字段
- 边界框格式：`[x, y, width, height]`

## 输出结果

### 目录结构
```
输出目录/
├── labels/            # YOLO格式标注文件
│   ├── image1.txt
│   ├── image2.txt
│   └── ...
├── images/            # 复制的图像文件（如果启用）
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── classes.txt        # 类别名称文件
```

### YOLO 格式说明
每个 `.txt` 文件包含该图像的所有标注，格式为：
```
class_id center_x center_y width height
```
- `class_id`: 类别ID（从0开始）
- `center_x, center_y`: 边界框中心点坐标（归一化到0-1）
- `width, height`: 边界框宽高（归一化到0-1）

## 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `labels_dir` | str | 必需 | COCO JSON文件所在目录 |
| `save_dir` | str | 必需 | 输出目录 |
| `use_segments` | bool | False | 是否转换分割标注（暂不支持） |
| `copy_images` | bool | True | 是否复制图像文件 |

## 对应关系

您提到的两段代码对应关系：

### 第一段代码（边界框转换）
```python
from ultralytics.data.converter import convert_coco
convert_coco(
    labels_dir=r"C:\Users\Matri\Desktop\DTH2\yoloserver\data\raw\original_annotations",
    save_dir="dirs"
)
```

**对应的本项目实现：**
```python
from yoloserver.utils.data_converter import convert_coco_ultralytics_style
convert_coco_ultralytics_style(
    labels_dir=r"C:\Users\Matri\Desktop\DTH2\yoloserver\data\raw\original_annotations",
    save_dir="dirs",
    use_segments=False,  # 边界框模式
    copy_images=True
)
```

### 第二段代码（分割转换）
```python
from ultralytics.data.converter import convert_coco
convert_coco(
    labels_dir=r"C:\Users\Matri\Desktop\DTH2\yoloserver\data\raw\original_annotations",
    save_dir="dirs",
    use_segments=True,
)
```

**对应的本项目实现：**
```python
from yoloserver.utils.data_converter import convert_coco_ultralytics_style
convert_coco_ultralytics_style(
    labels_dir=r"C:\Users\Matri\Desktop\DTH2\yoloserver\data\raw\original_annotations",
    save_dir="dirs",
    use_segments=True,   # 分割模式（暂不支持）
    copy_images=True
)
```

## 注意事项

1. **分割标注支持**：目前暂不支持分割标注转换，设置 `use_segments=True` 会显示警告并转换为边界框格式
2. **路径格式**：支持绝对路径和相对路径
3. **图像文件**：如果启用 `copy_images`，会自动查找并复制对应的图像文件
4. **日志记录**：所有操作都有详细的日志记录，便于调试和监控

## 故障排除

### 常见问题

1. **找不到JSON文件**
   - 检查 `labels_dir` 路径是否正确
   - 确保目录中包含 `.json` 文件

2. **转换失败**
   - 检查JSON文件格式是否符合COCO标准
   - 查看日志输出获取详细错误信息

3. **图像文件未复制**
   - 检查图像文件是否在正确的位置
   - 确保 `copy_images=True`

### 获取帮助

如果遇到问题，请：
1. 查看日志输出获取详细错误信息
2. 检查输入文件格式是否正确
3. 确认所有路径都存在且可访问
