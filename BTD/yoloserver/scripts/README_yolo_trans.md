# YOLO数据转换工具使用说明

## 工具简介

`yolo_trans.py` 是一个一站式YOLO数据转换工具，可以帮您：

1. **转换标注格式** - 将COCO、Pascal VOC等格式转换为YOLO格式
2. **分割数据集** - 自动按比例分割为训练集/验证集/测试集
3. **生成配置文件** - 自动生成训练需要的`data.yaml`文件
4. **验证数据** - 检查转换结果是否正确

## 快速开始

### 1. 最简单的使用方式（自动检测格式）

```bash
# 假设您的标注文件在 data/raw/annotations 目录下
python BTD/yoloserver/scripts/yolo_trans.py --input data/raw/annotations
```

这个命令会：
- 自动检测标注格式（COCO/Pascal VOC/YOLO）
- 转换为YOLO格式
- 按7:2:1比例分割数据集
- 生成data.yaml配置文件

### 2. 指定标注格式

```bash
# COCO格式
python BTD/yoloserver/scripts/yolo_trans.py --input data/raw/annotations --format coco

# Pascal VOC格式（需要指定类别）
python BTD/yoloserver/scripts/yolo_trans.py --input data/raw/annotations --format pascal --classes cat dog bird

# 已经是YOLO格式
python BTD/yoloserver/scripts/yolo_trans.py --input data/raw/annotations --format yolo --classes cat dog bird
```

### 3. 自定义分割比例

```bash
# 训练集80%，验证集15%，测试集5%
python BTD/yoloserver/scripts/yolo_trans.py --input data/raw/annotations --train-ratio 0.8 --val-ratio 0.15 --test-ratio 0.05
```

## 常用参数说明

| 参数 | 说明 | 默认值 | 示例 |
|------|------|--------|------|
| `--input` | 输入标注文件目录 | 必需 | `data/raw/annotations` |
| `--format` | 标注格式 | `auto` | `coco`, `pascal`, `yolo` |
| `--classes` | 类别名称列表 | 自动检测 | `cat dog bird` |
| `--train-ratio` | 训练集比例 | `0.7` | `0.8` |
| `--val-ratio` | 验证集比例 | `0.2` | `0.15` |
| `--test-ratio` | 测试集比例 | `0.1` | `0.05` |
| `--no-clean` | 不清理之前的数据 | 默认清理 | - |
| `--no-validate` | 跳过数据验证 | 默认验证 | - |

## 实际使用示例

### 示例1：转换COCO数据集

```bash
# 假设您有COCO格式的标注文件
python BTD/yoloserver/scripts/yolo_trans.py \
    --input data/raw/coco_annotations \
    --format coco
```

### 示例2：转换Pascal VOC数据集

```bash
# Pascal VOC格式需要指定类别名称
python BTD/yoloserver/scripts/yolo_trans.py \
    --input data/raw/voc_annotations \
    --format pascal \
    --classes person car bicycle motorcycle airplane bus train truck
```

### 示例3：处理医学影像数据集

```bash
# 假设是脑肿瘤检测数据集
python BTD/yoloserver/scripts/yolo_trans.py \
    --input data/raw/brain_tumor_annotations \
    --format pascal \
    --classes glioma_tumor meningioma_tumor pituitary_tumor \
    --train-ratio 0.8 \
    --val-ratio 0.15 \
    --test-ratio 0.05
```

## 输出结果

运行成功后，您会得到：

```
BTD/yoloserver/data/
├── train/
│   ├── images/     # 训练图像
│   └── labels/     # 训练标签
├── val/
│   ├── images/     # 验证图像
│   └── labels/     # 验证标签
├── test/
│   ├── images/     # 测试图像
│   └── labels/     # 测试标签
└── data.yaml       # 训练配置文件
```

## 生成的data.yaml示例

```yaml
path: /path/to/BTD/yoloserver/data
train: train/images
val: val/images
test: test/images
nc: 3
names: [glioma_tumor, meningioma_tumor, pituitary_tumor]
```

## 下一步：开始训练

转换完成后，您可以：

1. **检查转换结果**
   ```bash
   # 查看生成的文件
   ls BTD/yoloserver/data/train/images/
   ls BTD/yoloserver/data/train/labels/
   ```

2. **开始训练模型**
   ```bash
   # 使用生成的配置文件训练
   python BTD/main.py train
   ```

3. **或使用YOLO11直接训练**
   ```bash
   # 如果您安装了ultralytics
   yolo train data=BTD/yoloserver/configs/data.yaml model=yolo11n.pt epochs=100
   ```

## 常见问题

### Q: 如何知道我的数据是什么格式？
A: 使用 `--format auto` 让工具自动检测，或查看文件类型：
- COCO: `.json` 文件
- Pascal VOC: `.xml` 文件  
- YOLO: `.txt` 文件

### Q: 转换失败怎么办？
A: 检查：
1. 输入目录是否存在
2. 标注文件格式是否正确
3. Pascal VOC格式是否提供了类别名称

### Q: 如何验证转换结果？
A: 工具会自动验证，您也可以手动检查：
1. 图像和标签文件数量是否匹配
2. 标签文件格式是否正确（每行：class_id x_center y_center width height）
3. 类别ID是否在合理范围内

## 技术支持

如果遇到问题，请检查：
1. Python环境是否正确
2. 依赖包是否安装完整
3. 输入数据格式是否符合要求

更多详细信息请查看日志文件：`BTD/logs/yolo_trans.log`
