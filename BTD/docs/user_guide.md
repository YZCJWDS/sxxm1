# BTD用户使用指南

## 📖 目录

1. [快速开始](#快速开始)
2. [项目管理](#项目管理)
3. [配置管理](#配置管理)
4. [数据管理](#数据管理)
5. [模型管理](#模型管理)
6. [训练流程](#训练流程)
7. [推理部署](#推理部署)
8. [故障排除](#故障排除)

## 🚀 快速开始

### 第一次使用

```bash
# 1. 克隆项目
git clone <repository_url>
cd BTD

# 2. 初始化项目
python main.py init

# 3. 运行演示
python examples/quick_start_demo.py
```

### 基本工作流程

```bash
# 检查环境 → 准备数据 → 训练模型 → 推理测试
python main.py check
python main.py config create dataset_config
python main.py train dataset.yaml
python main.py infer model.pt image.jpg
```

## 🛠️ 项目管理

### 环境检查

```bash
# 检查Python版本、依赖包、GPU支持等
python main.py check

# 查看详细系统信息
python main.py info --export system_info.json
```

### 依赖管理

```bash
# 安装所有依赖
python main.py install

# 从requirements文件安装
python main.py install --requirements requirements.txt

# 生成requirements文件
python main.py requirements
```

### 项目备份

```bash
# 完整备份
python main.py backup --output backups/

# 备份包含数据但不包含模型
python main.py backup --include-data --no-models

# 恢复备份
python main.py restore backup_file.tar.gz --output restored/
```

## ⚙️ 配置管理

### 创建配置文件

```bash
# 创建模型配置
python main.py config create model_config

# 创建数据集配置
python main.py config create dataset_config --output data/my_dataset.yaml

# 使用自定义值创建配置
python main.py config create model_config --values '{"model": {"name": "yolov8m"}}'
```

### 配置验证和管理

```bash
# 验证配置文件
python main.py config validate config.yaml

# 列出所有配置
python main.py config list

# 查看配置模板
python main.py config template model_config

# 更新配置
python main.py config update config.yaml '{"epochs": 200}'
```

## 📊 数据管理

### 数据集分析

```bash
# 分析完整数据集
python main.py data analyze dataset.yaml --output analysis/

# 分析特定分割
python main.py data analyze dataset.yaml --split train

# 生成可视化报告
python main.py data analyze dataset.yaml --output reports/
```

### 数据格式转换

```bash
# COCO到YOLO格式
python main.py data convert --from coco --to yolo --input coco_data/ --output yolo_data/

# Pascal VOC到YOLO格式
python main.py data convert --from pascal --to yolo --input voc_data/ --output yolo_data/

# 数据集分割
python main.py data split --input images/ --train 0.7 --val 0.2 --test 0.1
```

## 🤖 模型管理

### 预训练模型

```bash
# 列出可用模型
python main.py model list

# 下载预训练模型
python main.py model download yolov8n
python main.py model download yolov8s yolov8m

# 查看本地模型
python main.py model local
```

### 模型转换和优化

```bash
# 导出ONNX格式
python main.py model export model.pt --format onnx

# 导出TensorRT格式
python main.py model export model.pt --format engine --workspace 4

# 模型量化
python main.py model optimize model.pt --type quantize

# 性能基准测试
python main.py model benchmark model.pt --data dataset.yaml
```

## 🏋️ 训练流程

### 基础训练

```bash
# 使用默认参数训练
python main.py train dataset.yaml

# 指定模型和基本参数
python main.py train dataset.yaml --model yolov8s --epochs 100 --batch-size 16
```

### 高级训练选项

```bash
# 完整参数训练
python main.py train dataset.yaml \
    --model yolov8m \
    --epochs 200 \
    --batch-size 32 \
    --img-size 640 \
    --lr0 0.01 \
    --momentum 0.937 \
    --weight-decay 0.0005 \
    --amp \
    --cache \
    --device 0

# 恢复训练
python main.py train dataset.yaml --resume runs/train/exp/weights/last.pt

# 多GPU训练
python main.py train dataset.yaml --device 0,1,2,3
```

### 训练监控

```bash
# 查看训练日志
tail -f logs/enhanced_train.log

# 验证训练结果
python main.py validate runs/train/exp/weights/best.pt

# 查看训练曲线
# 结果保存在 runs/train/exp/ 目录下
```

## 🔍 推理部署

### 单张图像推理

```bash
# 基础推理
python main.py infer model.pt image.jpg

# 指定输出目录
python main.py infer model.pt image.jpg --output results/

# 调整置信度阈值
python main.py infer model.pt image.jpg --conf 0.5 --iou 0.4
```

### 批量推理

```bash
# 批量处理图像
python main.py infer model.pt images/ --output results/

# 视频推理
python main.py infer model.pt video.mp4 --output results/

# 实时推理 (摄像头)
python main.py infer model.pt 0  # 使用摄像头0
```

### Web服务部署

```bash
# 启动Web服务器
python main.py server start

# 检查服务状态
python main.py server status

# 停止服务
python main.py server stop
```

## 🔧 故障排除

### 常见错误及解决方案

#### 1. 环境问题

**错误**: `ModuleNotFoundError: No module named 'ultralytics'`
```bash
# 解决方案
python main.py install
```

**错误**: `CUDA out of memory`
```bash
# 解决方案：减小批次大小
python main.py train dataset.yaml --batch-size 8
```

#### 2. 数据问题

**错误**: `Dataset not found`
```bash
# 解决方案：检查数据集配置
python main.py config validate dataset.yaml
python main.py data analyze dataset.yaml
```

**错误**: `No images found`
```bash
# 解决方案：检查图像路径和格式
ls data/train/images/
python main.py data analyze dataset.yaml --split train
```

#### 3. 训练问题

**错误**: `Training stopped early`
```bash
# 解决方案：调整早停参数或增加训练轮数
python main.py train dataset.yaml --epochs 300 --patience 100
```

**错误**: `Loss becomes NaN`
```bash
# 解决方案：降低学习率
python main.py train dataset.yaml --lr0 0.001
```

#### 4. 推理问题

**错误**: `Model loading failed`
```bash
# 解决方案：检查模型文件
python main.py model local
python main.py validate model.pt
```

### 性能优化建议

#### 训练优化
- 使用混合精度训练: `--amp`
- 启用数据缓存: `--cache`
- 调整批次大小以充分利用GPU内存
- 使用多GPU训练: `--device 0,1,2,3`

#### 推理优化
- 导出为ONNX格式提高推理速度
- 使用半精度推理: `--half`
- 批量推理提高吞吐量
- 使用TensorRT进行GPU加速

### 日志和调试

```bash
# 查看详细日志
python main.py --verbose command

# 静默模式
python main.py --quiet command

# 查看特定日志文件
tail -f logs/enhanced_train.log
tail -f logs/server.log
tail -f logs/project_manager.log
```

## 📚 进阶使用

### 自定义配置

1. 复制示例配置: `cp examples/example_configs.yaml my_config.yaml`
2. 编辑配置文件
3. 验证配置: `python main.py config validate my_config.yaml`
4. 使用配置: `python main.py train my_dataset.yaml --config my_config.yaml`

### 脚本集成

```python
# 在Python脚本中使用BTD功能
import subprocess

# 训练模型
result = subprocess.run([
    "python", "main.py", "train", "dataset.yaml",
    "--model", "yolov8s", "--epochs", "100"
])

# 推理
result = subprocess.run([
    "python", "main.py", "infer", "model.pt", "image.jpg"
])
```

### 自动化工作流

```bash
#!/bin/bash
# 自动化训练脚本

# 1. 检查环境
python main.py check || exit 1

# 2. 分析数据
python main.py data analyze dataset.yaml

# 3. 训练模型
python main.py train dataset.yaml --model yolov8s --epochs 100

# 4. 验证模型
python main.py validate runs/train/exp/weights/best.pt

# 5. 导出模型
python main.py model export runs/train/exp/weights/best.pt --format onnx

echo "训练流程完成！"
```

## 🆘 获取帮助

- 查看命令帮助: `python main.py command --help`
- 查看项目文档: `docs/`
- 运行演示: `python examples/quick_start_demo.py`
- 查看配置示例: `examples/example_configs.yaml`

---

**提示**: 本指南涵盖了BTD的主要功能，更多详细信息请参考各个脚本的帮助文档和源代码注释。
