# BTD项目 - 目标检测系统

BTD (Behavior Target Detection) 是一个基于YOLO的完整目标检测解决方案，包含模型训练、推理、Web界面和桌面应用。

## 项目结构

```
BTD/
├── yoloserver/              # YOLO服务端核心模块
│   ├── configs/             # 配置文件
│   │   ├── model_config.yaml    # 模型配置
│   │   └── dataset_config.yaml  # 数据集配置
│   ├── data/                # 数据目录
│   │   ├── raw/             # 原始数据
│   │   │   ├── images/          # 原始图片
│   │   │   ├── original_annotations/  # 原始标注文件
│   │   │   └── yolo_staged_labels/    # YOLO格式标注
│   │   ├── train/           # 训练数据
│   │   │   ├── images/
│   │   │   └── labels/
│   │   ├── val/             # 验证数据
│   │   │   ├── images/
│   │   │   └── labels/
│   │   └── test/            # 测试数据
│   │       ├── images/
│   │       └── labels/
│   ├── models/              # 模型目录
│   │   ├── checkpoints/     # 训练检查点
│   │   └── pretrained/      # 预训练模型
│   ├── runs/                # 运行结果
│   │   ├── train/           # 训练结果
│   │   ├── val/             # 验证结果
│   │   └── detect/          # 检测结果
│   ├── scripts/             # 自动化脚本
│   │   ├── enhanced_train.py    # 增强训练脚本
│   │   ├── model_manager.py     # 模型管理脚本
│   │   ├── dataset_analyzer.py  # 数据集分析脚本
│   │   ├── project_manager.py   # 项目管理脚本
│   │   ├── config_manager.py    # 配置管理脚本
│   │   ├── train.py         # 基础训练脚本
│   │   ├── validate.py      # 验证脚本
│   │   ├── inference.py     # 推理脚本
│   │   └── data_processing.py  # 数据处理脚本
│   └── utils/               # 工具模块
│       ├── path_utils.py    # 路径管理
│       ├── logger.py        # 日志工具
│       ├── data_converter.py # 数据转换
│       └── file_utils.py    # 文件操作
├── BTDWeb/                  # Web前端
│   ├── src/
│   │   ├── components/
│   │   ├── pages/
│   │   ├── assets/
│   │   └── utils/
│   └── public/
├── BTDUi/                   # 桌面应用
│   ├── src/
│   ├── assets/
│   └── config/
├── main.py                 # 主启动脚本 (统一入口)
├── initialize_project.py   # 项目初始化脚本
└── README.md               # 项目说明
```

## 快速开始

### 1. 环境准备

```bash
# 克隆项目
git clone <repository-url>
cd BTD

# 创建虚拟环境 (推荐)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate     # Windows
```

#### 方法一：自动安装依赖 (推荐)

```bash
# 使用自动安装脚本
python install_dependencies.py
```

#### 方法二：手动安装依赖

```bash
# 安装所有依赖
pip install -r requirements.txt

# 或者安装核心依赖
pip install ultralytics opencv-python pyyaml pillow numpy matplotlib seaborn tqdm pandas torch torchvision

# 可选依赖 (用于系统监控)
pip install psutil requests
```

#### 方法三：最小安装

```bash
# 仅安装核心功能所需的依赖
pip install ultralytics opencv-python pyyaml pillow
```

### 2. 项目初始化

```bash
# 使用新的统一管理工具初始化
python main.py init

# 或使用原始初始化脚本
python initialize_project.py
```

## 🚀 新功能：统一管理工具

BTD现在提供了统一的命令行管理工具 `main.py`，让项目管理更加简单高效！

### 主要功能

```bash
# 查看所有可用命令
python main.py --help

# 项目管理
python main.py check          # 检查环境
python main.py install        # 安装依赖
python main.py backup         # 备份项目
python main.py info           # 项目信息

# 配置管理
python main.py config create model_config     # 创建模型配置
python main.py config validate config.yaml   # 验证配置
python main.py config list                   # 列出配置

# 数据管理
python main.py data analyze dataset.yaml     # 分析数据集
python main.py data convert --from coco      # 转换数据格式

# 模型管理
python main.py model list                    # 列出可用模型
python main.py model download yolov8n        # 下载模型
python main.py model export model.pt         # 导出模型

# 训练和推理
python main.py train dataset.yaml            # 训练模型
python main.py infer model.pt image.jpg      # 推理测试

# 服务器
python main.py server start                  # 启动Web服务
```

### 3. 数据准备

#### 方法一：使用数据处理脚本

```bash
# 处理COCO格式数据
python yoloserver/scripts/data_processing.py process --input /path/to/coco/data --format coco

# 处理Pascal VOC格式数据
python yoloserver/scripts/data_processing.py process --input /path/to/voc/data --format pascal --classes person car bicycle

# 分割数据集
python yoloserver/scripts/data_processing.py split --images /path/to/images --labels /path/to/labels

# 验证数据集
python yoloserver/scripts/data_processing.py validate --labels yoloserver/data/train/labels --images yoloserver/data/train/images
```

#### 方法二：手动准备

1. 将原始图片放入 `yoloserver/data/raw/images/`
2. 将标注文件放入 `yoloserver/data/raw/original_annotations/`
3. 运行数据转换和分割脚本

### 4. 模型训练

#### 使用统一管理工具 (推荐)

```bash
# 基础训练
python main.py train dataset.yaml

# 自定义参数训练
python main.py train dataset.yaml \
    --model yolov8s \
    --epochs 200 \
    --batch-size 32 \
    --img-size 640 \
    --device 0
```

#### 使用增强训练脚本

```bash
# 功能更丰富的训练脚本
python yoloserver/scripts/enhanced_train.py \
    --data dataset.yaml \
    --model yolov8n \
    --epochs 100 \
    --batch-size 16 \
    --amp \
    --cache
```

#### 使用原始训练脚本

```bash
# 基础训练脚本
python yoloserver/scripts/train.py --model yolov8n --epochs 100 --batch-size 16
```

### 5. 模型验证

```bash
# 验证模型
python yoloserver/scripts/validate.py --model runs/train/exp/weights/best.pt

# 比较多个模型
python yoloserver/scripts/validate.py --compare model1.pt model2.pt model3.pt
```

### 6. 模型推理

```bash
# 图像推理
python yoloserver/scripts/inference.py --model best.pt --source image.jpg

# 视频推理
python yoloserver/scripts/inference.py --model best.pt --source video.mp4

# 批量推理
python yoloserver/scripts/inference.py --model best.pt --source /path/to/images --batch

# 实时推理
python yoloserver/scripts/inference.py --model best.pt --webcam --source 0
```

## 配置说明

### 模型配置 (model_config.yaml)

主要配置项：
- `model.name`: 模型名称 (yolov8n/s/m/l/x)
- `model.nc`: 类别数量
- `model.names`: 类别名称列表
- `train.epochs`: 训练轮数
- `train.batch_size`: 批次大小
- `train.lr0`: 初始学习率

### 数据集配置 (dataset_config.yaml)

主要配置项：
- `path`: 数据集根目录
- `train/val/test`: 训练/验证/测试数据路径
- `nc`: 类别数量
- `names`: 类别名称映射

## 🛠️ 增强功能脚本

### 1. 增强训练脚本 (enhanced_train.py)

功能特性：
- ✅ 支持所有YOLO模型 (v8n/s/m/l/x)
- ✅ 自动数据增强配置
- ✅ 混合精度训练 (AMP)
- ✅ 学习率调度
- ✅ 早停机制
- ✅ 训练进度监控
- ✅ 自动保存最佳模型

```bash
# 完整参数示例
python yoloserver/scripts/enhanced_train.py \
    --data dataset.yaml \
    --model yolov8s \
    --epochs 200 \
    --batch-size 32 \
    --img-size 640 \
    --lr0 0.01 \
    --momentum 0.937 \
    --weight-decay 0.0005 \
    --amp \
    --cache \
    --device 0
```

### 2. 模型管理脚本 (model_manager.py)

功能特性：
- ✅ 预训练模型下载
- ✅ 本地模型管理
- ✅ 模型格式转换 (ONNX, TensorRT)
- ✅ 模型优化 (量化, 剪枝)
- ✅ 性能基准测试

```bash
# 列出可用模型
python yoloserver/scripts/model_manager.py list

# 下载预训练模型
python yoloserver/scripts/model_manager.py download yolov8n

# 导出ONNX格式
python yoloserver/scripts/model_manager.py export model.pt --format onnx

# 模型基准测试
python yoloserver/scripts/model_manager.py benchmark model.pt --data dataset.yaml
```

### 3. 数据集分析脚本 (dataset_analyzer.py)

功能特性：
- ✅ 数据集统计分析
- ✅ 类别分布可视化
- ✅ 图像尺寸分析
- ✅ 边界框统计
- ✅ 数据质量检查
- ✅ 生成分析报告

```bash
# 分析数据集
python yoloserver/scripts/dataset_analyzer.py --config dataset.yaml --output analysis/

# 分析特定分割
python yoloserver/scripts/dataset_analyzer.py --config dataset.yaml --split train
```

### 4. 项目管理脚本 (project_manager.py)

功能特性：
- ✅ 环境检查
- ✅ 依赖安装
- ✅ 项目备份/恢复
- ✅ 系统信息收集
- ✅ 项目统计

```bash
# 检查环境
python yoloserver/scripts/project_manager.py check

# 安装依赖
python yoloserver/scripts/project_manager.py install

# 备份项目
python yoloserver/scripts/project_manager.py backup --output backups/

# 项目信息
python yoloserver/scripts/project_manager.py info --export project_info.json
```

### 5. 配置管理脚本 (config_manager.py)

功能特性：
- ✅ 配置文件模板
- ✅ 配置验证
- ✅ 配置更新
- ✅ 批量配置导出

```bash
# 创建配置文件
python yoloserver/scripts/config_manager.py create model_config

# 验证配置
python yoloserver/scripts/config_manager.py validate config.yaml

# 查看模板
python yoloserver/scripts/config_manager.py template dataset_config

# 更新配置
python yoloserver/scripts/config_manager.py update config.yaml '{"epochs": 200}'
```

## 工具模块

### 路径管理 (path_utils.py)
- `get_project_root()`: 获取项目根目录
- `get_data_paths()`: 获取数据相关路径
- `get_model_paths()`: 获取模型相关路径
- `ensure_dir()`: 确保目录存在

### 日志工具 (logger.py)
- `setup_logger()`: 设置日志记录器
- `TimerLogger`: 计时日志上下文管理器
- `ProgressLogger`: 进度日志记录器

### 数据转换 (data_converter.py)
- `convert_coco_to_yolo()`: COCO到YOLO格式转换
- `convert_pascal_to_yolo()`: Pascal VOC到YOLO格式转换
- `split_dataset()`: 数据集分割
- `validate_annotations()`: 标注验证

### 文件操作 (file_utils.py)
- `read_yaml()/write_yaml()`: YAML文件读写
- `read_json()/write_json()`: JSON文件读写
- `copy_files()/move_files()`: 文件复制移动

## 📋 使用示例

### 完整工作流程

```bash
# 1. 初始化项目
python main.py init

# 2. 检查环境
python main.py check

# 3. 创建数据集配置
python main.py config create dataset_config --output data/dataset.yaml

# 4. 分析数据集
python main.py data analyze data/dataset.yaml --output analysis/

# 5. 下载预训练模型
python main.py model download yolov8s

# 6. 开始训练
python main.py train data/dataset.yaml --model yolov8s --epochs 100

# 7. 验证模型
python main.py validate runs/train/exp/weights/best.pt

# 8. 导出模型
python main.py model export runs/train/exp/weights/best.pt --format onnx

# 9. 推理测试
python main.py infer runs/train/exp/weights/best.pt test_image.jpg

# 10. 启动Web服务
python main.py server start
```

## ❓ 常见问题

### Q: 如何使用新的统一管理工具？
A: 使用 `python main.py --help` 查看所有可用命令，每个命令都有详细的帮助信息。

### Q: 如何添加新的类别？
A: 使用配置管理工具：`python main.py config create dataset_config`，然后修改生成的配置文件。

### Q: 训练过程中出现内存不足？
A: 使用 `python main.py train dataset.yaml --batch-size 8` 减小批次大小，或使用更小的模型。

### Q: 如何恢复中断的训练？
A: 使用 `python main.py train dataset.yaml --resume runs/train/exp/weights/last.pt`。

### Q: 推理速度慢怎么办？
A: 导出为ONNX格式：`python main.py model export model.pt --format onnx`，然后使用ONNX模型推理。

### Q: 如何分析数据集质量？
A: 使用 `python main.py data analyze dataset.yaml` 生成详细的数据集分析报告。

### Q: 如何备份项目？
A: 使用 `python main.py backup --output backups/` 创建项目备份。

### Q: 如何比较不同模型的性能？
A: 使用 `python main.py model benchmark model1.pt --data dataset.yaml` 进行基准测试。

## 🎯 功能特性

### ✅ 已完成功能

- ✅ 统一命令行管理工具 (`main.py`)
- ✅ 增强训练脚本 (支持AMP、缓存、学习率调度)
- ✅ 智能模型管理 (下载、转换、优化、基准测试)
- ✅ 数据集分析和可视化
- ✅ 项目管理和备份
- ✅ 配置文件管理和验证
- ✅ 完整的日志系统
- ✅ 进度监控和计时
- ✅ 多种数据格式支持
- ✅ 模型性能基准测试

### 🚧 开发计划

- [ ] Web界面完善 (BTDWeb)
- [ ] 桌面应用开发 (BTDUi)
- [ ] 分布式训练支持
- [ ] 模型量化和剪枝优化
- [ ] TensorRT部署支持
- [ ] 实时视频流处理
- [ ] 模型版本管理
- [ ] 自动超参数调优
- [ ] 云端训练支持
- [ ] 移动端部署工具

## 贡献指南

1. Fork 项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开 Pull Request

## 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 联系方式

- 项目维护者: BTD Team
- 邮箱: btd-team@example.com
- 项目链接: [https://github.com/btd-team/BTD](https://github.com/btd-team/BTD)
