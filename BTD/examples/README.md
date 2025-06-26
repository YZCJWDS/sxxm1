# BTD 示例目录

本目录包含BTD项目的使用示例和演示脚本。

## 📁 文件说明

### 🚀 演示脚本
- [`quick_start_demo.py`](quick_start_demo.py) - 快速开始演示脚本
  - 展示BTD的主要功能
  - 交互式演示各个命令
  - 适合初次使用者了解项目

### ⚙️ 配置示例
- [`example_configs.yaml`](example_configs.yaml) - 配置文件示例集合
  - 模型配置示例 (基础/高精度/快速训练)
  - 数据集配置示例 (COCO/自定义/单类别)
  - 推理配置示例 (高精度/快速/实时)
  - 服务器配置示例 (开发/生产/高性能)

## 🎯 使用方法

### 运行快速演示
```bash
# 进入项目根目录
cd BTD

# 运行演示脚本
python examples/quick_start_demo.py
```

演示脚本将展示：
- 统一管理工具的帮助信息
- 环境检查功能
- 项目信息查看
- 配置文件管理
- 模型管理功能
- 可选的下载和备份功能

### 使用配置示例

#### 1. 查看配置模板
```bash
# 查看所有配置示例
cat examples/example_configs.yaml

# 查看特定配置模板
python main.py config template model_config
```

#### 2. 基于示例创建配置
```bash
# 复制示例配置
cp examples/example_configs.yaml my_configs.yaml

# 编辑配置文件
# 然后提取需要的部分创建单独的配置文件

# 或者直接使用配置管理工具创建
python main.py config create model_config --values '{"model": {"name": "yolov8s"}}'
```

#### 3. 验证配置
```bash
# 验证配置文件
python main.py config validate my_config.yaml
```

## 📋 示例场景

### 场景1: 快速原型开发
```bash
# 使用快速训练配置
python main.py config create model_config --values '{
  "model": {"name": "yolov8n", "input_size": [416, 416]},
  "training": {"epochs": 50, "batch_size": 32}
}'
```

### 场景2: 高精度模型训练
```bash
# 使用高精度配置
python main.py config create model_config --values '{
  "model": {"name": "yolov8l", "input_size": [1280, 1280]},
  "training": {"epochs": 300, "batch_size": 8, "optimizer": "AdamW"}
}'
```

### 场景3: 生产环境部署
```bash
# 使用生产环境服务器配置
python main.py config create server_config --values '{
  "host": "0.0.0.0", "port": 80, "debug": false,
  "workers": 4, "rate_limiting": {"enabled": true}
}'
```

## 🔧 自定义示例

您可以基于现有示例创建自己的配置：

1. **复制基础示例**
   ```bash
   cp examples/example_configs.yaml my_custom_configs.yaml
   ```

2. **修改配置参数**
   - 编辑 `my_custom_configs.yaml`
   - 调整模型参数、训练设置等

3. **验证和使用**
   ```bash
   # 验证配置
   python main.py config validate my_custom_configs.yaml
   
   # 使用配置进行训练
   python main.py train dataset.yaml --config my_custom_configs.yaml
   ```

## 💡 最佳实践

### 配置管理
- 为不同的项目/实验创建独立的配置文件
- 使用有意义的文件名，如 `person_detection_config.yaml`
- 在配置文件中添加注释说明参数用途

### 实验管理
- 使用不同的项目名称区分实验: `--name experiment_v1`
- 保存重要的配置文件到版本控制系统
- 记录实验结果和配置的对应关系

### 性能优化
- 根据硬件配置选择合适的批次大小
- 使用混合精度训练提高速度: `--amp`
- 启用数据缓存减少I/O开销: `--cache`

## 🆘 常见问题

### Q: 如何修改示例配置？
A: 复制示例文件后直接编辑，或使用配置管理工具的更新功能。

### Q: 配置文件格式错误怎么办？
A: 使用 `python main.py config validate config.yaml` 检查配置文件格式。

### Q: 如何恢复默认配置？
A: 重新运行 `python main.py config create <type>` 创建默认配置。

---

**提示**: 这些示例涵盖了BTD的主要使用场景，您可以根据具体需求进行调整和扩展。
