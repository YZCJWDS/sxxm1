# 配置工具 (config_utils.py) 使用说明

## 概述

`config_utils.py` 是一个健壮的配置管理工具，专门为YOLO项目设计，提供参数合并、类型转换、路径标准化等核心功能。

## 核心功能

### 1. 配置文件管理
- **自动加载配置文件**：支持train/val/infer三种模式
- **自动生成默认配置**：如果配置文件不存在，自动创建
- **容错机制**：即使在不完整的项目环境中也能正常工作

### 2. 参数合并系统
- **多源参数合并**：命令行 > YAML > 默认值
- **参数来源追踪**：记录每个参数的来源
- **动态参数支持**：支持extra_args扩展参数

### 3. 类型转换
- **智能类型推断**：字符串自动转换为正确类型
- **特殊参数处理**：如classes参数的列表转换
- **布尔值识别**：支持多种布尔值表达方式

### 4. 路径标准化
- **自动路径解析**：相对路径转绝对路径
- **目录创建**：自动创建必要的目录
- **路径验证**：检查文件和目录的存在性

## 主要函数

### `load_config(config_type='train')`
加载指定类型的配置文件
```python
train_config = load_config('train')
val_config = load_config('val')
infer_config = load_config('infer')
```

### `generate_default_config(config_type)`
生成默认配置文件
```python
generate_default_config('train')  # 生成train.yaml
```

### `merge_config(args, yaml_config=None, mode='train')`
合并多源参数，返回YOLO参数和项目参数
```python
yolo_args, project_args = merge_config(args, yaml_config, 'train')
```

### `_process_params_value(key, value)`
处理参数值的类型转换
```python
epochs = _process_params_value('epochs', '100')  # 返回 int(100)
save = _process_params_value('save', 'true')     # 返回 bool(True)
```

## 使用示例

### 基本使用
```python
from config_utils import load_config, merge_config
import argparse

# 加载配置
config = load_config('train')

# 创建命令行参数
args = argparse.Namespace()
args.epochs = 200
args.batch = 32
args.use_yaml = True

# 合并参数
yolo_args, project_args = merge_config(args, config, 'train')

# 使用参数
print(f"训练轮数: {yolo_args.epochs}")
print(f"参数来源: {project_args.epochs_specified}")
```

### 命令行集成
```python
import argparse
from config_utils import merge_config, load_config

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batch', type=int, default=16)
parser.add_argument('--config', type=str, help='YAML配置文件')
parser.add_argument('--use_yaml', action='store_true')

args = parser.parse_args()

# 加载YAML配置（如果指定）
yaml_config = None
if args.config:
    import yaml
    with open(args.config, 'r') as f:
        yaml_config = yaml.safe_load(f)

# 合并参数
yolo_args, project_args = merge_config(args, yaml_config, 'train')
```

## 配置文件格式

### train.yaml 示例
```yaml
data: 'data.yaml'
epochs: 100
batch: 16
imgsz: 640
device: 'cpu'
lr0: 0.01
momentum: 0.937
weight_decay: 0.0005
save: true
plots: true
```

### val.yaml 示例
```yaml
data: 'data.yaml'
imgsz: 640
batch: 16
conf: 0.25
iou: 0.7
save_txt: true
save_conf: true
```

## 参数优先级

1. **命令行参数** (最高优先级)
2. **YAML配置文件**
3. **默认配置** (最低优先级)

## 类型转换规则

| 输入 | 输出 | 类型 |
|------|------|------|
| "100" | 100 | int |
| "0.01" | 0.01 | float |
| "true" | True | bool |
| "false" | False | bool |
| "0,1,2" | [0,1,2] | list |
| "none" | None | NoneType |

## 错误处理

- **导入失败**：自动使用备用配置
- **文件不存在**：自动生成默认配置
- **类型转换失败**：保持原始值并记录警告
- **路径问题**：提供详细错误信息

## 测试运行

直接运行文件进行功能验证：
```bash
python config_utils.py
```

## 注意事项

1. **目录权限**：确保有创建目录的权限
2. **YAML格式**：配置文件必须是有效的YAML格式
3. **参数名称**：使用YOLO官方支持的参数名称
4. **路径分隔符**：使用正斜杠或反斜杠都可以，会自动处理

## 依赖要求

- Python 3.6+
- PyYAML
- pathlib (Python 3.4+内置)
- argparse (Python标准库)
- logging (Python标准库)

## 扩展性

该工具设计为模块化，可以轻松扩展：
- 添加新的配置类型
- 扩展参数验证规则
- 增加新的类型转换逻辑
- 集成到更大的项目中
