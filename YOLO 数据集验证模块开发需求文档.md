## YOLO 数据集验证模块开发需求文档

本项目旨在开发一个核心模块，用于**验证 YOLO 数据集的结构、内容和划分的正确性**。在 YOLO 模型训练之前，确保数据集的高质量是至关重要的。此功能将帮助用户在数据准备阶段就发现并修复潜在问题，避免在模型训练后期出现错误或导致模型性能不佳。

好的，明白了！这次我们专注于 **YOLO 数据集验证功能**。为您准备一份针对这一个功能的简单需求文档。

------

## YOLO 数据集验证模块开发需求文档

### 功能概述

本项目旨在开发一个核心模块，用于**验证 YOLO 数据集的结构、内容和划分的正确性**。在 YOLO 模型训练之前，确保数据集的高质量是至关重要的。此功能将帮助用户在数据准备阶段就发现并修复潜在问题，避免在模型训练后期出现错误或导致模型性能不佳。

### 核心功能需求

你需要开发或完善以下两个主要部分：

#### `utils/dataset_validation.py`：数据验证逻辑模块

这个文件将包含所有具体的、可复用的数据验证逻辑函数。它不直接与命令行交互，只提供验证功能。

- **`verify_dataset_config(yaml_path: Path, current_logger: logging.Logger, mode: str, task_type: str) -> Tuple[bool, List[Dict]]`**
  - 职责：
    - 读取并解析 `data.yaml` 文件，确保其格式正确。
    - 验证 `data.yaml` 中定义的类别数量（`nc`）与类别名称列表（`names`）是否一致。
    - 遍历 `data.yaml` 中指定的每个数据集分割（`train`, `val`, `test`）的图像路径。
    - 检查图像目录是否存在且包含图像文件。
    - 对于抽样或全部图像，检查每张图像是否都有对应的 YOLO 格式 (`.txt`) 标签文件。
    - 验证标签文件的内容：
      - 每行数据是否符合指定 `task_type`（`detection` 或 `segmentation`）的 YOLO 格式（例如，检测任务每行5个值，分割任务每行至少7个值且坐标对数量正确）。
      - 所有数值是否为有效数字。
      - 类别 ID 是否在 `[0, nc-1]` 范围内。
      - 所有坐标值是否在 `[0, 1]` 范围内。
  - 参数：
    - `yaml_path` (Path): `data.yaml` 文件的路径。
    - `current_logger` (logging.Logger): 用于记录日志的 logger 实例。
    - `mode` (str): 验证模式，"FULL" (完整验证) 或 "SAMPLE" (抽样验证)。
    - `task_type` (str): 任务类型，"detection" 或 "segmentation"。
  - 返回值：
    - `bool`: 表示验证是否通过（`True` 为通过，`False` 为未通过）。
    - `List[Dict]`: 包含所有不合法样本信息的列表，每个字典包含 `'image_path'`, `'label_path'`, `'error_message'`。
- **`verify_split_uniqueness(yaml_path: Path, current_logger: logging.Logger) -> bool`**
  - 职责：
    - 读取 `data.yaml` 文件。
    - 检查 `train`, `val`, `test` 三个数据集分割之间是否存在**重复的图像文件**（通过文件名判断）。数据集中不同分割之间不应有重叠。
  - 参数：
    - `yaml_path` (Path): `data.yaml` 文件的路径。
    - `current_logger` (logging.Logger): 用于记录日志的 logger 实例。
  - 返回值：
    - `bool`: 表示分割唯一性验证是否通过（`True` 为无重复，`False` 为存在重复）。
- **`delete_invalid_files(invalid_data_list: list, current_logger: logging.Logger)`**
  - 职责：
    - 根据传入的不合法文件列表，删除对应的图像文件和标签文件。
    - **注意：** 删除操作不可逆，应谨慎执行。
  - 参数：
    - `invalid_data_list` (list): `verify_dataset_config` 返回的不合法样本列表。
    - `current_logger` (logging.Logger): 用于记录日志的 logger 实例。

#### `scripts/yolo_validate.py`：数据集验证入口脚本

这个文件将是用户直接运行的脚本，它负责命令行解析和调用 `utils/dataset_validation.py` 中的验证逻辑。

- 职责：
  - 命令行参数解析：
    - 接收 `--mode` (验证模式: FULL/SAMPLE)。
    - 接收 `--task` (任务类型: detection/segmentation)。
    - 接收 `--delete-invalid` (布尔标志，是否在验证失败后启用删除选项)。
  - **日志设置：** 初始化一个专用的日志记录器，将验证过程的日志输出到文件和控制台。
  - 调用验证逻辑：
    - 调用 `verify_dataset_config` 函数进行基础验证。
    - 如果基础验证发现不合法数据且 `--delete-invalid` 为真，则**提供用户交互**（询问是否删除）或在非交互式环境下**自动删除**不合法文件。
    - 调用 `verify_split_uniqueness` 函数进行分割唯一性验证。
  - **结果总结：** 根据所有验证结果，输出最终的通过/失败状态，并提供清晰的日志信息。

### 技术要求

- **Python 版本：** Python 3.8+
- **依赖库：** `PyYAML`、`scikit-learn`（用于 `train_test_split`，尽管这个脚本直接使用了 `random.sample`，但作为项目依赖，保留即可）、`pathlib` (Python 内置)。
- **日志：** 必须使用 `logging` 模块和项目已有的 `logging_utils.py` 进行日志管理。
- **文件路径：** 统一使用 `pathlib.Path` 对象处理文件路径。