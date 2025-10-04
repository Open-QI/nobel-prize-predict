# nobel-prize-predict

通过机器学习预测是否可能获得诺贝尔奖，本项目仅用于学术研究和科普用途，预测结果不代表实际获奖可能性。

## 预测思路
本项目以**学术引用量**为核心特征，结合多背景信息，通过机器学习方法预测学者获得诺贝尔奖的可能性。

## 特征工程 
### 基础数值特征
- **基础计量指标**：总引用量、h-index、i10-Index
- **时序引用分析**：分年度的总引用统计，避免获奖后引用激增导致的预测偏差
- **代表作影响力**：代表性论文的总引用与年度引用变化
  
### 类别特征
- **地域信息**：出生国家与当前所在国家
- **学术归属**：研究机构
- **人口统计**：性别
  
### 其他衍生特征
- **语义类别标签**：通过作者维基百科页面，利用LLM提取并总结标签特征
- **机构标准化**：将分支机构或子机构处理为统一的上层机构，减少类别过度分散
- **时间对齐处理**：将作者分年引用量对齐到获奖年份的前n年
- **累积指标构建**：计算不同时间段的累积引用统计
- **增长动态特征**：引用量的增长率及变化趋势分析

## 模型架构
我们的基础模型采用XGBoost作为核心算法，将作者的类别特征与时序特征结合，建立预测基线。考虑到学术奖项评选依赖学者影响力的演变历程，我们引入GRU网络优化时序特征表达，专门处理论文引用的动态变化模式，捕捉引用增长率、爆发与衰减特征。通过记忆门和更新门机制，有效识别长短期学术影响的里程碑和转折点。

## 主要数据源
- [Google Scholar](https://scholar.google.com/)
- [OpenAlex](https://openalex.org/)
- [Wikipedia](https://www.wikipedia.org/)

## 模型融合
同时训练分类器和回归器，并将其结果进行集成融合，提高预测稳定性。

## 如何运行

### 环境准备

本项目可在 CPU 上运行，无需 GPU；若有 CUDA/GPU 可获得更快速度。

**方式 1：使用 uv（推荐）**
```bash
uv venv --python 3.11
source .venv/bin/activate  # Windows: .\.venv\Scripts\activate
uv pip install -r requirements.txt
```

**方式 2：使用 pip**
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .\.venv\Scripts\activate
pip install -r requirements.txt
```

### 数据配置

- 默认使用 `data/nobel_dataset_sampled_100.csv`
- 可通过环境变量自定义数据路径：
  ```bash
  # macOS/Linux
  export DATA_CSV=/path/to/your.csv

  # Windows PowerShell
  $env:DATA_CSV="C:\path\to\your.csv"
  ```

### 模型训练与评估

**1. XGBoost 基线模型**
```bash
python model.py
```
控制台将输出交叉验证性能指标。

**2. GRU 时序模型**
```bash
python model_gru.py
```
ROC 曲线将保存至 `outputs/roc_curves_gru.png`。

**3. 生成预测候选特征**
```bash
python tools/features_predict.py
```
此脚本会：
- 加载候选学者数据（label=0）
- 加载 2022-2023 年获奖者数据用于测试验证
- 输出合并后的特征数据，用于后续预测打分

**4. 探索性数据分析（可选）**
```bash
python tools/eda.py
```
在 `eda_compare/` 目录下生成分布对比图，便于探索特征差异。

### GPU 加速（可选）

- 代码会自动检测 CUDA(only tests on LINUX CUDA 12.4)，无 GPU 时自动使用 CPU
- 如需使用 GPU：
  - 安装对应 CUDA 版本的 PyTorch
  - 在 XGBoost 中可选设置 `tree_method='gpu_hist'`

### 注意事项

- ⚠️ 请从仓库根目录运行以上命令（代码导入时会进行特征构建）
- 依赖列表见 `requirements.txt`
- 如仅需运行 XGBoost 基线模型，可不安装 PyTorch
