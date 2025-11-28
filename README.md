# MR-TADF 分子数据集

## 数据集说明

本数据集包含585个MR-TADF（多重共振热激活延迟荧光）分子数据，用于机器学习预测分子的光物理性质。

**数据集特点：**
- 提供多种分子指纹和图特征
- 包含两个目标属性：deerta_EST（单三态能级差）和FWHM（半峰全宽）

## 数据集内容

每个分子记录包含以下信息：

- `FileName`: 分子文件名
- `molecule_id`: 分子唯一标识
- `morgan_fp_2048`: Morgan指纹（2048位）
- `morgan_fp_1024`: Morgan指纹（1024位）
- `maccs_fp`: MACCS指纹
- `atom_features`: 原子特征列表
- `bond_features`: 化学键特征列表
- `adjacency_matrix`: 邻接矩阵
- `num_atoms`: 原子数量
- `num_bonds`: 化学键数量
- `deerta_EST`: 单三态能级差（目标属性1）
- `FWHM`: 半峰宽（目标属性2）

## 安装依赖
```bash
pip install pandas numpy pickle
```

## 快速开始

### 1. 加载数据集
```python
import pickle
import numpy as np
import pandas as pd

class SecureDataset:
    """数据集加载器"""
    
    def __init__(self, pkl_file='MR585.pkl'):
        with open(pkl_file, 'rb') as f:
            self.data = pickle.load(f)
        print(f"加载 {len(self.data)} 个分子")
    
    def get_morgan_fp(self, nbits=2048):
        """获取Morgan指纹"""
        key = f'morgan_fp_{nbits}'
        return np.array([item[key] for item in self.data])
    
    def get_maccs_fp(self):
        """获取MACCS指纹"""
        return np.array([item['maccs_fp'] for item in self.data])
    
    def get_properties(self):
        """获取目标属性"""
        return {
            'deerta_EST': np.array([item['deerta_EST'] for item in self.data]),
            'FWHM': np.array([item['FWHM'] for item in self.data])
        }
    
    def get_graph_data(self, idx):
        """获取单个分子的图数据"""
        item = self.data[idx]
        return {
            'atom_features': item['atom_features'],
            'bond_features': item['bond_features'],
            'adjacency_matrix': item['adjacency_matrix'],
            'num_atoms': item['num_atoms'],
            'num_bonds': item['num_bonds']
        }
    
    def to_dataframe(self):
        """转换为DataFrame"""
        records = []
        for item in self.data:
            records.append({
                'FileName': item['FileName'],
                'molecule_id': item['molecule_id'],
                'num_atoms': item['num_atoms'],
                'num_bonds': item['num_bonds'],
                'deerta_EST': item['deerta_EST'],
                'FWHM': item['FWHM']
            })
        return pd.DataFrame(records)

# 加载数据
dataset = SecureDataset('MR585.pkl')
```

### 2. 查看数据概览
```python
# 转换为DataFrame
df = dataset.to_dataframe()
print(df.head())
print(df.describe())
```

### 3. 获取分子指纹
```python
# 获取Morgan指纹（2048位）
morgan_fp = dataset.get_morgan_fp(2048)
print(f"指纹矩阵形状: {morgan_fp.shape}")

# 获取MACCS指纹
maccs_fp = dataset.get_maccs_fp()
print(f"MACCS指纹形状: {maccs_fp.shape}")
```

### 4. 获取目标属性
```python
# 获取目标属性
properties = dataset.get_properties()
deerta_EST = properties['deerta_EST']
FWHM = properties['FWHM']

# 打印每个分子的目标属性
for i in range(len(deerta_EST)):
    print(f"分子 {i+1}: deerta_EST = {deerta_EST[i]:.4f}, FWHM = {FWHM[i]:.4f}")
```

### 5. 获取图结构数据
```python
# 获取单个分子的图数据
graph_data = dataset.get_graph_data(0)

print(f"原子数: {graph_data['num_atoms']}")
print(f"化学键数: {graph_data['num_bonds']}")
print(f"邻接矩阵形状: {graph_data['adjacency_matrix'].shape}")
print(f"原子特征示例: {graph_data['atom_features'][0]}")
```

## 使用场景

- 分子性质预测
- 分子相似性分析
- 图神经网络建模
- 化学信息学研究

## 许可

本数据集仅供学术研究使用，未经授权不得用于商业用途。
