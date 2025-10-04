import warnings
import os

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt

from features import (df, 
    years_before_award_columns,
    growth_analysis_columns,
    work_growth_analysis_columns,
    work_years_before_award_columns)  
from mappings import institution_onehot

warnings.filterwarnings('ignore')

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

# 数据准备
df = df.copy()
keep_cols = ['original_name','label','birth_year']
weight_cols = ['internet_weight']
cat_cols = ['extraction.field','gender','origin_country','current_country']

# 构建时序数据列名
def get_sequential_columns():
    """构建GRU输入的时序数据列名"""
    sequential_cols = []
    
    # 添加citations的时序数据 (years_before_award_columns)
    for year in range(1, 20):
        sequential_cols.append(f'citations_{year}_years_before_award')
    
    # 添加citations的累积列
    sequential_cols.extend([
        'citations_20_to_25_years_before_cumulative',
        'citations_26_to_35_years_before_cumulative',
        'citations_36_to_50_years_before_cumulative'
    ])
    
    # 为每个top_k (1-5) 构建时序数据
    for top_k in range(1, 6):
        # 1-19年的独立数据
        for year in range(1, 20):
            sequential_cols.append(f'top{top_k}_citations_{year}_years_before_award')
        
        # 3个累积列
        sequential_cols.extend([
            f'top{top_k}_citations_20_to_25_years_before_cumulative',
            f'top{top_k}_citations_26_to_35_years_before_cumulative',
            f'top{top_k}_citations_36_to_50_years_before_cumulative'
        ])
    
    return sequential_cols

# 获取所有特征列
sequential_cols = get_sequential_columns()
num_cols = years_before_award_columns + growth_analysis_columns + work_growth_analysis_columns + work_years_before_award_columns + institution_onehot

# 检查列是否存在
available_sequential_cols = [col for col in sequential_cols if col in df.columns]
print(f"Available sequential columns: {len(available_sequential_cols)}")

# 构建特征矩阵
X_sequential = df[available_sequential_cols].fillna(0).values
X_numerical = df[num_cols].fillna(0).values
X_categorical = df[cat_cols].fillna('Unknown').values
y = df['label'].values

# 计算正负样本比例
neg_count = (y == 0).sum()
pos_count = (y == 1).sum()
ratio = neg_count / pos_count
print(f"Positive count: {pos_count}, Negative count: {neg_count}, PosRatio: {(pos_count / (pos_count + neg_count)):.4f}")

# 标准化数值特征
scaler_numerical = StandardScaler()
X_numerical_scaled = scaler_numerical.fit_transform(X_numerical)

# 对类别特征进行编码
label_encoders = {}
X_categorical_encoded = np.zeros_like(X_categorical, dtype=int)

for i, col in enumerate(cat_cols):
    le = LabelEncoder()
    X_categorical_encoded[:, i] = le.fit_transform(X_categorical[:, i])
    label_encoders[col] = le

# 计算类别特征的维度
cat_dims = [len(le.classes_) for le in label_encoders.values()]

# 构建时序数据 (batch_size, time_steps, features)
# 现在有6个维度：1个citations + 5个top_k，每个维度有22个时间步：1-19年 + 3个累积列
time_steps = 22 
features_per_dim = 1
num_dims = 6 

# 重塑时序数据
X_sequential_reshaped = X_sequential.reshape(-1, num_dims, time_steps, features_per_dim)

print(f"Sequential data shape: {X_sequential_reshaped.shape}")
print(f"Numerical data shape: {X_numerical_scaled.shape}")
print(f"Categorical data shape: {X_categorical_encoded.shape}")

class GRUDataset(Dataset):
    def __init__(self, X_seq, X_num, X_cat, y):
        self.X_seq = torch.FloatTensor(X_seq)
        self.X_num = torch.FloatTensor(X_num)
        self.X_cat = torch.LongTensor(X_cat)
        self.y = torch.LongTensor(y)
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X_seq[idx], self.X_num[idx], self.X_cat[idx], self.y[idx]

class GRUModel(nn.Module):
    def __init__(self, seq_input_dim, num_input_dim, cat_dims, hidden_dim=128, dropout=0.3):
        super(GRUModel, self).__init__()
        
        # 时序特征处理 (GRU)
        self.gru = nn.GRU(
            input_size=seq_input_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )
        
        # 数值特征处理 (MLP)
        self.num_mlp = nn.Sequential(
            nn.Linear(num_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 类别特征处理 (Embedding + MLP)
        self.cat_embeddings = nn.ModuleList([
            nn.Embedding(cat_dim, min(50, (cat_dim + 1) // 2)) for cat_dim in cat_dims
        ])
        
        cat_embed_dim = sum([min(50, (cat_dim + 1) // 2) for cat_dim in cat_dims])
        self.cat_mlp = nn.Sequential(
            nn.Linear(cat_embed_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 融合层
        total_dim = hidden_dim * 2 + hidden_dim // 2 + hidden_dim // 2  # gru + num + cat
        self.fusion = nn.Sequential(
            nn.Linear(total_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 分类和回归头
        self.classifier = nn.Linear(hidden_dim // 2, 1)
        self.regressor = nn.Linear(hidden_dim // 2, 1)
        
    def forward(self, x_seq, x_num, x_cat):
        # 处理时序数据
        # x_seq shape: (batch, num_dims, time_steps, features)
        batch_size = x_seq.size(0)
        
        # 重塑为 (batch * num_dims, time_steps, features)
        x_seq_reshaped = x_seq.view(batch_size * 6, -1, 1)
        
        # GRU处理
        gru_out, _ = self.gru(x_seq_reshaped)  # (batch * 6, time_steps, hidden_dim * 2)
        
        # 取最后一个时间步的输出
        gru_final = gru_out[:, -1, :]  # (batch * 6, hidden_dim * 2)
        
        # 重塑回 (batch, num_dims, hidden_dim * 2)
        gru_final = gru_final.view(batch_size, 6, -1)
        
        # 对6个维度进行平均池化
        gru_pooled = torch.mean(gru_final, dim=1)  # (batch, hidden_dim * 2)
        
        # 处理数值特征
        num_out = self.num_mlp(x_num)
        
        # 处理类别特征
        cat_embeds = []
        for i, embedding in enumerate(self.cat_embeddings):
            cat_embeds.append(embedding(x_cat[:, i]))
        cat_embed = torch.cat(cat_embeds, dim=1)
        cat_out = self.cat_mlp(cat_embed)
        
        # 融合所有特征
        combined = torch.cat([gru_pooled, num_out, cat_out], dim=1)
        fused = self.fusion(combined)
        
        # 分类和回归输出
        cls_output = torch.sigmoid(self.classifier(fused))
        reg_output = self.regressor(fused)
        
        return cls_output.squeeze(), reg_output.squeeze()

def train_model(model, train_loader, val_loader, device, epochs=50, lr=0.001):
    """训练模型，并在验证阶段进行阈值优化"""
    criterion_cls = nn.BCELoss()
    criterion_reg = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    
    best_val_loss = float('inf')
    best_threshold = 0.5
    patience = 5
    patience_counter = 0
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        for x_seq, x_num, x_cat, y in train_loader:
            x_seq, x_num, x_cat, y = x_seq.to(device), x_num.to(device), x_cat.to(device), y.to(device)
            
            optimizer.zero_grad()
            cls_pred, reg_pred = model(x_seq, x_num, x_cat)
            
            # 计算损失
            cls_loss = criterion_cls(cls_pred, y.float())
            reg_loss = criterion_reg(reg_pred, y.float())
            total_loss = cls_loss + 0.1 * reg_loss
            
            total_loss.backward()
            optimizer.step()
            
            train_loss += total_loss.item()
        
        # 验证阶段
        model.eval()
        val_loss = 0
        all_cls_preds = []
        all_labels = []
        
        with torch.no_grad():
            for x_seq, x_num, x_cat, y in val_loader:
                x_seq, x_num, x_cat, y = x_seq.to(device), x_num.to(device), x_cat.to(device), y.to(device)
                cls_pred, reg_pred = model(x_seq, x_num, x_cat)
                
                cls_loss = criterion_cls(cls_pred, y.float())
                reg_loss = criterion_reg(reg_pred, y.float())
                total_loss = cls_loss + 0.1 * reg_loss
                
                val_loss += total_loss.item()
                
                # 收集预测结果用于阈值优化
                all_cls_preds.extend(cls_pred.cpu().numpy())
                all_labels.extend(y.cpu().numpy())
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        # 在验证集上进行阈值优化
        all_cls_preds = np.array(all_cls_preds)
        all_labels = np.array(all_labels)
        
        best_f1 = 0
        current_threshold = 0.5
        for threshold in np.linspace(0.2, 0.8, 25):
            y_pred = (all_cls_preds >= threshold).astype(int)
            f1 = f1_score(all_labels, y_pred)
            if f1 > best_f1:
                best_f1 = f1
                current_threshold = threshold
        
        # 如果当前epoch的F1分数更好，更新最佳阈值
        if best_f1 > 0:
            best_threshold = current_threshold
        
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Best F1: {best_f1:.4f}, Best Threshold: {best_threshold:.3f}")
    
    return model, best_threshold

def evaluate_model(model, test_loader, device, best_threshold=0.5):
    """评估模型，使用训练时确定的最优阈值"""
    model.eval()
    all_cls_preds = []
    all_reg_preds = []
    all_labels = []
    
    with torch.no_grad():
        for x_seq, x_num, x_cat, y in test_loader:
            x_seq, x_num, x_cat, y = x_seq.to(device), x_num.to(device), x_cat.to(device), y.to(device)
            cls_pred, reg_pred = model(x_seq, x_num, x_cat)
            
            all_cls_preds.extend(cls_pred.cpu().numpy())
            all_reg_preds.extend(reg_pred.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    
    all_cls_preds = np.array(all_cls_preds)
    all_reg_preds = np.array(all_reg_preds)
    all_labels = np.array(all_labels)
    
    # 使用训练时确定的最佳阈值进行预测
    y_pred_cls = (all_cls_preds >= best_threshold).astype(int)
    
    # 模型融合（类似原版）
    def model_merge(Y_HAT_REG, Y_HAT_CLS, a=3.5, b=1.5, c=0.4):
        y_fun = (Y_HAT_REG>0)*c*(np.abs(Y_HAT_REG))**(b)
        x_fun =(Y_HAT_CLS>0)*(np.abs(Y_HAT_CLS))**(a)
        res = (1-y_fun)*x_fun+y_fun
        res = pd.Series(res).rank()/len(res)
        return res
    
    # 将回归结果缩放到0-1范围（避免分母为0）
    denom = (all_reg_preds.max() - all_reg_preds.min())
    if denom == 0:
        denom = 1e-8
    y_pred_reg_scaled = (all_reg_preds - all_reg_preds.min()) / denom
    
    # 融合预测结果
    y_pred_merged = model_merge(y_pred_reg_scaled, all_cls_preds)
    y_pred_merged_binary = (y_pred_merged >= 0.5).astype(int)
    
    # 计算指标
    accuracy_cls = accuracy_score(all_labels, y_pred_cls)
    precision_cls = precision_score(all_labels, y_pred_cls)
    recall_cls = recall_score(all_labels, y_pred_cls)
    f1_cls = f1_score(all_labels, y_pred_cls)
    auc_cls = roc_auc_score(all_labels, all_cls_preds)
    
    accuracy_merged = accuracy_score(all_labels, y_pred_merged_binary)
    precision_merged = precision_score(all_labels, y_pred_merged_binary)
    recall_merged = recall_score(all_labels, y_pred_merged_binary)
    f1_merged = f1_score(all_labels, y_pred_merged_binary)
    auc_merged = roc_auc_score(all_labels, y_pred_merged)
    
    # 计算ROC曲线数据
    fpr_cls, tpr_cls, _ = roc_curve(all_labels, all_cls_preds)
    fpr_merged, tpr_merged, _ = roc_curve(all_labels, y_pred_merged)
    
    return {
        'cls': {
            'accuracy': accuracy_cls,
            'precision': precision_cls,
            'recall': recall_cls,
            'f1': f1_cls,
            'auc': auc_cls,
            'threshold': best_threshold,
            'fpr': fpr_cls,
            'tpr': tpr_cls
        },
        'merged': {
            'accuracy': accuracy_merged,
            'precision': precision_merged,
            'recall': recall_merged,
            'f1': f1_merged,
            'auc': auc_merged,
            'fpr': fpr_merged,
            'tpr': tpr_merged
        }
    }

def plot_roc_curves(all_results):
    """绘制所有fold的平均ROC曲线"""
    plt.figure(figsize=(12, 5))
    
    # 子图1：分类模型ROC曲线
    plt.subplot(1, 2, 1)
    
    # 计算每个fold的ROC曲线
    fold_fprs_cls = []
    fold_tprs_cls = []
    fold_aucs_cls = []
    
    for fold_results in all_results:
        fpr = fold_results['cls']['fpr']
        tpr = fold_results['cls']['tpr']
        auc = fold_results['cls']['auc']
        
        fold_fprs_cls.append(fpr)
        fold_tprs_cls.append(tpr)
        fold_aucs_cls.append(auc)
        
        plt.plot(fpr, tpr, alpha=0.3, color='lightblue', linewidth=1)
    
    # 计算平均ROC曲线
    mean_fpr = np.linspace(0, 1, 100)
    mean_tprs = []
    
    for i in range(len(fold_fprs_cls)):
        interp_tpr = np.interp(mean_fpr, fold_fprs_cls[i], fold_tprs_cls[i])
        mean_tprs.append(interp_tpr)
    
    mean_tpr = np.mean(mean_tprs, axis=0)
    mean_auc = np.mean(fold_aucs_cls)
    std_auc = np.std(fold_aucs_cls)
    
    plt.plot(mean_fpr, mean_tpr, color='blue', linewidth=2, 
             label=f'Classification Model (AUC = {mean_auc:.3f} ± {std_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Classifier')
    
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Classification Model ROC Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 子图2：融合模型ROC曲线
    plt.subplot(1, 2, 2)
    
    # 计算每个fold的ROC曲线
    fold_fprs_merged = []
    fold_tprs_merged = []
    fold_aucs_merged = []
    
    for fold_results in all_results:
        fpr = fold_results['merged']['fpr']
        tpr = fold_results['merged']['tpr']
        auc = fold_results['merged']['auc']
        
        fold_fprs_merged.append(fpr)
        fold_tprs_merged.append(tpr)
        fold_aucs_merged.append(auc)
        
        plt.plot(fpr, tpr, alpha=0.3, color='lightcoral', linewidth=1)
    
    # 计算平均ROC曲线
    mean_tprs_merged = []
    
    for i in range(len(fold_fprs_merged)):
        interp_tpr = np.interp(mean_fpr, fold_fprs_merged[i], fold_tprs_merged[i])
        mean_tprs_merged.append(interp_tpr)
    
    mean_tpr_merged = np.mean(mean_tprs_merged, axis=0)
    mean_auc_merged = np.mean(fold_aucs_merged)
    std_auc_merged = np.std(fold_aucs_merged)
    
    plt.plot(mean_fpr, mean_tpr_merged, color='red', linewidth=2,
             label=f'Merged Model (AUC = {mean_auc_merged:.3f} ± {std_auc_merged:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Classifier')
    
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Merged Model ROC Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    # Save to a repo-local output directory
    out_dir = "outputs"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'roc_curves_gru.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nROC Curves saved to: {out_path}")
    print(f"Classification Model - Mean AUC: {mean_auc:.4f} ± {std_auc:.4f}")
    print(f"Merged Model - Mean AUC: {mean_auc_merged:.4f} ± {std_auc_merged:.4f}")

# 主训练循环
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 交叉验证
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

accuracy_scores = []
precision_scores = []
recall_scores = []
f1_scores = []
auc_scores = []

accuracy_scores_merged = []
precision_scores_merged = []
recall_scores_merged = []
f1_scores_merged = []
auc_scores_merged = []

fold_dfs = []
all_results = []

for fold, (train_index, test_index) in enumerate(cv.split(X_sequential_reshaped, y)):
    print(f"\nProcessing Fold {fold + 1}...")
    
    # 分割数据
    X_seq_train, X_seq_test = X_sequential_reshaped[train_index], X_sequential_reshaped[test_index]
    X_num_train, X_num_test = X_numerical_scaled[train_index], X_numerical_scaled[test_index]
    X_cat_train, X_cat_test = X_categorical_encoded[train_index], X_categorical_encoded[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    # 进一步分割训练集和验证集
    X_seq_train_fold, X_seq_val_fold, X_num_train_fold, X_num_val_fold, X_cat_train_fold, X_cat_val_fold, y_train_fold, y_val_fold = train_test_split(
        X_seq_train, X_num_train, X_cat_train, y_train, test_size=0.2, stratify=y_train, random_state=42
    )
    
    # 创建数据加载器
    train_dataset = GRUDataset(X_seq_train_fold, X_num_train_fold, X_cat_train_fold, y_train_fold)
    val_dataset = GRUDataset(X_seq_val_fold, X_num_val_fold, X_cat_val_fold, y_val_fold)
    test_dataset = GRUDataset(X_seq_test, X_num_test, X_cat_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # 创建模型
    model = GRUModel(
        seq_input_dim=1,
        num_input_dim=X_numerical_scaled.shape[1],
        cat_dims=cat_dims,
        hidden_dim=128,
        dropout=0.3
    ).to(device)
    
    # 训练模型
    model, best_threshold = train_model(model, train_loader, val_loader, device, epochs=100, lr=0.001)
    
    # 评估模型，使用训练时确定的最优阈值
    results = evaluate_model(model, test_loader, device, best_threshold)
    
    # 记录结果
    accuracy_scores.append(results['cls']['accuracy'])
    precision_scores.append(results['cls']['precision'])
    recall_scores.append(results['cls']['recall'])
    f1_scores.append(results['cls']['f1'])
    auc_scores.append(results['cls']['auc'])
    
    accuracy_scores_merged.append(results['merged']['accuracy'])
    precision_scores_merged.append(results['merged']['precision'])
    recall_scores_merged.append(results['merged']['recall'])
    f1_scores_merged.append(results['merged']['f1'])
    auc_scores_merged.append(results['merged']['auc'])
    
    # 保存详细结果用于ROC曲线绘制
    all_results.append(results)
    
    print(f"Fold {fold + 1} - Classification: Accuracy={results['cls']['accuracy']:.4f}, Precision={results['cls']['precision']:.4f}, Recall={results['cls']['recall']:.4f}, F1-score={results['cls']['f1']:.4f}, AUC={results['cls']['auc']:.4f} (threshold={results['cls']['threshold']:.3f})")
    print(f"Fold {fold + 1} - Merged: Accuracy={results['merged']['accuracy']:.4f}, Precision={results['merged']['precision']:.4f}, Recall={results['merged']['recall']:.4f}, F1-score={results['merged']['f1']:.4f}, AUC={results['merged']['auc']:.4f}")

# 输出最终结果
print("\n" + "="*60)
print("GRU Model Results:")
print("="*60)
print("Classification Model:")
print(f"Mean Accuracy: {np.mean(accuracy_scores):.4f} ± {np.std(accuracy_scores):.4f}")
print(f"Mean Precision: {np.mean(precision_scores):.4f} ± {np.std(precision_scores):.4f}")
print(f"Mean Recall: {np.mean(recall_scores):.4f} ± {np.std(recall_scores):.4f}")
print(f"Mean F1-score: {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")
print(f"Mean AUC: {np.mean(auc_scores):.4f} ± {np.std(auc_scores):.4f}")
print("="*60)
print("Merged Model:")
print(f"Mean Accuracy: {np.mean(accuracy_scores_merged):.4f} ± {np.std(accuracy_scores_merged):.4f}")
print(f"Mean Precision: {np.mean(precision_scores_merged):.4f} ± {np.std(precision_scores_merged):.4f}")
print(f"Mean Recall: {np.mean(recall_scores_merged):.4f} ± {np.std(recall_scores_merged):.4f}")
print(f"Mean F1-score: {np.mean(f1_scores_merged):.4f} ± {np.std(f1_scores_merged):.4f}")
print(f"Mean AUC: {np.mean(auc_scores_merged):.4f} ± {np.std(auc_scores_merged):.4f}")
print("="*60)
