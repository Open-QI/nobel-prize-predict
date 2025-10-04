import pandas as pd
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import numpy as np

from features import (df, 
    years_before_award_columns,
    growth_analysis_columns,
    work_growth_analysis_columns,
    work_years_before_award_columns)  

from mappings import institution_onehot

# 使用 features.py 中已构建的 df
# 保留列n
keep_cols = ['original_name','label','birth_year']
weight_cols = ['internet_weight']
num_cols = years_before_award_columns + growth_analysis_columns + work_growth_analysis_columns + work_years_before_award_columns + institution_onehot
cat_cols = ['extraction.field','gender','origin_country','current_country']


initial_hyperparameters = {
    'n_estimators': 100,
    'learning_rate': 0.1,
    'max_depth': 3,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1
}


accuracy_scores = []
precision_scores = []
recall_scores = []
f1_scores = []
auc_scores = []

# 融合模型性能统计
accuracy_scores_merged = []
precision_scores_merged = []
recall_scores_merged = []
f1_scores_merged = []
auc_scores_merged = []


X = pd.concat([
    df.loc[:, num_cols].apply(pd.to_numeric, errors='coerce'),
    df.loc[:, cat_cols].astype('category')
    ],axis=1).copy()

y = df['label'].copy()

# 计算正负样本比例
neg_count = (y == 0).sum()
pos_count = (y == 1).sum()
ratio = neg_count / pos_count
print(f"Positive count: {pos_count}, Negative count: {neg_count}, PosRatio: {(pos_count / (pos_count + neg_count)):.4f}")

xgb_cls_model = XGBClassifier(enable_categorical=True, scale_pos_weight=ratio, **initial_hyperparameters)
xgb_reg_model = XGBRegressor(enable_categorical=True, **initial_hyperparameters)
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

fold_dfs = []

for fold, (train_index, test_index) in enumerate(cv.split(X, y)):
    print(f"Processing Fold {fold + 1}...")
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # 从训练集分出验证集用于阈值优化
    X_train_fold, X_val_fold, y_train_fold, y_val_fold = train_test_split(
        X_train, y_train, test_size=0.2, stratify=y_train, random_state=42
    )
    
    # 训练分类模型
    xgb_cls_model.fit(X_train_fold, y_train_fold)
    
    # 训练回归模型
    xgb_reg_model.fit(X_train_fold, y_train_fold)
    
    # 阈值优化：在验证集上找最佳阈值
    y_val_proba = xgb_cls_model.predict_proba(X_val_fold)[:, 1]
    best_threshold = 0.5
    best_f1 = 0
    for threshold in np.linspace(0.2, 0.8, 25):
        y_val_pred = (y_val_proba >= threshold).astype(int)
        f1 = f1_score(y_val_fold, y_val_pred)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    # 用最佳阈值预测测试集
    y_pred_proba_cls = xgb_cls_model.predict_proba(X_test)[:, 1]
    y_pred_cls = (y_pred_proba_cls >= best_threshold).astype(int)
    
    # 回归模型预测
    y_pred_reg = xgb_reg_model.predict(X_test)
    
    # 将回归结果缩放到0-1范围（避免分母为0）
    denom = (y_pred_reg.max() - y_pred_reg.min())
    if denom == 0:
        denom = 1e-8
    y_pred_reg_scaled = (y_pred_reg - y_pred_reg.min()) / denom
    
    # 模型融合
    def model_merge(Y_HAT_REG, Y_HAT_CLS, a=3.5, b=1.5, c=0.4):
        '''
        Y_HAT_REG and Y_HAT_CLS need be scaled to 0~1
        a,b,c need to be tuned with optuna 
        '''
        y_fun = (Y_HAT_REG>0)*c*(np.abs(Y_HAT_REG))**(b)
        x_fun =(Y_HAT_CLS>0)*(np.abs(Y_HAT_CLS))**(a)
        res = (1-y_fun)*x_fun+y_fun
        res = pd.Series(res).rank()/len(res)
        return res
    
    # 融合预测结果
    y_pred_merged = model_merge(y_pred_reg_scaled, y_pred_proba_cls)
    y_pred_merged_binary = (y_pred_merged >= 0.5).astype(int)

    # 构建包含原始信息的fold级别结果表
    train_df = df.iloc[train_index].copy()
    train_df['tag'] = 'train'


    test_df = df.iloc[test_index].copy()
    test_df['tag'] = 'test'
    test_df['y_pred_cls'] = y_pred_cls
    test_df['y_pred_proba_cls'] = y_pred_proba_cls
    test_df['y_pred_reg'] = y_pred_reg
    test_df['y_pred_reg_scaled'] = y_pred_reg_scaled
    test_df['y_pred_merged'] = y_pred_merged
    test_df['y_pred_merged_binary'] = y_pred_merged_binary

    fold_df = pd.concat([train_df, test_df], axis=0)
    fold_df['fold'] = fold + 1
    fold_dfs.append(fold_df)


    # 计算分类模型性能
    accuracy_cls = accuracy_score(y_test, y_pred_cls)
    precision_cls = precision_score(y_test, y_pred_cls)
    recall_cls = recall_score(y_test, y_pred_cls)
    f1_cls = f1_score(y_test, y_pred_cls)
    auc_cls = roc_auc_score(y_test, y_pred_proba_cls)
    
    # 计算融合模型性能
    accuracy_merged = accuracy_score(y_test, y_pred_merged_binary)
    precision_merged = precision_score(y_test, y_pred_merged_binary)
    recall_merged = recall_score(y_test, y_pred_merged_binary)
    f1_merged = f1_score(y_test, y_pred_merged_binary)
    auc_merged = roc_auc_score(y_test, y_pred_merged)

    accuracy_scores.append(accuracy_cls)
    precision_scores.append(precision_cls)
    recall_scores.append(recall_cls)
    f1_scores.append(f1_cls)
    auc_scores.append(auc_cls)
    
    # 添加融合模型性能
    accuracy_scores_merged.append(accuracy_merged)
    precision_scores_merged.append(precision_merged)
    recall_scores_merged.append(recall_merged)
    f1_scores_merged.append(f1_merged)
    auc_scores_merged.append(auc_merged)

    print(f"Fold {fold + 1} - Classification: Accuracy={accuracy_cls:.4f}, Precision={precision_cls:.4f}, Recall={recall_cls:.4f}, F1-score={f1_cls:.4f}, AUC={auc_cls:.4f} (threshold={best_threshold:.3f})")
    print(f"Fold {fold + 1} - Merged: Accuracy={accuracy_merged:.4f}, Precision={precision_merged:.4f}, Recall={recall_merged:.4f}, F1-score={f1_merged:.4f}, AUC={auc_merged:.4f}")


print("\nMean scores across all folds:")
print("=" * 50)
print("Classification Model:")
print(f"Mean Accuracy: {np.mean(accuracy_scores):.4f}")
print(f"Mean Precision: {np.mean(precision_scores):.4f}")
print(f"Mean Recall: {np.mean(recall_scores):.4f}")
print(f"Mean F1-score: {np.mean(f1_scores):.4f}")
print(f"Mean AUC: {np.mean(auc_scores):.4f}")
print("=" * 50)
print("Merged Model:")
print(f"Mean Accuracy: {np.mean(accuracy_scores_merged):.4f}")
print(f"Mean Precision: {np.mean(precision_scores_merged):.4f}")
print(f"Mean Recall: {np.mean(recall_scores_merged):.4f}")
print(f"Mean F1-score: {np.mean(f1_scores_merged):.4f}")
print(f"Mean AUC: {np.mean(auc_scores_merged):.4f}")
print("=" * 50)
