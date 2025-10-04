import pandas as pd
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score,  f1_score, roc_auc_score
import numpy as np
import optuna
import json
from datetime import datetime



HYPEROPT_CONFIG = {
    'n_trials': 50,  
    'cv_folds': 3,    
    'timeout': 1800,  
    'objective': 'f1', 
    'direction': 'maximize'  
}


def model_merge(Y_HAT_REG, Y_HAT_CLS, a=3.5, b=1.5, c=0.4):
    '''
    模型融合函数
    Y_HAT_REG and Y_HAT_CLS need be scaled to 0~1
    a,b,c need to be tuned with optuna 
    '''
    y_fun = (Y_HAT_REG>0)*c*(np.abs(Y_HAT_REG))**(b)
    x_fun =(Y_HAT_CLS>0)*(np.abs(Y_HAT_CLS))**(a)
    res = (1-y_fun)*x_fun+y_fun
    res = pd.Series(res).rank()/len(res)
    return res

def objective_function(trial, X, y, ratio):
    """
    Optuna目标函数 - 针对小样本高维度数据优化
    """
    # XGBoost超参数搜索空间（针对200行1000列数据调整）
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 20, 150),
        'learning_rate': trial.suggest_float('learning_rate', 0.05, 0.2),
        'max_depth': trial.suggest_int('max_depth', 2, 6), 
        'subsample': trial.suggest_float('subsample', 0.7, 0.95),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 0.8),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.1, 2.0), 
        'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 2.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 3, 15),
        'gamma': trial.suggest_float('gamma', 0.1, 1.0) 
    }
    
    # 模型融合参数搜索空间（针对小样本数据调整）
    merge_params = {
        'a': trial.suggest_float('merge_a', 1.5, 4.0),
        'b': trial.suggest_float('merge_b', 0.8, 2.5),
        'c': trial.suggest_float('merge_c', 0.2, 0.8)
    }
    
    # 阈值搜索空间
    threshold = trial.suggest_float('threshold', 0.2, 0.8)
    
    # 交叉验证评估
    cv = StratifiedKFold(n_splits=HYPEROPT_CONFIG['cv_folds'], shuffle=True, random_state=42)
    scores = []
    
    for train_idx, val_idx in cv.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # 训练分类模型
        xgb_cls = XGBClassifier(enable_categorical=True, scale_pos_weight=ratio, **params)
        xgb_cls.fit(X_train, y_train)
        
        # 训练回归模型
        xgb_reg = XGBRegressor(enable_categorical=True, **params)
        xgb_reg.fit(X_train, y_train)
        
        # 预测
        y_pred_proba_cls = xgb_cls.predict_proba(X_val)[:, 1]
        y_pred_reg = xgb_reg.predict(X_val)
        
        # 回归结果缩放
        y_pred_reg_scaled = (y_pred_reg - y_pred_reg.min()) / (y_pred_reg.max() - y_pred_reg.min() + 1e-8)
        
        # 模型融合
        y_pred_merged = model_merge(y_pred_reg_scaled, y_pred_proba_cls, **merge_params)
        y_pred_merged_binary = (y_pred_merged >= threshold).astype(int)
        
        # 计算目标指标
        if HYPEROPT_CONFIG['objective'] == 'f1':
            score = f1_score(y_val, y_pred_merged_binary)
        elif HYPEROPT_CONFIG['objective'] == 'auc':
            score = roc_auc_score(y_val, y_pred_merged)
        elif HYPEROPT_CONFIG['objective'] == 'accuracy':
            score = accuracy_score(y_val, y_pred_merged_binary)
        else:
            score = f1_score(y_val, y_pred_merged_binary)
            
        scores.append(score)
    
    return np.mean(scores)

def run_hyperparameter_optimization(X, y, ratio):
    """
    运行超参数优化
    """
    print("开始超参数优化...")
    print(f"优化目标: {HYPEROPT_CONFIG['objective']}")
    print(f"优化轮数: {HYPEROPT_CONFIG['n_trials']}")
    print(f"交叉验证折数: {HYPEROPT_CONFIG['cv_folds']}")
    
    # 创建study
    study = optuna.create_study(
        direction=HYPEROPT_CONFIG['direction'],
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    
    # 运行优化
    study.optimize(
        lambda trial: objective_function(trial, X, y, ratio),
        n_trials=HYPEROPT_CONFIG['n_trials'],
        timeout=HYPEROPT_CONFIG['timeout']
    )
    
    print(f"优化完成! 最佳分数: {study.best_value:.4f}")
    print("最佳超参数:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    return study

def save_optimization_results(study, filename=None):
    """
    保存优化结果
    """
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"hyperopt_results_{timestamp}.json"
    
    results = {
        'best_value': study.best_value,
        'best_params': study.best_params,
        'n_trials': len(study.trials),
        'config': HYPEROPT_CONFIG,
        'timestamp': datetime.now().isoformat()
    }
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"优化结果已保存到: {filename}")
    return filename

