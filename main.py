import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr
from catboost import CatBoostRegressor
from data_loader import load_cv_folds, load_snp_data, load_exp_data, load_combined_data
import shap
import json
import os
from config import (DATA_DIR, CROSS_VALIDATION_FILE, EXP_FILE, PHENOTYPES_FILE, SNP_FILE,
                    RESULTS_DIR, PHENOTYPES, CATBOOST_PARAMS, USE_GPU, SHAP_COMPUTE)


def train_catboost(X_train, y_train, X_val, y_val):
    """
    使用CatBoost训练模型。

    Args:
        X_train (np.array): 训练特征数据
        y_train (np.array): 训练目标数据
        X_val (np.array): 验证特征数据
        y_val (np.array): 验证目标数据

    Returns:
        model: 训练好的CatBoost模型
    """
    # 初始化CatBoost回归模型
    model = CatBoostRegressor(
        **CATBOOST_PARAMS,
        task_type="GPU" if USE_GPU else "CPU",
        devices='0' if USE_GPU else None  # 指定GPU设备
    )

    # 训练模型
    model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val)
    )
    model.save_model(model_path)
    print(f"Model saved to {model_path}")
    return model


def evaluate_model(model, X_test, y_test):
    """
    评估模型性能，计算Pearson相关系数、p-value、MSE和MAE。

    Args:
        model: 训练好的模型
        X_test (np.array): 测试特征数据
        y_test (np.array): 测试目标数据

    Returns:
        dict: 包含评估指标的结果
    """
    # 预测
    y_pred = model.predict(X_test)

    # 计算评估指标
    pearson_corr, p_value = pearsonr(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    # 返回结果
    return {
        "pearson_corr": pearson_corr,
        "p_value": p_value,
        "mse": mse,
        "mae": mae
    }


def save_feature_importance_and_shap(model, X_train, feature_names, output_dir, fold_idx):
    """
    保存特征重要性和SHAP值。

    Args:
        model: 训练好的模型
        X_train (np.array): 训练特征数据
        feature_names (list): 特征名称列表
        output_dir (str): 输出目录
        fold_idx (int): 当前折的索引
    """
    # 获取特征重要性
    feature_importance = model.get_feature_importance()
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": feature_importance
    }).sort_values(by="importance", ascending=False)

    # 保存特征重要性到文件
    importance_file = f"{output_dir}/feature_importance_fold_{fold_idx}.csv"
    importance_df.to_csv(importance_file, index=False)
    print(f"Feature importance saved to {importance_file}")

    # 如果需要计算 SHAP 值
    if SHAP_COMPUTE:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_train)

        # 保存SHAP值到文件
        shap_df = pd.DataFrame(shap_values, columns=feature_names)
        shap_file = f"{output_dir}/shap_values_fold_{fold_idx}.csv"
        shap_df.to_csv(shap_file, index=False)
        print(f"SHAP values saved to {shap_file}")


def run_experiment(phenotype):
    """
       执行完整的实验流程：仅使用SNP数据进行训练、评估模型、保存结果。

       Args:
           phenotype (str): 表型名称
       """
    # 加载SNP数据
    snp_X_standardized, y_scaled, sample_ids, snp_feature_names = load_snp_data(phenotype)

    # 加载交叉验证划分
    cv_folds = load_cv_folds(CROSS_VALIDATION_FILE)

    # 创建输出目录
    output_dir = os.path.join(RESULTS_DIR, f"snp_only/{phenotype}")
    os.makedirs(output_dir, exist_ok=True)

    all_metrics = []

    # 对每个折进行训练和评估
    for fold_idx, (train_indices, val_indices) in enumerate(cv_folds):
        print(f"Processing fold {fold_idx + 1}")

        # 划分训练集和验证集
        X_train_snp, X_val_snp = snp_X_standardized[train_indices], snp_X_standardized[val_indices]
        y_train, y_val = y_scaled[train_indices], y_scaled[val_indices]

        # 训练模型
        model = train_catboost(X_train_snp, y_train, X_val_snp, y_val, snp_feature_names)

        # 评估模型
        metrics = evaluate_model(model, X_val_snp, y_val)
        print(f"Fold {fold_idx + 1} Metrics: {metrics}")
        all_metrics.append(metrics)

        # 保存特征重要性和SHAP值
        save_feature_importance_and_shap(model, X_train_snp, snp_feature_names, output_dir, fold_idx)

    # 打印所有折的平均评估指标
    avg_metrics = {
        "pearson_corr": np.mean([m["pearson_corr"] for m in all_metrics]),
        "p_value": np.mean([m["p_value"] for m in all_metrics]),
        "mse": np.mean([m["mse"] for m in all_metrics]),
        "mae": np.mean([m["mae"] for m in all_metrics])
    }
    print(f"Average Metrics across all folds: {avg_metrics}")

    # 保存平均评估指标
    with open(f"{output_dir}/average_metrics.json", "w") as f:
        json.dump(avg_metrics, f, indent=4)
    print(f"Average metrics saved to {output_dir}/average_metrics.json")


# 示例：如何调用run_experiment函数
if __name__ == "__main__":
    # 遍历所有表型并运行实验
    for phenotype in PHENOTYPES:
        print(f"Running experiment for phenotype: {phenotype}")
        run_experiment(phenotype)