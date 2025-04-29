# data_loader.py
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from config import *


def load_cv_folds(cv_file_path):
    """
    从JSON文件加载交叉验证划分
    """
    with open(cv_file_path, 'r') as f:
        folds_data = json.load(f)
    return folds_data


def load_exp_data(phenotype):
    """
    加载基因表达数据，处理缺失值和异常值

    Returns:
        exp_X_standardized (np.array): 标准化的表达数据
        y_scaled (np.array): 标准化的表型数据
        sample_ids (list): 样本ID列表
        exp_feature_names (list): 表达特征名称列表
    """
    # 加载表型数据并过滤缺失值
    phenotypes_df = pd.read_csv(PHENOTYPES_FILE)
    phenotypes_df = phenotypes_df[phenotypes_df[phenotype].notna()]

    # 加载表达数据
    data_frame = pd.read_csv(EXP_FILE)

    # 保留同时存在表型和表达数据的样本
    data_frame = data_frame[data_frame['ID'].isin(phenotypes_df['ID'])]

    # 处理数据中的异常值
    numeric_columns = data_frame.select_dtypes(include=[np.number]).columns.drop('ID')

    # 获取特征名称（去掉ID列）
    exp_feature_names = list(numeric_columns)

    # 替换无穷大值
    data_frame[numeric_columns] = data_frame[numeric_columns].replace([np.inf, -np.inf], np.nan)

    # 移除全为NaN的列
    data_frame = data_frame.dropna(axis=1, how='all')

    # 使用列均值填充剩余的NaN
    data_frame[numeric_columns] = data_frame[numeric_columns].fillna(data_frame[numeric_columns].mean())

    # 提取样本ID和特征
    sample_ids = data_frame['ID'].tolist()
    exp_X = data_frame.drop(['ID'], axis=1).astype(np.float32).values

    # 获取表型值
    y = np.array([phenotypes_df.loc[phenotypes_df['ID'] == item, phenotype].values[0] for item in sample_ids])

    # 处理表型数据中的异常值
    y = np.nan_to_num(y, nan=np.nanmean(y))

    # 标准化
    X_scaler = StandardScaler()
    y_scaler = StandardScaler()

    exp_X_standardized = X_scaler.fit_transform(exp_X).astype(np.float32)
    y_scaled = y_scaler.fit_transform(y.reshape(-1, 1)).flatten().astype(np.float32)

    return exp_X_standardized, y_scaled, sample_ids, exp_feature_names


def load_snp_data(phenotype):
    """
    加载SNP数据，处理缺失值和异常值

    Returns:
        snp_X_standardized (np.array): 标准化的SNP数据
        y_scaled (np.array): 标准化的表型数据
        sample_ids (list): 样本ID列表
        snp_feature_names (list): SNP特征名称列表
    """
    # 加载表型数据并过滤缺失值
    phenotypes_df = pd.read_csv(PHENOTYPES_FILE)
    phenotypes_df = phenotypes_df[phenotypes_df[phenotype].notna()]

    # 直接加载SNP数据
    snp_df = pd.read_csv(SNP_FILE, sep=' ')

    # 删除非数值列
    non_numeric_columns = ['FID', 'PAT', 'MAT', 'SEX', 'PHENOTYPE']
    snp_df = snp_df.drop(columns=non_numeric_columns)

    # 获取特征名称（去掉ID列）
    snp_feature_names = list(snp_df.columns.drop('ID'))

    # 保留同时存在表型和SNP数据的样本
    snp_df = snp_df[snp_df['ID'].isin(phenotypes_df['ID'])]

    # 处理数据中的异常值
    numeric_columns = snp_df.select_dtypes(include=[np.number]).columns

    # 替换无穷大值
    snp_df[numeric_columns] = snp_df[numeric_columns].replace([np.inf, -np.inf], np.nan)

    # 移除全为NaN的列
    snp_df = snp_df.dropna(axis=1, how='all')

    # 使用列均值填充剩余的NaN
    snp_df[numeric_columns] = snp_df[numeric_columns].fillna(snp_df[numeric_columns].mean())

    # 提取样本ID和特征
    sample_ids = snp_df['ID'].tolist()
    snp_X = snp_df.drop(['ID'], axis=1).astype(np.float32).values

    # 获取表型值
    y = np.array([phenotypes_df.loc[phenotypes_df['ID'] == item, phenotype].values[0] for item in sample_ids])

    # 处理表型数据中的异常值
    y = np.nan_to_num(y, nan=np.nanmean(y))

    # 标准化
    X_scaler = StandardScaler()
    y_scaler = StandardScaler()

    snp_X_standardized = X_scaler.fit_transform(snp_X).astype(np.float32)
    y_scaled = y_scaler.fit_transform(y.reshape(-1, 1)).flatten().astype(np.float32)

    return snp_X_standardized, y_scaled, sample_ids, snp_feature_names


def load_combined_data(phenotype):
    """
    加载组合数据，处理缺失值和异常值

    Returns:
        exp_X_standardized (np.array): 标准化的表达数据
        snp_X_standardized (np.array): 标准化的SNP数据
        y_scaled (np.array): 标准化的表型数据
        sample_ids (list): 样本ID列表
        exp_feature_names (list): 表达特征名称列表
        snp_feature_names (list): SNP特征名称列表
    """
    # 加载表型数据并过滤缺失值
    phenotypes_df = pd.read_csv(PHENOTYPES_FILE)
    phenotypes_df = phenotypes_df[phenotypes_df[phenotype].notna()]

    # 加载基因表达和SNP数据
    exp_data = pd.read_csv(EXP_FILE)
    snp_data = pd.read_csv(SNP_FILE, sep='\t')

    # 删除非数值列
    snp_data = snp_data.drop(columns=['FID', 'PAT', 'MAT', 'SEX', 'PHENOTYPE'])

    # 获取特征名称
    exp_feature_names = list(exp_data.columns.drop('ID'))
    snp_feature_names = list(snp_data.columns.drop('ID'))

    # 找出共同的样本
    common_ids = set(exp_data['ID']) & set(snp_data['ID']) & set(phenotypes_df['ID'])

    if len(common_ids) == 0:
        raise ValueError("没有同时具有SNP和表达数据的样本")

        # 过滤数据
    exp_data = exp_data[exp_data['ID'].isin(common_ids)]
    snp_data = snp_data[snp_data['ID'].isin(common_ids)]
    phenotypes_df = phenotypes_df[phenotypes_df['ID'].isin(common_ids)]

    # 处理异常值
    exp_numeric_cols = exp_data.select_dtypes(include=[np.number]).columns.drop('ID')
    snp_numeric_cols = snp_data.select_dtypes(include=[np.number]).columns.drop('ID')

    # 替换无穷大值并填充NaN
    exp_data[exp_numeric_cols] = exp_data[exp_numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(
        exp_data[exp_numeric_cols].mean())
    snp_data[snp_numeric_cols] = snp_data[snp_numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(
        snp_data[snp_numeric_cols].mean())

    # 提取数据
    sample_ids = list(common_ids)
    exp_X = exp_data.set_index('ID').loc[sample_ids].drop(columns=['ID']).values
    snp_X = snp_data.set_index('ID').loc[sample_ids].drop(columns=['ID']).values

    # 获取表型值
    y = np.array([phenotypes_df.loc[phenotypes_df['ID'] == item, phenotype].values[0] for item in sample_ids])

    # 处理表型数据中的异常值
    y = np.nan_to_num(y, nan=np.nanmean(y))

    # 标准化
    exp_scaler = StandardScaler()
    snp_scaler = StandardScaler()
    y_scaler = StandardScaler()

    exp_X_standardized = exp_scaler.fit_transform(exp_X).astype(np.float32)
    snp_X_standardized = snp_scaler.fit_transform(snp_X).astype(np.float32)
    y_scaled = y_scaler.fit_transform(y.reshape(-1, 1)).flatten().astype(np.float32)

    return exp_X_standardized, snp_X_standardized, y_scaled, sample_ids, exp_feature_names, snp_feature_names