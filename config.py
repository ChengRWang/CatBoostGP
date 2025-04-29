import os

# 路径配置
DATA_DIR = '/data1/wangchengrui/final_results/eqtl'
CROSS_VALIDATION_FILE = os.path.join(DATA_DIR, 'rice4k_cv_splits.json')
EXP_FILE = os.path.join(DATA_DIR, 'rice4k_exp.csv')
PHENOTYPES_FILE = os.path.join(DATA_DIR, 'rice4k_ph.csv')
SNP_FILE = os.path.join(DATA_DIR, 'vcf/rice4k_eQTL_GWAS/Heading_date.raw')
RESULTS_DIR = os.path.join(DATA_DIR, 'rice4k.eQTL_GWAS.results')

# 待分析的表型列表
PHENOTYPES = [
    'Heading_date', 'Plant_height', 'Num_panicles',
    'Num_effective_panicles', 'Yield', 'Grain_weight',
    'Spikelet_length', 'Grain_length', 'Grain_width',
    'Grain_thickness'
]

# CatBoost 模型参数
CATBOOST_PARAMS = {
    "iterations": 1000,
    "learning_rate": 0.1,
    "depth": 6,
    "loss_function": "RMSE",
    "random_seed": 42,
    "verbose": 100,
    "early_stopping_rounds": 50,
    "use_best_model": True
}

# GPU 设置
USE_GPU = True

# 是否计算 SHAP 值
SHAP_COMPUTE = True