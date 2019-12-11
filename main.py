# %%
import pprint
import plt_aucks
from pandas.core.dtypes.common import is_categorical_dtype
from sklearn.metrics import roc_curve, auc
from sklearn.feature_selection import RFE
import time
import constant
import numpy as np
import lightgbm as lgb
from lightgbm.sklearn import LGBMClassifier, LGBMRegressor
import bairong
import functions
import pandas as pd
from hyperopt import fmin, tpe, hp, Trials
import csv
import matplotlib.pyplot as plt
import ast
from tqdm import tqdm

out_file = "./output/bayes_output"
max_evals = 200  # max_evals迭代次数越大越慢，可设置合理的值

fixed_params = {
    "importance_type": "gain",  # result contains total gains of splits which use the feature.
    "class_weight": None,  # 类别型变量权重
    # 'n_estimators': 500,
    "boost_from_average": True,
    "boosting_type": "gbdt",  # 提升树的类型 boost/boosting
    "objective": "binary",
    "subsample": 0.8,  # 数据采样比例  bagging_fraction
    "colsample_bytree": 0.8,  # 每棵树特征选取比例 sub_feature/feature_fraction
    "learning_rate": 0.1,  # 学习率
    "num_leaves": 8,  # 最大叶数小于2^max_depth
    "max_depth": 3,  # 最大层数
    "min_child_weight": 0.02,
    "min_split_gain": 0,
    "random_state": 555,
    "reg_alpha": 0.0,
    "reg_lambda": 0.0,
    "subsample_for_bin": 200000,
    "subsample_freq": 0,
    # "scale_pos_weight ": scale_pos_weight,
    'save_binary ': True
}

Y = pd.read_pickle("./temp/Y_rfe.pkl")
X = pd.read_pickle("./temp/X_rfe_train_impt.pkl")

Data = pd.read_pickle("./temp/Data.pkl")
Y_test = pd.read_pickle("./temp/Y_test.pkl")
Y_train = pd.read_pickle("./temp/Y_train.pkl")
X_test = pd.read_pickle("./temp/X_test.pkl")
X_train = pd.read_pickle("./temp/X_train.pkl")

# %%
"""
1. 读取数据
"""
data_path = r"C:\Users\imhgy\Desktop\360循环贷_新总去中文加Y.csv"
_, header_detail = functions.get_header(data_path, verbose=True)
read_options = {
    "data": data_path,
    "header": 0,
    "index_col": "cus_num",
    # 'skiprows': [1],
    "skip_cols": constant.NONSENSE_COLS,
}
Data = functions.read_from(**read_options)

"""
2. 数据清洗和划分
"""

Data_Y = Data["Y"]
del Data["Y"]
Data_Y.to_pickle("./temp/Data_Y.pkl")
Data.to_pickle("./temp/Data.pkl")

# %%
Data_Y = pd.read_pickle("./temp/Data_Y.pkl")
Data = pd.read_pickle("./temp/Data.pkl")
Data.info()

# 判断样本bad_rate是否低于0.05，若低于则设置scale_pos_weight,用于处理不平衡样本,在lgbm训练中使用
if Data_Y.mean() < 0.05:
    scale_pos_weight = 0.05 * len(Data_Y) / len(np.where(Data_Y == 1)[0])
else:
    scale_pos_weight = 1.0

# 剔除缺失值与同值较高的、类别较多的、转换数据类型，节省内存
Data_X = functions.del_nan(Data, nan_ratio_threshold=0.95)
Data_X = functions.del_mode(Data_X, mode_ratio_threshold=0.95)
Data_X = functions.slim(Data_X)
Data_Y = functions.slim(Data_Y)
Data_X = functions.del_cat(Data_X, cat_threshold=10)
cat_features = [f for f in Data_X.columns if is_categorical_dtype(Data_X[f])]
Data_X_dummied = functions.get_dummied(Data_X)
Data_X.info()
Data_X_dummied.info()

# 数据划分
X_train, X_test, Y_train, Y_test = functions.split_train_test(Data_X, Data_Y, test_size=0.3, random_state=555)
X_train.to_pickle("./temp/X_train.pkl")
X_test.to_pickle("./temp/X_test.pkl")
Y_train.to_pickle("./temp/Y_train.pkl")
Y_test.to_pickle("./temp/Y_test.pkl")

Data_X_dummied_train = Data_X_dummied.loc[X_train.index.tolist(), :]
# %%
"""
3.以flag不全为0的样本进行模型训练
"""
Y_test = pd.read_pickle("./temp/Y_test.pkl")
Y_train = pd.read_pickle("./temp/Y_train.pkl")
X_test = pd.read_pickle("./temp/X_test.pkl")
X_train = pd.read_pickle("./temp/X_train.pkl")

# 获取全部flag
flags = bairong.get_flags(Data_X.columns.tolist())
# 获取flag不全为0的样本index
hit_indices = bairong.get_hit_indices(Data, flags)

# 以flag不全为0的样本进行模型训练
X_train_hit = X_train.loc[X_train.index.isin(hit_indices)]
Y_train_hit = Y_train.loc[Y_train.index.isin(hit_indices)]
Data_X_dummied_train = Data_X_dummied_train.loc[Data_X_dummied_train.index.isin(hit_indices)]
Y_train_hit.to_pickle("./temp/Y_train_hit.pkl")
X_train_hit.to_pickle("./temp/X_train_hit.pkl")

# %%

Y_train_hit = pd.read_pickle("./temp//Y_train_hit.pkl")
X_train_hit = pd.read_pickle("./temp//X_train_hit.pkl")
starttime = time.time()
lgbm_train = lgb.Dataset(X_train_hit, Y_train_hit, silent=True)
cv_results = lgb.cv(
    fixed_params,
    lgbm_train,
    num_boost_round=10000,  # n_estimators
    early_stopping_rounds=100,
    # Early stopping is an effective method for choosing the number of estimators
    # rather than setting this as another hyperparameter that needs to be tuned!
    nfold=10,
    seed=7,
    metrics="auc",  # Evaluation metrics to be monitored while CV.
    verbose_eval=False,
)
print("cv time consuming:", time.time() - starttime)
print(
    "The ideal number of iterations was {}.".format(
        np.argmax(cv_results["auc-mean"]) + 1
    )
)

# Extract the Highest score
results_best_score = max(cv_results["auc-mean"])
# Standard deviation of best score
results_best_std = cv_results["auc-stdv"][np.argmax(cv_results["auc-mean"])]
print(
    "The maximium ROC AUC on the validation set was {:.5f} with std of {:.5f}.".format(
        results_best_score, results_best_std
    )
)

# %%

"""
挑选特征
"""
starttime = time.time()
fixed_params["n_estimators"] = np.argmax(cv_results["auc-mean"]) + 1
lgbm_model = LGBMClassifier(**fixed_params)
lgbm_model.fit(X_train_hit, Y_train_hit, eval_metric="auc")
origin_features = X_train_hit.columns
lgbm_importance = lgbm_model.feature_importances_
lgbm_importance_gain = lgbm_model.booster_.feature_importance(importance_type="gain")
lgbm_importance_split = lgbm_model.booster_.feature_importance(importance_type="split")

# 保存特征重要性
chosen_feature_df = pd.DataFrame(origin_features)
chosen_feature_df["lgbm_importance"] = lgbm_importance
chosen_feature_df = chosen_feature_df.loc[chosen_feature_df.lgbm_importance > 0]
chosen_feature_df.to_excel("./output/chosen_feature_importance.xlsx")

chosen_feature = functions.get_sorted_feature(
    origin_features, lgbm_importance, threshold=0.0
)
chosen_cat_features = [f for f in chosen_feature if f in cat_features]

X_train_impt = X_train_hit[chosen_feature]

dummied_feature = [f for f in chosen_feature if f not in cat_features]
Data_X_dummied_train_impt = Data_X_dummied_train[dummied_feature]
print("time consuming:", time.time() - starttime)

# %%
"""
使用rfe对初筛的特征重新排序，选取排名前100/150（可修改）的特征
"""
starttime = time.time()

# Recursive Feature Elimination(RFE)

lgbm_model = LGBMClassifier(**fixed_params)
# create the RFE model and select 50 attributes
rfe = RFE(lgbm_model, n_features_to_select=50, step=1, verbose=1)
rfe.fit(Data_X_dummied_train_impt, Y_train_hit)

# summarize the selection of the attributes
print(rfe.support_)

rfe_rank = rfe.ranking_
rfe_lgb_df = pd.DataFrame(rfe_rank, index=dummied_feature)
rfe_lgb_df = rfe_lgb_df.rename(columns={0: "rank_lgb_rfe"})
rfe_lgb_df.to_excel("./output/rfe_lgb_df.xlsx")

lgb_rfe_features = rfe_lgb_df.loc[
    rfe_lgb_df.rank_lgb_rfe <= 101
    ].index.tolist()  # 这里取了前150个特征
X_rfe = X_train[lgb_rfe_features + chosen_cat_features]

# 获取rfe选出的特征的flag，获取当前特征对应flag不全为0的index，更新hit_indices
flags = bairong.get_flags(lgb_rfe_features + chosen_cat_features)
Data = pd.read_pickle("./temp/Data.pkl")
hit_indices = bairong.get_hit_indices(Data, flags)

# 以flag不全为0的样本进行模型训练
X_rfe = X_rfe.loc[X_rfe.index.isin(hit_indices)]
Y_rfe = Y_train.loc[Y_train.index.isin(hit_indices)]
X_rfe.to_pickle("./temp//X_rfe.pkl")
Y_rfe.to_pickle("./temp/Y_rfe.pkl")

# %% 用lightgbm重新排序
fixed_params["n_estimators"] = np.argmax(cv_results["auc-mean"]) + 1
lgbm_model = LGBMClassifier(**fixed_params)
lgbm_model.fit(X_rfe, Y_rfe, eval_metric="auc")
origin_features = X_rfe.columns
lgbm_importance = lgbm_model.feature_importances_
lgbm_importance_gain = lgbm_model.booster_.feature_importance(importance_type="gain")
lgbm_importance_split = lgbm_model.booster_.feature_importance(importance_type="split")

# 保存特征重要性
chosen_feature_df = pd.DataFrame(origin_features)
chosen_feature_df["lgbm_rfe_importance"] = lgbm_importance
chosen_feature_df = chosen_feature_df.loc[chosen_feature_df.lgbm_rfe_importance > 0]
chosen_feature_df.to_excel("./output/chosen_feature_rfe_importance.xlsx")

chosen_feature = functions.get_sorted_feature(
    origin_features, lgbm_importance, threshold=0.0
)
X_rfe_train_impt = X_rfe[chosen_feature]
X_rfe_train_impt.to_pickle("./temp/X_rfe_train_impt.pkl")

print("time consuming:", time.time() - starttime)

# %% 贝叶斯调参
Y = pd.read_pickle("./temp/Y_rfe.pkl")
X = pd.read_pickle("./temp/X_rfe_train_impt.pkl")

# File to save first results
of_connection = open("{}.csv".format(out_file), "w")
writer = csv.writer(of_connection)
# Write the headers to the file
writer.writerow(["loss", "params", "estimators", "train_time", "i"])
of_connection.close()

fixed_params_bay = fixed_params.copy()
# ks_score = kfold_objective(X, Y, model='lgb', params=fixed_params_bay)
ks_score = functions.ks_kfold_objective(
    X, Y, model="lgb", params=fixed_params_bay, out_file=out_file
)

para_space_mlp = {
    "class_weight": hp.choice("class_weight", [None, "balanced"]),
    "boosting_type": hp.choice(
        "boosting_type",
        [
            {
                "boosting_type": "gbdt",
                "subsample": hp.uniform("gdbt_subsample", 0.5, 1),
            },
            {"boosting_type": "goss", "subsample": 1.0},
        ],
    ),
    # "subsample": hp.uniform("subsample", 0.5, 1),
    "learning_rate": hp.loguniform("learning_rate", np.log(0.01), np.log(0.2)),
    "num_leaves": hp.quniform("num_leaves", 2, 32, 1),
    "subsample_for_bin": hp.quniform("subsample_for_bin", 20000, 300000, 20000),
    "feature_num": hp.quniform("feature_num", 100, 150, 1),  # 步长可修改
    "max_depth": hp.quniform("max_depth", 3, 5, 1),
    "min_child_weight": hp.quniform("min_child_weight", 0, 100, 5),  # 该参数的值域适当大些
    "min_child_samples": hp.quniform("min_child_samples", 1000, 3000, 50),  # 该参数与样本量相关
    "min_split_gain": hp.quniform("min_split_gain", 1, 10, 0.1),
    "colsample_bytree": hp.quniform("colsample_bytree", 0.6, 1, 0.1),
    "reg_alpha": hp.uniform("reg_alpha", 1, 10),
    "reg_lambda": hp.uniform("reg_lambda", 1, 10),
}
# 进行贝叶斯调参
trials = Trials()

starttime = time.time()
best = fmin(
    ks_score,
    para_space_mlp,
    algo=tpe.suggest,
    max_evals=max_evals,
    rstate=np.random.RandomState(555),
    trials=trials,
)
# 对贝叶斯调参后的所有参数，拟合计算训练测试ks、auc，寻找出效果最好且相差最小的那组参数
trials_result = trials.trials
print("贝叶斯调参结束", time.time() - starttime)

# %% 根据贝叶斯每一次迭代的参数，对训练集、测试集的ks、auc作图
ks_trains = []
ks_tests = []
auc_trains = []
auc_tests = []

bayesresults = pd.read_csv("{}.csv".format(out_file))
# Sort with best scores on top and reset index for slicing
bayesresults = bayesresults[["i", "loss", "params", "estimators", "train_time"]]
bayesresults["i"] = bayesresults.index

for i in tqdm(range(max_evals)):
    params_1 = ast.literal_eval(bayesresults.loc[i, "params"])  # 将csv中的参数转为字典
    fixed_params.update(params_1)

    # 提取出命中前feature_num个特征所在flag的index
    feature_num = int(fixed_params.pop("feature_num"))
    # chosen_feature_1 = lgb_rfe_features[:feature_num]
    print(feature_num)
    chosen_feature_1 = X.columns[:feature_num]
    flags_1 = bairong.get_flags(chosen_feature_1)
    hit_indices_1 = bairong.get_hit_indices(Data, flags_1)
    X_tr, X_te = X_train[chosen_feature_1], X_test[chosen_feature_1]  # 选择特征
    X_tr = X_tr.loc[X_tr.index.isin(hit_indices_1)]  # 选择index
    Y_tr = Y.loc[Y.index.isin(hit_indices_1)]  # 选择index
    X_te = X_te.loc[X_te.index.isin(hit_indices_1)]  # 选择index
    Y_te = Y_test.loc[Y_test.index.isin(hit_indices_1)]  # 选择index
    pprint.pprint(fixed_params)
    lgbm_tuner = functions.LGBModelTuner(
        LGBMClassifier(**fixed_params), X_tr, Y_tr, X_te, Y_te, hit_indices_1
    )
    result_ks = lgbm_tuner.get_model_result(fixed_params)
    train_ks = result_ks["ks"][0]
    test_ks = result_ks["ks"][1]
    train_auc = result_ks["auc"][0]
    test_auc = result_ks["auc"][1]
    ks_trains.append(train_ks)
    ks_tests.append(test_ks)
    auc_trains.append(train_auc)
    auc_tests.append(test_auc)

# ks_auc_df=pd.DataFrame({
#     "ks_trains"  : ks_trains,
#     "ks_tests"    : ks_tests,
#     "auc_trains": auc_trains,
#     "auc_tests"  : auc_tests,})


bayesresults["ks_trains"] = ks_trains
bayesresults["ks_tests"] = ks_tests
bayesresults["auc_trains"] = auc_trains
bayesresults["auc_tests"] = auc_tests
ks_auc_df = bayesresults
ks_auc_df.to_pickle("./output/ks_auc_df.pkl")

# %% 打印全部的迭代数据点ks auc
ks_auc_df = pd.read_pickle("./output/ks_auc_df.pkl")
plt_aucks.plot_ks_auc(ks_auc_df, './output/ks_auc_df.html')


# %% 选择贝叶斯最优参数
ks_auc_df = pd.read_pickle("output/ks_auc_df.pkl")
# Sort with best scores on top and reset index for slicing
ks_tr = ks_auc_df.sort_values("auc_trains", ascending=False).reset_index(drop=True)
ks_te = ks_auc_df.sort_values("auc_tests", ascending=False).reset_index(drop=True)
auc_tr = ks_auc_df.sort_values("ks_trains", ascending=False).reset_index(drop=True)
auc_te = ks_auc_df.sort_values("ks_tests", ascending=False).reset_index(drop=True)
auc_dif = pd.Series((ks_auc_df.auc_trains-ks_auc_df.auc_tests).sort_values(ascending=True).iloc[0:50].index)
ks_dif = pd.Series((ks_auc_df.ks_trains-ks_auc_df.ks_tests).sort_values(ascending=True).iloc[0:50].index)
top_idx = pd.Series()
dif_idx = pd.Series()
chose_idx = pd.Series()

for i in  ks_te,  auc_te:
    top_idx = pd.concat([top_idx, i.loc[0:20, 'i'].copy()],ignore_index=True)
    chose_idx = pd.concat([chose_idx, i.loc[0:20, 'i'].copy()],ignore_index=True)
for i in auc_dif,ks_dif:
    dif_idx = pd.concat([dif_idx, i],ignore_index=True)
    chose_idx = pd.concat([chose_idx, i],ignore_index=True)
top_idx = pd.DataFrame(top_idx,columns=['i'])
dif_idx = pd.DataFrame(dif_idx,columns=['i'])
chose_idx = pd.DataFrame(chose_idx,columns=['i'])
top_idx = top_idx.groupby(by=['i']).size().sort_values(ascending=False)
dif_idx = dif_idx.groupby(by=['i']).size().sort_values(ascending=False)
chose_idx = chose_idx.groupby(by=['i']).size().sort_values(ascending=False)
# %% 手动调参
ks_auc_df = pd.read_pickle("output/ks_auc_df.pkl")

params_193 = ast.literal_eval(ks_auc_df.loc[193, "params"])  # 将csv中的参数转为字典


fixed_params.update(params_193)
feature_num = fixed_params.get("feature_num")
chosen_final_feature = X.columns[:feature_num]  # 根据贝叶斯确定最终特征个数
X_train = X_train[chosen_final_feature]
X_test = X_test[chosen_final_feature]
flags = bairong.get_flags(chosen_final_feature)
hit_indices = bairong.get_hit_indices(Data, flags)
X_train = X_train.loc[X_train.index.isin(hit_indices)]
Y_train = Y_train.loc[Y_train.index.isin(hit_indices)]
X_test = X_test.loc[X_test.index.isin(hit_indices)]
Y_test = Y_test.loc[Y_test.index.isin(hit_indices)]

"""
手动调参
"""
"""
ks_trains     0.377806
ks_tests       0.35709
auc_trains     0.75572
auc_tests     0.739925
"""
lgbm_tuner = functions.LGBModelTuner(
    LGBMClassifier(**fixed_params), X_train, Y_train, X_test, Y_test, hit_indices
)
result_ks = lgbm_tuner.get_model_result(fixed_params)

print(lgbm_tuner.estimator)
print(ks_auc_df.loc[193, "ks_trains":])

# 调max_depth
# max_depth是提高精确度的最重要的参数，一般会选取3-5
max_depth_ks_auc_df = lgbm_tuner.try_tune("max_depth", [1, 2, 3, 4, 5, 6, 7])
plt_aucks.plot_ks_auc(max_depth_ks_auc_df, 'output/depth.html')
lgbm_tuner.tune("max_depth", 6)

# 调num_leaves
print(lgbm_tuner.estimator)
# 大致换算关系：num_leaves = 2^(max_depth)，但是它的值的设置应该小于 2^(max_depth)，否则可能会导致过拟合。
num_leaves_space = [i for i in range(5, 32)]
num_leaves_ks_auc_df = lgbm_tuner.try_tune("num_leaves", num_leaves_space)
plt_aucks.plot_ks_auc(num_leaves_ks_auc_df, 'output/num_leaves.html')
lgbm_tuner.tune("num_leaves", 11)

# 调完精度的参数，开始调控制过拟合的参数 调min_child_weight/min_sum_hessian_in_leaf
# min_sum_hessian_in_leaf：也叫min_child_weight,
# 使一个结点分裂的最小海森值之和（Minimum sum of hessians in one leaf to allow a split.
# Higher values potentially decrease overfitting）,可以很好的控制过拟合。
# min_child_weight与min_sum_hessian_in_leaf只需要调节一个就好，两者是同一个东西。
# 先粗调后细调，参数范围不要思维定式
min_child_weight_space = np.arange(1, 210, 10)
min_child_weight_space = [0.001, 0.1, 1]
min_child_weight_ks_auc_df = lgbm_tuner.try_tune("min_child_weight", min_child_weight_space)
functions.models_ks(
    "min_child_weight",
    min_child_weight_space,
    min_child_weight_ks_auc_df["ks_trains"],
    min_child_weight_ks_auc_df["ks_tests"],
)
functions.models_auc(
    "min_child_weight",
    min_child_weight_space,
    min_child_weight_ks_auc_df["auc_trains"],
    min_child_weight_ks_auc_df["auc_tests"],
)
lgbm_tuner.tune("min_child_weight", 1)

# 调min_data_in_leaf/min_child_samples 这两者是同一个参数，只需要调节一个就好
# min_data_in_leaf 是一个很重要的参数, 也叫min_child_samples，它的值取决于训练数据的样本个树 和num_leaves.
# 将其设置的较大可以避免生成一个过深的树, 但有可能导致欠拟合。可设置取值范围在 样本数/num_leaves附近调节。
print(lgbm_tuner.estimator)
min_child_samples_space = np.arange(1500, 2500, 100)
min_child_samples_space = [100, 500, 1000, 1500, 2000, 2500, 3000, 3500]
min_child_samples_ks_auc_df = lgbm_tuner.try_tune(
    "min_child_samples", min_child_samples_space
)
min_child_samples_ks_auc_df['value'] = min_child_samples_space

functions.models_ks(
    "min_child_samples",
    min_child_samples_space,
    min_child_samples_ks_auc_df["ks_trains"],
    min_child_samples_ks_auc_df["ks_tests"],
)
functions.models_auc(
    "min_child_samples",
    min_child_samples_space,
    min_child_samples_ks_auc_df["auc_trains"],
    min_child_samples_ks_auc_df["auc_tests"],
)

lgbm_tuner.tune("min_child_samples", 2100)

# 调正则化参数lambda_l1:reg_alpha
# 毫无疑问，是降低过拟合的， 参数越大正则化越强
print(lgbm_tuner.estimator)
reg_alpha_space = np.arange(2.5, 4.5, 0.1)
reg_alpha_ks_auc_df = lgbm_tuner.try_tune("reg_alpha", reg_alpha_space)
reg_alpha_ks_auc_df['value'] = reg_alpha_space
reg_alpha_ks_auc_df['ks_diff'] = reg_alpha_ks_auc_df['ks_trains'] - reg_alpha_ks_auc_df['ks_tests']
reg_alpha_ks_auc_df['auc_diff'] = reg_alpha_ks_auc_df['auc_trains'] - reg_alpha_ks_auc_df['auc_tests']

functions.models_ks(
    "reg_alpha",
    reg_alpha_space,
    reg_alpha_ks_auc_df["ks_trains"],
    reg_alpha_ks_auc_df["ks_tests"],
)
functions.models_auc(
    "reg_alpha",
    reg_alpha_space,
    reg_alpha_ks_auc_df["auc_trains"],
    reg_alpha_ks_auc_df["auc_tests"],
)
lgbm_tuner.tune("reg_alpha", 3.1)

# 正则化参数lambda_l1:reg_lambda
reg_lambda_space = [i for i in np.arange(2.6, 4.5, 0.1)]
reg_lambda_ks_auc_df = lgbm_tuner.try_tune("reg_lambda", reg_lambda_space)
reg_lambda_ks_auc_df['value'] = reg_lambda_space
reg_lambda_ks_auc_df['ks_diff'] = reg_lambda_ks_auc_df['ks_trains'] - reg_lambda_ks_auc_df['ks_tests']
reg_lambda_ks_auc_df['auc_diff'] = reg_lambda_ks_auc_df['auc_trains'] - reg_lambda_ks_auc_df['auc_tests']

functions.models_ks(
    "reg_lambda",
    reg_lambda_space,
    reg_lambda_ks_auc_df["ks_trains"],
    reg_lambda_ks_auc_df["ks_tests"],
)
functions.models_auc(
    "reg_lambda",
    reg_lambda_space,
    reg_lambda_ks_auc_df["auc_trains"],
    reg_lambda_ks_auc_df["auc_tests"],
)
lgbm_tuner.tune("reg_lambda", 3.2)

# 调min_split_gain
# 执行切分的最小增益，也是控制过拟合的参数，值越大，对过拟合控制越明显，设置过大会欠拟合
print(lgbm_tuner.estimator)
min_split_gain_space = [i for i in np.arange(0, 2.1, 0.1)]
min_split_gain_ks_auc_df = lgbm_tuner.try_tune("min_split_gain", min_split_gain_space)
min_split_gain_ks_auc_df['value'] = min_split_gain_space
min_split_gain_ks_auc_df['ks_diff'] = min_split_gain_ks_auc_df['ks_trains'] - min_split_gain_ks_auc_df['ks_tests']
min_split_gain_ks_auc_df['auc_diff'] = min_split_gain_ks_auc_df['auc_trains'] - min_split_gain_ks_auc_df['auc_tests']

functions.models_ks(
    "min_split_gain",
    min_split_gain_space,
    min_split_gain_ks_auc_df["ks_trains"],
    min_split_gain_ks_auc_df["ks_tests"],
)
functions.models_auc(
    "min_split_gain",
    min_split_gain_space,
    min_split_gain_ks_auc_df["auc_trains"],
    min_split_gain_ks_auc_df["auc_tests"],
)
lgbm_tuner.tune("min_split_gain", 0.9)

# 通过设置 feature_fraction/colsample_bytree 参数来使用特征的子抽样
colsample_bytree_space = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
colsample_bytree_ks_auc_df = lgbm_tuner.try_tune("colsample_bytree", colsample_bytree_space)
lgbm_tuner.tune("colsample_bytree", 0.8)

# 通过设置 bagging_fraction/subsample 参数来使用特征的子抽样
subsample_space = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
subsample_ks_auc_df = lgbm_tuner.try_tune("subsample", subsample_space)
lgbm_tuner.tune("subsample", 0.8)

# 调learning_rate
# 在调完所有的参数后，再来对learning_rate进行调节，
learning_rate_space = [i for i in np.arange(0.01, 0.15, .005)]
learning_rate_ks_auc_df = lgbm_tuner.try_tune("learning_rate", learning_rate_space)
learning_rate_ks_auc_df['value'] = learning_rate_space
learning_rate_ks_auc_df.sort_values("auc_tests", ascending=False, inplace=True)
learning_rate_ks_auc_df.sort_values("ks_tests", ascending=False, inplace=True)

functions.models_ks(
    "learning_rate",
    learning_rate_space,
    learning_rate_ks_auc_df["ks_trains"],
    learning_rate_ks_auc_df["ks_tests"],
)
functions.models_auc(
    "learning_rate",
    learning_rate_space,
    learning_rate_ks_auc_df["auc_trains"],
    learning_rate_ks_auc_df["auc_tests"],
)
lgbm_tuner.tune("learning_rate", 0.0048)

df_train_auc, df_test_auc = functions.tune_n_estimators_learning_rate(
    lgbm_tuner,
    learning_rate=[0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1],
    n_estimators=[i for i in range(400, 800, 20)],
)
functions.plot_n_estimators_learning_rate(df_train_auc, df_test_auc)
lgbm_tuner.tune("n_estimators", 540)
lgbm_tuner.tune("learning_rate", 0.03)

params = lgbm_tuner.params
lgbm_tuner.get_model_result(params)

# 使用 save_binary 在未来的学习过程对数据加载进行加速
# max_bin       bagging_freq  subsample_freq


# 调好的参数对特征再次排序
lgbm_model = LGBMClassifier(**params)
rfe = RFE(lgbm_model, n_features_to_select=1, step=1, verbose=1)  # 终止时特征数1个
rfe.fit(X, Y)
rfe_rank_2 = rfe.ranking_
rfe_lgb_df_2 = pd.DataFrame(rfe_rank_2, index=chosen_final_feature)
rfe_lgb_df_2 = rfe_lgb_df_2.rename(columns={0: 'rank_lgb_rfe'})

# 查看特征数对模型效果的影响
feature_num_space, feature_num_ks_auc_df = functions.tune_feature_num(params, X, Y, X_test, Y_test, rfe_lgb_df_2,
                                                                      hit_indices, min_feature_num=5, step=1)
i, j = feature_num_space.index(100), feature_num_space.index(136)
functions.models_ks('feature_num', feature_num_space[i:], feature_num_ks_auc_df.loc[i:, ['ks_trains']]['ks_trains'],
                    feature_num_ks_auc_df.loc[i:, ['ks_tests']]['ks_tests'])
functions.models_auc('feature_num', feature_num_space[i:], feature_num_ks_auc_df.loc[i:, ['auc_trains']],
                     feature_num_ks_auc_df.loc[i:, ['auc_tests']])

# 选择最终的特征数
chosen_final_feature_rfe = rfe_lgb_df_2.loc[rfe_lgb_df_2.rank_lgb_rfe <= 135].index.tolist()
lgbm_tuner = functions.LGBModelTuner(LGBMClassifier(**params), X[chosen_final_feature_rfe], Y,
                                     X_test[chosen_final_feature_rfe],
                                     Y_test, hit_indices)
lgbm_tuner.get_model_result(params)
print(lgbm_tuner.estimator)

# 可根据最终的特征数再一次的调优

flags = bairong.get_flags(lgb_rfe_features)
hit_indices = bairong.get_hit_indices(Data, flags)

"""
使用调好的参数进行模型训练
"""
train = X.copy()
train_y = Y.copy()

lgbm_model = LGBMRegressor(boost_from_average=True, boosting_type='gbdt',
                           class_weight='balanced', colsample_bytree=0.8, feature_num=137,
                           importance_type='split', learning_rate=0.03, max_depth=5,
                           min_child_samples=2100, min_child_weight=70.0,
                           min_split_gain=0.9, n_estimators=540, n_jobs=-1, num_leaves=13,
                           objective='binary', random_state=555, reg_alpha=3.1,
                           reg_lambda=3.2, silent=True, subsample=0.9695126917137413,
                           subsample_for_bin=220000, subsample_freq=0)

X_use = train[chosen_final_feature_rfe]
lgbm_model.fit(X_use, train_y, eval_metric='auc')
preds_train = lgbm_model.predict(X_use)
ks_value, bad_percent, good_percent = cal_ks(-preds_train, train_y, section_num=20)
max_ks0 = np.max(ks_value)
false_positive_rate, recall, thresholds = roc_curve(train_y, preds_train)
roc_auc0 = auc(false_positive_rate, recall)
print('当前模型在样本内训练集的KS值和AUC值分别为{0}'.format([max_ks0, roc_auc0]))

plt_aucks.figure()
plt_aucks.hist(preds_train)
plt_aucks.ylabel('Number of samples')
plt_aucks.xlabel('probability of y=1')
plt_aucks.title('Probability Distribution on test samples')
plt_aucks.show()

plt_aucks.figure()
plt_aucks.plot(list(range(0, 21)), np.append([0], bad_percent), '-r', label='Bad Percent')
plt_aucks.plot(list(range(0, 21)), np.append([0], good_percent), '-g', label='Good Percent')
plt_aucks.plot(list(range(0, 21)), np.append([0], ks_value), '-b', label='Max KS value = %0.2f' % max_ks0)
plt_aucks.legend(loc='lower right')
plt_aucks.ylabel('% of total Good/Bad')
plt_aucks.xlabel('% of population')
plt_aucks.show()

plt_aucks.figure()
plt_aucks.title('Receiver Operating Characteristic')
plt_aucks.plot(false_positive_rate, recall, 'b', label='AUC = %0.2f' % roc_auc0)
plt_aucks.legend(loc='lower right')
plt_aucks.plot([0, 1], [0, 1], 'r--')
plt_aucks.xlim([0.0, 1.0])
plt_aucks.ylim([0.0, 1.0])
plt_aucks.ylabel('Recall')
plt_aucks.xlabel('Fall-out')
plt_aucks.show()

"""
使用调好的参数进行模型测试 
"""
valid = X_test.copy()
valid_y = Y_test.copy()
X_test_use = valid[chosen_final_feature_rfe]
lgbm_model.fit(X_use, train_y, eval_metric='auc')
preds_test = lgbm_model.predict(X_test_use)
ks_value, bad_percent, good_percent = cal_ks(-preds_test, valid_y, section_num=20)
max_ks0 = np.max(ks_value)
false_positive_rate, recall, thresholds = roc_curve(valid_y, preds_test)
roc_auc0 = auc(false_positive_rate, recall)
print('当前模型在样本内测试集的KS值和AUC值分别为{0}'.format([max_ks0, roc_auc0]))

plt_aucks.figure()
plt_aucks.hist(preds_test)
plt_aucks.ylabel('Number of samples')
plt_aucks.xlabel('probability of y=1')
plt_aucks.title('Probability Distribution on test samples')
plt_aucks.show()

plt_aucks.figure()
plt_aucks.plot(list(range(0, 21)), np.append([0], bad_percent), '-r', label='Bad Percent')
plt_aucks.plot(list(range(0, 21)), np.append([0], good_percent), '-g', label='Good Percent')
plt_aucks.plot(list(range(0, 21)), np.append([0], ks_value), '-b', label='Max KS value = %0.2f' % max_ks0)
plt_aucks.legend(loc='lower right')
plt_aucks.ylabel('% of total Good/Bad')
plt_aucks.xlabel('% of population')
plt_aucks.show()

plt_aucks.figure()
plt_aucks.title('Receiver Operating Characteristic')
plt_aucks.plot(false_positive_rate, recall, 'b', label='AUC = %0.2f' % roc_auc0)
plt_aucks.legend(loc='lower right')
plt_aucks.plot([0, 1], [0, 1], 'r--')
plt_aucks.xlim([0.0, 1.0])
plt_aucks.ylim([0.0, 1.0])
plt_aucks.ylabel('Recall')
plt_aucks.xlabel('Fall-out')
plt_aucks.show()

# 模型拉伸
basic_badrate = sum(Y) / len(Y)
point0 = 630
odds0 = basic_badrate / (1 - basic_badrate)
PDO = 72

B = PDO / np.log(2)
A = point0 + B * np.log(odds0)

point_train = A - B * np.log(preds_train / (1 - preds_train))
point_test = A - B * np.log(preds_test / (1 - preds_test))

# point_train[np.where(np.array(hit_indices) != 1)[0]] = np.NaN
# point_test[np.where(np.array(flag_test_all) != 1)[0]] = np.NaN
point_train[np.where(point_train > 1000)[0]] = 1000
point_train[np.where(point_train < 300)[0]] = 300
point_test[np.where(point_test > 1000)[0]] = 1000
point_test[np.where(point_test < 300)[0]] = 300

plt_aucks.figure()
plt_aucks.hist(point_train)
plt_aucks.ylabel('Number of samples')
plt_aucks.xlabel('Score')
plt_aucks.title('Distribution on train samples')

plt_aucks.figure()
plt_aucks.hist(point_test)
plt_aucks.ylabel('Number of samples')
plt_aucks.xlabel('Score')
plt_aucks.title('Distribution on test samples')
plt_aucks.show()

# 计算PSI
PSI_value = PSI(point_train, point_test)

from model_parser import LGBModelParser

p = LGBModelParser(lgbm_model, chosen_final_feature_rfe)
