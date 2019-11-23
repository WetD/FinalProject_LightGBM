# %%
import pickle
from sklearn.feature_selection import RFE
import time
import constant
import numpy as np
import lightgbm as lgb
from lightgbm.sklearn import LGBMClassifier
import bairong
import functions
import pandas as pd
from hyperopt import fmin, tpe, hp, Trials
import csv
import matplotlib.pyplot as plt
import ast
from tqdm import tqdm

fixed_params = {
    # "importance_type": "gain",  # result contains total gains of splits which use the feature.
    "class_weight": None,  # 类别型变量权重
    # 'n_estimators': 500,
    "boost_from_average": True,
    "boosting_type": "gbdt",  # 提升树的类型 boost/boosting
    "objective": "binary",
    "subsample": 0.8,  # 数据采样比例  bagging_fraction
    "colsample_bytree": 0.8,  # 每棵树特征选取比例 sub_feature/feature_fraction
    "learning_rate": 0.04,  # 学习率
    "num_leaves": 8,  # 最大叶数小于2^max_depth
    "max_depth": 3,  # 最大层数
    "min_child_weight": 0.02,
    "min_split_gain": 0,
    "random_state": 7,
    "reg_alpha": 0.0,
    "reg_lambda": 0.0,
    "subsample_for_bin": 200000,
    "subsample_freq": 0,
    # "scale_pos_weight ": scale_pos_weight,
}

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

Y = Data["Y"]
del Data["Y"]
Y.mean()  # bad in total rate

Y.to_pickle("./Y.pkl")
Data.to_pickle("./Data.pkl")
# %%
Y = pd.read_pickle("./Y.pkl")
Data = pd.read_pickle("./Data.pkl")
Data.info()
# 判断样本bad_rate是否低于0.05，若低于则设置scale_pos_weight,用于处理不平衡样本,在lgbm训练中使用
if Y.mean() < 0.05:
    scale_pos_weight = 0.05 * len(Y) / len(np.where(Y == 1)[0])
else:
    scale_pos_weight = 1.0

# 剔除缺失值与同值较高的、类别较多的、转换数据类型，节省内存
X = functions.del_nan(Data, nan_ratio_threshold=0.95)
X = functions.del_mode(X, mode_ratio_threshold=0.95)
X = functions.slim(X)
Y = functions.slim(Y)
X = functions.del_cat(X, cat_threshold=10)
X = functions.get_dummied(X)

# 获取全部flag
flags = bairong.get_flags(X.columns.tolist())
# 获取flag不全为0的样本index
hit_indices = bairong.get_hit_indices(Data, flags)

# 数据划分
X, X_test, Y, Y_test = functions.split_train_test(X, Y, test_size=0.3, random_state=3)
X.to_pickle("./X.pkl")
X_test.to_pickle("./X_test.pkl")
Y.to_pickle("./Y.pkl")
Y_test.to_pickle("./Y_test.pkl")

Y_test = pd.read_pickle("./Y_test.pkl")
Y = pd.read_pickle("./Y.pkl")
X_test = pd.read_pickle("./X_test.pkl")
X = pd.read_pickle("./X.pkl")

# %%


plt.hist(Y, edgecolor="k")
plt.xlabel("Label")
plt.ylabel("Count")
plt.title("Counts of Labels")
plt.show()
# %%

"""
初步确定chosen_feature，通过cv确定n_estimators
"""
# 以flag不全为0的样本进行模型训练
X = X.loc[X.index.isin(hit_indices)]
Y = Y.loc[Y.index.isin(hit_indices)]

Y.to_pickle("./Y.pkl")
X.to_pickle("./X.pkl")
# %%

Y = pd.read_pickle("./Y.pkl")
X = pd.read_pickle("./X.pkl")

lgbm_train = lgb.Dataset(X, Y, silent=True)

starttime = time.time()

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
fig = plt.figure(figsize=(10, 6), dpi=100)
plt.title("Validation Curve with LightGBM (eta = 0.3)")
plt.xlabel("number of rounds")
plt.ylabel("AUC")
plt.ylim(0.6, 0.8)
plt.plot(
    np.linspace(1, len(cv_results["auc-mean"]), len(cv_results["auc-mean"])).astype(
        "int"
    ),
    cv_results["auc-mean"],
    label="Training score",
    color="r",
)
plt.fill_between(
    np.linspace(1, len(cv_results["auc-mean"]), len(cv_results["auc-mean"])).astype(
        "int"
    ),
    [
        cv_results["auc-mean"][i] - cv_results["auc-stdv"][i]
        for i in range(len(cv_results["auc-mean"]))
    ],
    [
        cv_results["auc-mean"][i] + cv_results["auc-stdv"][i]
        for i in range(len(cv_results["auc-mean"]))
    ],
    alpha=0.2,
    color="r",
)

plt.axhline(y=1, color="k", ls="dashed")  # Add a horizontal line across the axis.

plt.legend(loc="best")
plt.show()
# %%

"""
挑选特征
"""
starttime = time.time()

fixed_params["n_estimators"] = len(cv_results["auc-mean"])

lgbm_model = LGBMClassifier(**fixed_params)
lgbm_model.fit(X, Y, eval_metric="auc")
origin_features = X.columns
lgbm_importance = lgbm_model.feature_importances_
lgbm_importance_gain = lgbm_model.booster_.feature_importance(importance_type="gain")
lgbm_importance_split = lgbm_model.booster_.feature_importance(importance_type="split")

# 保存特征重要性
chosen_feature_df = pd.DataFrame(origin_features)
chosen_feature_df["lgbm_importance"] = lgbm_importance
chosen_feature_df = chosen_feature_df.loc[chosen_feature_df.lgbm_importance > 0]
chosen_feature_df.to_excel("chosen_feature_importance.xlsx")

chosen_feature = functions.get_sorted_feature(
    origin_features, lgbm_importance, threshold=0.0
)
X = X[chosen_feature]

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
rfe.fit(X, Y)

# summarize the selection of the attributes
print(rfe.support_)

rfe_rank = rfe.ranking_
rfe_lgb_df = pd.DataFrame(rfe_rank, index=chosen_feature)
rfe_lgb_df = rfe_lgb_df.rename(columns={0: "rank_lgb_rfe"})
rfe_lgb_df.to_excel("rfe_lgb_df.xlsx")

lgb_rfe_features = rfe_lgb_df.loc[
    rfe_lgb_df.rank_lgb_rfe <= 101
].index.tolist()  # 这里取了前150个特征
X = X[lgb_rfe_features]
# 获取rfe选出的特征的flag，获取当前特征对应flag不全为0的index，更新hit_indices
flags = bairong.get_flags(lgb_rfe_features)
Data = pd.read_pickle("./Data.pkl")
hit_indices = bairong.get_hit_indices(Data, flags)
# 以flag不全为0的样本进行模型训练
X = X.loc[X.index.isin(hit_indices)]
Y = Y.loc[Y.index.isin(hit_indices)]
Y.to_pickle("./Y_rfe.pkl")
X.to_pickle("./X_rfe.pkl")

print("time consuming:", time.time() - starttime)

# %% 贝叶斯调参
"""
贝叶斯调参
"""
Y = pd.read_pickle("./Y_rfe.pkl")
X = pd.read_pickle("./X_rfe.pkl")
out_file = "gbm_trials_100_ziyang1123.csv"

# File to save first results
of_connection = open(out_file, "w")
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
            {
                "boosting_type": "dart",
                "subsample": hp.uniform("dart_subsample", 0.5, 1),
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
max_evals = 5  # max_evals迭代次数越大越慢，可设置合理的值

starttime = time.time()
best = fmin(
    ks_score,
    para_space_mlp,
    algo=tpe.suggest,
    max_evals=max_evals,
    rstate=np.random.RandomState(7),
    trials=trials,
)
# 对贝叶斯调参后的所有参数，拟合计算训练测试ks、auc，寻找出效果最好且相差最小的那组参数
trials_result = trials.trials
print("贝叶斯调参结束", time.time() - starttime)
# %%
# Sort the trials with lowest loss (highest AUC) first
bayes_trials_results = sorted(trials.results, key=lambda x: x["loss"])
print(bayes_trials_results[:5])

# 保存贝叶斯参数迭代过程
with open("trials.pkl", "wb") as f:
    pickle.dump(trials, f)

with open("trials_ziyang_500.pkl", "rb") as f:
    load = pickle.load(f)

results = pd.read_csv("gbm_trials_500_ziyang.csv")
# Sort with best scores on top and reset index for slicing
results.sort_values("loss", ascending=True, inplace=True)
results.reset_index(inplace=True, drop=True)
results.head()

results = pd.read_csv("gbm_trials_2.csv")

# Convert from a string to a dictionary
best_params = ast.literal_eval(results.loc[0, "params"])

# 获取best参数所对应的迭代次数
for i in range(0, max_evals):
    param = trials_result[i]["misc"]["vals"]
    param_1 = {k: v[0] for k, v in param.items()}
    if param_1 == best:
        print(i)
# %%   根据贝叶斯每一次迭代的参数，对训练集、测试集的ks、auc作图


max_evals = 500  # max_evals迭代次数越大越慢，可设置合理的值
ks_trains = []
ks_tests = []
auc_trains = []
auc_tests = []
space = []
X_rfe = pd.read_pickle("./X_rfe.pkl")
Data = pd.read_pickle("./Data.pkl")
Y_test = pd.read_pickle("./Y_test.pkl")
Y = pd.read_pickle("./Y.pkl")
X_test = pd.read_pickle("./X_test.pkl")
X = pd.read_pickle("./X.pkl")
# with open("trials_ziyang_500.pkl", "rb") as f:
#     load = pickle.load(f)

results = pd.read_csv("gbm_trials_500_ziyang.csv")

# Convert from a string to a dictionary
# params_1 = ast.literal_eval(results.loc[:, "params"])

# for i in range(0, max_evals):
#     params_1 = ast.literal_eval(results.loc[i, "params"])
#     print(params_1["min_child_samples"])
#     print(params_1["class_weight"])

for i in tqdm(range(max_evals)):
    params_1 = ast.literal_eval(results.loc[i, "params"])  # 将csv中的参数转为字典
    fixed_params.update(params_1)

    # 提取出命中前feature_num个特征所在flag的index
    feature_num = int(fixed_params.pop("feature_num"))
    # chosen_feature_1 = lgb_rfe_features[:feature_num]
    chosen_feature_1 = X_rfe.columns[:feature_num]
    flags_1 = bairong.get_flags(chosen_feature_1)
    hit_indices_1 = bairong.get_hit_indices(Data, flags_1)
    X_tr, X_te = X[chosen_feature_1], X_test[chosen_feature_1]  # 选择特征
    X_tr = X_tr.loc[X_tr.index.isin(hit_indices_1)]  # 选择index
    Y_tr = Y.loc[Y.index.isin(hit_indices_1)]  # 选择index
    X_te = X_te.loc[X_te.index.isin(hit_indices_1)]  # 选择index
    Y_te = Y_test.loc[Y_test.index.isin(hit_indices_1)]  # 选择index
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
    space.append(i)

ks_auc_df = pd.DataFrame(
    {
        "ks_trains": ks_trains,
        "ks_tests": ks_tests,
        "auc_trains": auc_trains,
        "auc_tests": auc_tests,
    }
)
ks_auc_df.to_pickle("./ks_auc_df.pkl")

# %%
ks_auc_df = pd.read_pickle("./ks_auc_df.pkl")
space = [i for i in range(500)]

# 打印全部的迭代数据点
functions.models_ks("max_evals", space, ks_auc_df["ks_trains"], ks_auc_df["ks_tests"])
functions.models_auc(
    "max_evals", space, ks_auc_df["auc_trains"], ks_auc_df["auc_tests"]
)

# 打印全部的迭代数据点太多，可选择打印出指定区间内的数据
i = 450
j = 500
functions.models_ks(
    "max_evals", space[i:j], ks_auc_df["ks_trains"][i:j], ks_auc_df["ks_tests"][i:j]
)
functions.models_auc(
    "max_evals", space[i:j], ks_auc_df["auc_trains"][i:j], ks_auc_df["auc_tests"][i:j]
)
# %%
results = pd.read_csv("gbm_trials_500_ziyang.csv")
# Sort with best scores on top and reset index for slicing
results.sort_values("loss", ascending=True, inplace=True)
results.reset_index(inplace=True, drop=True)
# results.head()

params_488 = ast.literal_eval(results.loc[488, "params"])  # 将csv中的参数转为字典
params_3 = ast.literal_eval(results.loc[3, "params"])  # 将csv中的参数转为字典

for i in range(10):
    print(
        "{0}, {1}, {2}".format(
            i, results.loc[i, "loss"], ast.literal_eval(results.loc[i, "params"])
        )
    )
# 2019-11-21 01:54:20,452  -> 2019-11-21 02:34:01,666

X_rfe = pd.read_pickle("./X_rfe.pkl")
Data = pd.read_pickle("./Data.pkl")
Y_test = pd.read_pickle("./Y_test.pkl")
Y = pd.read_pickle("./Y.pkl")
X_test = pd.read_pickle("./X_test.pkl")
X = pd.read_pickle("./X.pkl")

fixed_params.update(params_3)
feature_num = fixed_params.get("feature_num")
chosen_final_feature = X_rfe.columns[:feature_num]  # 根据贝叶斯确定最终特征个数
X = X[chosen_final_feature]
X_test = X_test[chosen_final_feature]

flags = bairong.get_flags(chosen_final_feature)
hit_indices = bairong.get_hit_indices(Data, flags)
"""
手动调参
"""

X = X.loc[X.index.isin(hit_indices)]
Y = Y.loc[Y.index.isin(hit_indices)]
X_test = X_test.loc[X_test.index.isin(hit_indices)]
Y_test = Y_test.loc[Y_test.index.isin(hit_indices)]
lgbm_tuner = functions.LGBModelTuner(
    LGBMClassifier(**fixed_params), X, Y, X_test, Y_test, hit_indices
)

# 调max_depth
# max_depth是提高精确度的最重要的参数，一般会选取3-5
max_depth_ks_auc_df = lgbm_tuner.try_tune("max_depth", [1, 2, 3, 4, 5])
lgbm_tuner.tune("max_depth", 4)

# 调num_leaves
# 大致换算关系：num_leaves = 2^(max_depth)，但是它的值的设置应该小于 2^(max_depth)，否则可能会导致过拟合。
num_leaves_space = [i for i in range(5, 16)]
num_leaves_ks_auc_df = lgbm_tuner.try_tune("num_leaves", num_leaves_space)
functions.models_ks(
    "num_leaves",
    num_leaves_space,
    num_leaves_ks_auc_df["ks_trains"],
    num_leaves_ks_auc_df["ks_tests"],
)
functions.models_auc(
    "num_leaves",
    num_leaves_space,
    num_leaves_ks_auc_df["auc_trains"],
    num_leaves_ks_auc_df["auc_tests"],
)
lgbm_tuner.tune("num_leaves", 14)

# 调min_child_weight/min_sum_hessian_in_leaf
# 调完精度的参数，开始调控制过拟合的参数
# min_sum_hessian_in_leaf：也叫min_child_weight,
# 使一个结点分裂的最小海森值之和（Minimum sum of hessians in one leaf to allow a split. Higher values potentially decrease overfitting）,可以很好的控制过拟合。
# min_child_weight与min_sum_hessian_in_leaf只需要调节一个就好，两者是同一个东西。
min_child_weight_space = [10, 20, 40, 60, 80, 100, 110, 120, 130, 140, 160, 200]
# min_child_weight_space = [i for i in range(135,145)]
min_child_weight_ks_auc_df = lgbm_tuner.try_tune(
    "min_child_weight", min_child_weight_space
)  # 先粗调后细调，参数范围不要思维定式
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
lgbm_tuner.tune("min_child_weight", 139)

# 调min_data_in_leaf/min_child_samples 这两者是同一个参数，只需要调节一个就好
# min_data_in_leaf 是一个很重要的参数, 也叫min_child_samples，它的值取决于训练数据的样本个树 和num_leaves.
# 将其设置的较大可以避免生成一个过深的树, 但有可能导致欠拟合。可设置取值范围在 样本数/num_leaves附近调节。
min_child_samples_space = [100, 500, 1000, 1500, 2000, 2500, 3000, 3500]
min_child_samples_ks_auc_df = lgbm_tuner.try_tune(
    "min_child_samples", min_child_samples_space
)
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

lgbm_tuner.tune("min_child_samples", 2000)
# lgbm_tuner.estimator  # 查看当前模型的参数组合


# 调正则化参数lambda_l1:reg_alpha
# 毫无疑问，是降低过拟合的， 参数越大正则化越强
reg_alpha_space = [
    1.5,
    1.6,
    1.7,
    1.8,
    1.9,
    2.0,
    2.1,
    2.2,
    2.3,
    2.4,
    2.5,
    2.6,
    2.7,
    2.8,
    2.9,
    3.0,
    3.1,
    3.2,
]
reg_alpha_ks_auc_df = lgbm_tuner.try_tune("reg_alpha", reg_alpha_space)
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
lgbm_tuner.tune("reg_alpha", 2.3)

# 正则化参数lambda_l1:reg_lambda
reg_lambda_space = [i for i in np.arange(2.6, 3.8, 0.1)]
reg_lambda_ks_auc_df = lgbm_tuner.try_tune("reg_lambda", reg_lambda_space)
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
lgbm_tuner.tune("reg_lambda", 3.5)

# 调min_split_gain
# 执行切分的最小增益，也是控制过拟合的参数，值越大，对过拟合控制越明显，设置过大会欠拟合
min_split_gain_space = [i for i in np.arange(0.5, 1.5, 0.1)]
min_split_gain_ks_auc_df = lgbm_tuner.try_tune("min_split_gain", min_split_gain_space)
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
lgbm_tuner.tune("min_split_gain", 1.0)

# 调learning_rate
# 在调完所有的参数后，再来对learning_rate进行调节，
learning_rate_space = [i for i in np.arange(0.005, 0.1, 0.005)]
learning_rate_ks_auc_df = lgbm_tuner.try_tune("learning_rate", learning_rate_space)
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

df_train_auc, df_test_auc = tune_n_estimators_learning_rate(
    lgbm_tuner,
    learning_rate=[0.001, 0.005, 0.01, 0.02, 0.04, 0.1, 0.2],
    n_estimators=[5000, 2300, 1200, 600, 300, 120, 60],
)
functions.plot_n_estimators_learning_rate(df_train_auc, df_test_auc)
lgbm_tuner.tune("n_estimators", 140)
lgbm_tuner.tune("learning_rate", 0.08)
