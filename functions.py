import csv
import math
import os
import pprint
import re
from pathlib import Path
import numpy as np
import pandas as pd
from typing import List, Union, Tuple, Callable
from logging import getLogger, INFO, FileHandler, StreamHandler, Formatter
from sklearn.model_selection import train_test_split, KFold
from pandas.core.dtypes.common import (
    is_numeric_dtype,
    is_string_dtype,
    is_categorical_dtype,
    is_object_dtype,
)
from tqdm import tqdm
import lightgbm as lgb
from lightgbm.sklearn import LGBMClassifier, LGBMRegressor
import time
from hyperopt import STATUS_OK
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import seaborn as sns

logger = getLogger("Data_process")
logger.setLevel(INFO)

log_file = FileHandler("Log_detail.log")  # 创建一个handler，用于写入日志文件
log_file.setLevel(INFO)

log_console = StreamHandler()  # 创建一个handler，用于输出到控制台
log_console.setLevel(INFO)

# 设置log格式
log_formatter = formatter = Formatter("%(asctime)s - %(levelname)s - %(message)s")
log_console.setFormatter(log_formatter)
log_file.setFormatter(log_formatter)

# 给logger添加handler
logger.addHandler(log_console)
logger.addHandler(log_file)

ZH_CN_REGEX = r"([\u4e00-\u9fff]+)"  # 中国大陆区域的中文


def get_header(data: str, verbose: bool = False):
    """
    检查中文在第几行

    :param verbose: show extra info
    :param data: input data, should be csv or excel
    :return: with_detail：True 返回header内容和中文所在行
                          False 只返回中文所在行
    """

    read_func, encoding = get_file_processing_method(data)
    with open(data, "rb") as f:
        tmp_df = read_func(f, header=[0], nrows=3, encoding=encoding)

    zh_cn_regex = re.compile(ZH_CN_REGEX)

    # 若第一行全是中文
    if all(zh_cn_regex.search(f) for f in tmp_df.columns):
        zh_header_num = 0

    # 用第一行做header，若第二行有中文 and 数据类型全是object
    elif np.all(tmp_df.dtypes == "object") and any(
        zh_cn_regex.search(f) for f in tmp_df.iloc[0, :].values
    ):
        zh_header_num = 1

    # 用第一行做header, 数据类型不全是object且原数据第二行没有中文
    else:
        zh_header_num = -1

    header_detail = tmp_df.columns.values.tolist()

    if verbose:
        if zh_header_num == -1:
            logger.info(f"元数据中无中文表头")
        else:
            logger.info(f"元数据中第{zh_header_num}行为中文表头")

    return zh_header_num, header_detail


def _sizeof_fmt(num):
    # returns size in human readable format
    for x in ["bytes", "KB", "MB", "GB", "TB"]:
        if num < 1024.0:
            return "{num:3.3f} " "{x}".format(num=num, x=x)
        num /= 1024.0
    return "{num:3.1f} {pb}".format(num=num, pb="PB")


def subset(ls1, ls2) -> bool:
    """
    Decide whether ls1 is subset of ls2.

    :param ls1: first iterable list
    :param ls2: second iterable list
    :return: subset check result

    :example:
    >>> subset([1, 2, 3], [3, 2, 1])
    True
    """
    l1, l2 = len(ls1), len(ls2)
    if l1 == 0:
        return True
    if l2 == 0 or l2 < l1:
        return False

    return all(e in ls2 for e in ls1)


def split_train_test(
    x: Union[pd.DataFrame, pd.Series],
    y: Union[pd.DataFrame, pd.Series],
    test_size: float = None,
    train_size: float = None,
    shuffle: bool = True,  # 在划分数据之前先打乱数据
    stratify: np.ndarray = None,
    *,
    random_state: int = None,
) -> Tuple:
    """
    Split dataframe or series into random train and test subsets. (wrap of sklearn `train_test_split` function)
    :param x: whole data
    :param y: whole label
    :param test_size: proportion of the dataset to include in the test split
    :param train_size: proportion of the dataset to include in the train split
    :param shuffle: Whether or not to shuffle the data before splitting. If shuffle=False, then stratify must be None.
    :param stratify: If not None, data is split in a stratified fashion, using this as the class labels.
    :param random_state: the seed used by the random number generator
    :return: List containing train-test split of inputs.
    """
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        train_size=train_size,
        shuffle=shuffle,
        stratify=stratify,
        random_state=random_state,
    )

    logger.info(
        "Splitting original shape x={}, y={} into X_train={}, X_test={}, y_train={}, y_test={}...".format(
            x.shape, y.shape, x_train.shape, x_test.shape, y_train.shape, y_test.shape
        )
    )
    return x_train, x_test, y_train, y_test


# 获取文件后缀
def get_file_size(file_name, with_unit: bool = False):
    """
    Get system file size.

    :param file_name: file name
    :param with_unit: 是否带单位

    :return: size_str: size with "bytes", "KB", "MB", "GB", "TB"
             size : size (B)
    """
    size = os.path.getsize(file_name)

    if with_unit:
        return _sizeof_fmt(size), size
    return size


def get_file_processing_method(file: str):
    file_type = Path(file).suffix
    if file_type == ".csv":
        read_func = pd.read_csv
        encoding = get_file_encoding(file)
    elif file_type in (".xlsx", ".xls"):
        read_func = pd.read_excel
        # for excel encoding, it should be 'UTF-8'
        encoding = "UTF-8"
    else:
        raise ValueError("Input data should be a csv/xlsx file!")
    return read_func, encoding


def get_file_encoding(file: str) -> str:
    """
    Get encoding of the file.

    :param file: absolute path of the file
    :return: file encoding

    """
    import chardet

    with open(file, "rb") as f:
        return chardet.detect(f.read())["encoding"]


# read_from(data,index_col,skiprows,skip_cols,verbose)
def read_from(
    data: str,
    header: int = 0,
    index_col: str = None,
    skiprows: List[str] = None,
    skip_cols: List[str] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Read data intelligently. (only read in the possible models features, nonsense columns are omitted!)

    :param header: 自定义header，默认为0
    :param data: input csv/excel data
    :param index_col: 设置index所在的col列
    :param skiprows: skip的行
    :param skip_cols: skip columns
    :param verbose: show extra info

    :return: data -> DafaFrame

    """
    if not os.path.exists(data):
        raise FileNotFoundError("Invalid file input!")
    os.chmod(data, 0o777)  # read/write by everyone
    data_name = Path(data).stem
    file_size_str, file_size = get_file_size(data, with_unit=True)
    if verbose:
        logger.info("Input file({}) has {}.".format(data_name, file_size_str))
    read_func, encoding = get_file_processing_method(data)

    _, header_detail = get_header(data)

    if skip_cols is None:
        skip_cols = []
    else:
        if verbose:
            logger.info(
                "skip_cols columns founded, result will skip all these columns."
            )

    use_cols = [col for col in header_detail if col not in skip_cols]
    read_options = {
        "header": header,
        "skiprows": skiprows,
        "index_col": index_col,
        "usecols": use_cols,
        "encoding": encoding,
        "low_memory": False,
    }

    if file_size < 50:
        with open(data, "rb") as f:
            res = read_func(f, **read_options)
    else:
        chunk_num = math.ceil((sum(1 for _ in open(data, "rb")) - 1) / 10000)
        read_options["chunksize"] = 10000
        if verbose:
            logger.info("Start reading file!")

        with open(data, "rb") as f:
            dfs = read_func(f, **read_options)
            res = pd.concat(
                df for df in tqdm(dfs, desc="Reading file", total=chunk_num)
            )
        if verbose:
            logger.info("Reading file completed!")
    return res


def del_nan(
    df: pd.DataFrame, nan_ratio_threshold: float = 0.95, inplace: bool = False
) -> Union[None, pd.DataFrame]:
    """
    Delete the columns with too much nan values.

    :param df: data object
    :param nan_ratio_threshold: nan ratio threshold
    :param inplace: inplace change or not
    :return: data with reasonable nan values
    """
    assert isinstance(df, pd.DataFrame), "Input should be a pandas dataframe!"
    nan_series = df.isna().mean()
    for feature, ratio in nan_series.iteritems():
        passed = ratio <= nan_ratio_threshold
        if not passed:
            logger.info(
                "{} Feature {} denied with nan ratio = {:.3f}...".format(
                    "×", repr(feature), ratio
                )
            )
    if inplace:
        drop_features = nan_series[nan_series > nan_ratio_threshold].index
        df.drop(columns=drop_features, inplace=True)
        logger.info(
            "删除缺失率在{}以上的变量, 还剩余{}个候选变量".format(nan_ratio_threshold, len(df.columns))
        )
        return None
    else:
        res = df.loc[:, nan_series.values <= nan_ratio_threshold].copy()
        logger.info(
            "删除缺失率在{}以上的变量, 还剩余{}个候选变量".format(nan_ratio_threshold, len(res.columns))
        )
        return res


def del_mode(
    df: pd.DataFrame, mode_ratio_threshold: float = 0.95, inplace=False
) -> Union[None, pd.DataFrame]:
    """
    Delete the columns with too much mode values. (mode does not contain nan)

    :param df: data object
    :param mode_ratio_threshold: mode ratio threshold
    :param inplace: inplace change or not
    :return: data with reasonable mode values
    """
    assert isinstance(df, pd.DataFrame), "Input should be a pandas dataframe!"

    def col_mode_ratio(col: pd.Series) -> float:
        summary = (
            col.value_counts()
        )  # Return a Series containing counts of unique values.其中不包含空值
        # summary = col.value_counts(dropna=False)  # Return a Series containing counts of unique values. 其中包含空值
        ratio = summary.iloc[0] / sum(summary) if len(summary) > 0 else 1.0  # 分母不包含空值
        # passed = ratio <= mode_ratio_threshold
        logger.info(
            "× Feature {} denied with mode ratio = {:.3f}...".format(
                repr(col.name), ratio
            )
        )
        return ratio

    if inplace:
        df.drop(
            columns=(
                f for f in df.columns if col_mode_ratio(df[f]) > mode_ratio_threshold
            ),
            inplace=True,
        )
        logger.info(
            "删除同值比例超过{}的变量后，还剩余{}个候选变量".format(mode_ratio_threshold, len(df.columns))
        )
        return None
    else:
        res = df[
            (f for f in df.columns if col_mode_ratio(df[f]) <= mode_ratio_threshold)
        ].copy()
        logger.info(
            "删除同值比例超过{}的变量后，还剩余{}个候选变量".format(mode_ratio_threshold, len(res.columns))
        )
        return res


# 填充缺失值 + 数据瘦身
def slim(
    df: Union[pd.DataFrame, pd.Series],
    nan_replacements: Tuple[int, str] = (-99, "blank"),
    inplace: bool = False,
) -> Union[pd.DataFrame, pd.Series]:
    """
    Type fit function to slim the whole data object.

    :param df: pandas data object, can be DataFrame or Series
    :param nan_replacements: column nan replacements, including numeric nan and category nan values, i.e. -99 or 'blank'
    :param inplace: inplace change or not (default is False)
    :return: slimed data

    """

    if isinstance(df, pd.Series):
        origin_memory_usage = df.memory_usage(index=True)
        res = slim_col(df, nan_replacements=nan_replacements, inplace=inplace)
        current_memory_usage = res.memory_usage(index=True)

    elif isinstance(df, pd.DataFrame):
        origin_memory_usage = df.memory_usage(index=True).sum()
        res = pd.DataFrame(
            {
                f: slim_col(df[f], nan_replacements=nan_replacements, inplace=inplace)
                for f in df.columns
            },
            index=df.index,
        )
        current_memory_usage = res.memory_usage(index=True).sum()

    else:
        raise TypeError(f"Unsupported type({type(df)}) for slim operation!")

    logger.info(
        "Memory usage reduced from {} to {}".format(
            _sizeof_fmt(origin_memory_usage), _sizeof_fmt(current_memory_usage),
        )
    )
    return res


def slim_col(
    x: pd.Series,
    nan_replacements: Tuple[int, str] = (-99, "blank"),
    inplace: bool = False,
) -> pd.Series:
    """
    Type fit function to slim a specific column.

    :param x: data column
    :param nan_replacements: column nan replacements, including numeric nan and category nan values, i.e. -99 or 'blank'
    :param inplace: inplace change or not (default is False)
    :return: slimed column
    """
    num_nan, cat_nan = nan_replacements
    origin_type = x.dtype.name
    x_has_nan = x.isna().values.any()  # bool: series中发现nan -> True
    if is_numeric_dtype(x):
        if x_has_nan:
            if inplace:
                x.fillna(value=num_nan, inplace=inplace)
            else:
                x = x.fillna(value=num_nan, inplace=inplace)
        as_int = x.fillna(0).astype(np.int64)  # 若inplace=False则已经填充过了-99
        is_int = np.allclose(x, as_int)
        mn, mx = x.min(), x.max()
        if is_int:
            if mn >= 0:
                if mx < 255:
                    x = x.astype(np.uint8)
                elif mx < 65535:
                    x = x.astype(np.uint16)
                elif mx < 4294967295:
                    x = x.astype(np.uint32)
                else:
                    x = x.astype(np.uint64)
            else:
                if -128 < mn and mx < 127:
                    x = x.astype(np.int8)
                elif -32768 < mn and mx < 32767:
                    x = x.astype(np.int16)
                elif -2147483648 < mn and mx < 2147483647:
                    x = x.astype(np.int32)
                elif -9223372036854775808 < mn and mx < 9223372036854775807:
                    x = x.astype(np.int64)
                else:
                    raise OverflowError("Integer overflow encountered!")

    # 若为分类型, 则填充为'blank'并将列类型设置为category,Converting string variable to a categorical variable will save memory
    elif is_string_dtype(x) or is_categorical_dtype(x) or is_object_dtype(x):
        if x_has_nan:
            if inplace:
                x.fillna(value=cat_nan, inplace=inplace)
            else:
                x = x.fillna(value=cat_nan, inplace=inplace)
        x = x.astype("category")

    logger.info(
        "Column {} data type changes from `{}` to `{}`.".format(
            repr(x.name), origin_type, x.dtype.name
        )
    )
    return x


def del_cat(
    df: pd.DataFrame, cat_threshold: int = 10, inplace: bool = False
) -> Union[None, pd.DataFrame]:
    """
    Delete the columns with too much category.

    :param df: data object
    :param cat_threshold: category count threshold
    :param inplace: inplace change or not
    :return: data with reasonable category count
    """

    def col_nunique(col: pd.Series) -> int:
        # make sure is slimed
        assert (
            col.notna().values.all()
        ), f"Column {col.name} should be filled with nan replacements: i.e. -99 or 'blank'!"
        cat_num = col.nunique(
            dropna=True
        )  # Return number of unique elements in the object.
        # passed = cat_num <= cat_threshold
        logger.info(
            "× Feature {} droped with {} unique values, exclude `np.nan`...".format(
                repr(col.name), cat_num,
            )
        )
        return cat_num

    cat_features = [f for f in df.columns if is_categorical_dtype(df[f])]
    if inplace:
        df.drop(
            columns=(f for f in cat_features if col_nunique(df[f]) > cat_threshold),
            inplace=True,
        )
        logger.info("删除类别型个数超过{}的变量后，还剩余{}个候选变量".format(cat_threshold, len(df.columns)))
        return None
    else:
        df = df.drop(
            columns=(f for f in cat_features if col_nunique(df[f]) > cat_threshold),
            inplace=False,
        ).copy()
        logger.info("删除类别型个数超过{}的变量后，还剩余{}个候选变量".format(cat_threshold, len(df.columns)))
        return df


def positions_of(df: pd.DataFrame, *, kind: str = "nan") -> List[str]:
    """
    Get features of a specific kind.

    :param df: pandas data object
    :param kind: feature kind ('cat' or 'num' or 'nan')
    :return: feature list with specific type

    """
    assert isinstance(df, pd.DataFrame), "Input should be a pandas dataframe!"
    df_features = df.columns
    num_kinds = ("i", "u", "f", "c")
    cat_kinds = ("b", "O", "S", "U")
    if kind == "nan":
        res = [f for f in df_features if df[f].isna().values.any()]
    elif kind == "num":
        res = [f for f in df_features if df[f].dtype.kind in num_kinds]
    elif kind == "cat":
        res = [f for f in df_features if df[f].dtype.kind in cat_kinds]
    else:
        raise ValueError(
            f"Unknown kind {kind} encountered, use 'cat', 'num' or 'nan' instead!"
        )
    logger.info("Found {} {} features...".format(len(res), kind))
    return res


def get_dummied(df: pd.DataFrame) -> pd.DataFrame:
    """"
    Get dummied data.

    :param df: data object
    :return: dummied data

    :example:

    """
    # 此步必须在进行类别变量转换之前
    num_features, cat_features = (
        positions_of(df, kind="num"),
        positions_of(df, kind="cat"),
    )
    df_num, df_cat = df[num_features], df[cat_features]
    if len(cat_features) == 0:
        logger.warning("No category feature in df, return original dataframe...")
        return df_num
    else:
        # 对类别型变量进行one-hot编码, 转化成dummy变量并生成原始变量到dummy变量的映射表
        logger.info(
            "Found {} category features in df, applying dummy transform...".format(
                len(cat_features)
            )
        )
        x_cat_dummied = pd.get_dummies(df_cat)
        res = df_num.join(x_cat_dummied)
        logger.info(
            "After dummy transform, there are {} features...".format(len(res.columns))
        )
        return res


def get_sorted_feature(
    origin_features: Union[pd.Index, np.ndarray, list],
    importances: np.ndarray,
    threshold: Union[int, float] = None,
    operator: str = ">",
) -> List[str]:
    """
    Sort feature by feature importance, if threshold provided, return the sorted feature only satisfying the threshold.

    :param origin_features: original features, extracted from dataframe
    :param importances: feature importance, obtain by model or rules
    :param threshold: flexible number to decide whether to accept a feature
    :param operator: greater than threshold or not, default is '>'
    :return: sorted feature, also satisfying the threshold

    :example:

    >>> import numpy as np
    >>> features = ['f1', 'f2', 'f3', 'f4']
    >>> importances = np.array([0.4, 1.2, -0.2, 1.0])
    >>> get_sorted_feature(features, importances, threshold=0)
    ['f2', 'f4', 'f1']
    """
    from operator import le, lt, eq, ne, ge, gt

    operator_map = {"<=": le, "<": lt, "=": eq, "!=": ne, ">=": ge, ">": gt}
    if threshold is None:
        feature_importance = {f: imp for f, imp in zip(origin_features, importances)}
    else:
        feature_importance = {
            f: imp
            for f, imp in zip(origin_features, importances)
            if operator_map[operator](imp, threshold)
        }
    res = sorted(
        feature_importance.keys(), key=feature_importance.__getitem__, reverse=True
    )
    logger.info(
        "Number of valid feature(importance {} {}) is {}".format(
            operator, threshold, len(res)
        )
    )
    return res


def calc_ks(
    score: Union[pd.Series, np.ndarray],
    target: Union[pd.Series, np.ndarray],
    method="origin_pf",
) -> float:
    """
    Calculate ks value between predict and target.

    :param score: predict result (can be score or probability) 代表模型得分（一般为预测正类(0)的概率）
    :param target: target label
    :param method: method name, e.g. 'origin', 'crosstab', 'roc'
    :return: ks value

    :example:
    >>> data_1 = pd.DataFrame({'y': [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0], 'X': [1, 2, 4, 2, 2, 6, 5, 3, 0, 5, 4, 18]})
    >>> X, y = data_1['X'], data_1['y']
    >>> calc_ks(X, y)
    0.5
    >>> calc_ks(X, y, method='roc')
    0.5
    >>> calc_ks(X, y, method='origin')
    0.5
    """
    if isinstance(score, pd.Series):
        score = score.values
    if isinstance(target, pd.Series):
        target = target.values
    assert len(np.unique(target[~np.isnan(target)])) == 2, "Binary target is required!"
    if method == "crosstab":
        freq = pd.crosstab(score, target)
        dens = freq.cumsum(axis=0) / freq.sum()
        ks = np.max(np.abs(dens[0] - dens[1]))
    elif method == "roc":
        from sklearn.metrics import roc_curve

        # roc_curve函数默认添加了原点
        fpr, tpr, _ = roc_curve(target, score)
        ks = np.max(np.abs(tpr - fpr))
        return ks
    # FIXED: 原始方法不能处理包含缺失值的情况
    elif method == "origin":
        notna_pos = ~np.isnan(score)
        score = score[notna_pos]
        target = target[notna_pos]
        sorted_score = np.sort(np.unique(score))
        bad_tot = np.sum(target == 1)
        good_tot = np.sum(target == 0)

        bad_good_diff = (
            abs(
                np.sum((score <= s) & (target == 1)) / bad_tot
                - np.sum((score <= s) & (target == 0)) / good_tot
            )
            for s in sorted_score
        )
        ks = max(bad_good_diff)
    elif method == "origin_pf":
        section_num = 20
        Y = pd.Series(target)
        sample_num = len(Y)
        bad_percent = np.zeros([section_num, 1])
        good_percent = np.zeros([section_num, 1])
        point = pd.DataFrame(score)
        sorted_point = point.sort_values(by=0)
        total_bad_num = len(np.where(Y == 1)[0])
        total_good_num = len(np.where(Y == 0)[0])
        for i in range(0, section_num):
            split_point = sorted_point.iloc[
                int(round(sample_num * (i + 1) / section_num)) - 1
            ]
            position_in_this_section = np.where(point <= split_point)[0]
            bad_percent[i] = (
                len(np.where(Y.iloc[position_in_this_section] == 1)[0]) / total_bad_num
            )
            good_percent[i] = (
                len(np.where(Y.iloc[position_in_this_section] == 0)[0]) / total_good_num
            )

        ks_value = bad_percent - good_percent
        ks = max(ks_value)[0]
    else:
        raise ValueError("Unsupported method {} encountered!".format(repr(method)))
    logger.info("Current predict KS is {}...".format(ks))
    return ks


# def init_iteration():
#     global iteration
#     iteration = 0

i = 0


def ks_kfold_objective(
    X_raw: pd.DataFrame,
    y: pd.Series,
    model: str = "lgb",
    params: dict = None,
    n_splits=3,
    random_state=7,
    shuffle=True,
    out_file="gbm_trials.csv",
) -> Callable:
    raw_features = X_raw.columns.values.tolist()
    # 将数据集分成了五份
    kf = KFold(n_splits=n_splits, random_state=random_state, shuffle=shuffle)
    global i
    i += 1

    def ks_score(kwargs: dict) -> dict:
        starttime = time.time()

        # Retrieve the subsample if present otherwise set to 1.0
        subsample = kwargs["boosting_type"].get("subsample", 1.0)

        # Extract the boosting type
        kwargs["boosting_type"] = kwargs["boosting_type"]["boosting_type"]
        kwargs["subsample"] = subsample

        # Make sure parameters that need to be integers are integers
        for k in (
            "feature_num",
            "max_depth",
            "num_leaves",
            "n_estimators",
            "min_child_samples",
            "subsample_for_bin",
        ):
            if isinstance(kwargs.get(k), float):
                kwargs[k] = int(kwargs[k])

        ks_list = []
        chosen_feature = raw_features[: kwargs["feature_num"]]
        X = X_raw[chosen_feature]

        logger.info("开始训练1111111111111111")
        logger.info("{}".format(kwargs))

        iter = 0
        estr = {}
        for train_idx, test_idx in kf.split(X):
            X_train, X_test = X.iloc[train_idx].values, X.iloc[test_idx].values
            y_train, y_test = y.iloc[train_idx].values, y.iloc[test_idx].values
            params.update(kwargs)
            try:
                params.pop("n_estimators")
            except KeyError:
                pass

            if model == "lgb":

                # # Retrieve the subsample if present otherwise set to 1.0
                # subsample = params["boosting_type"].get("subsample", 1.0)
                #
                # # Extract the boosting type
                # params["boosting_type"] = params["boosting_type"]["boosting_type"]
                # params["subsample"] = subsample
                train_data = lgb.Dataset(X_train, y_train, silent=True)
                cv_result = lgb.cv(
                    params,
                    train_data,
                    num_boost_round=10000,
                    nfold=5,
                    metrics="auc",
                    early_stopping_rounds=100,
                    verbose_eval=False,
                )
                # Boosting rounds that returned the highest cv score
                n_estimators = int(np.argmax(cv_result["auc-mean"]) + 1)

                logger.info("n_estimators:{}".format(n_estimators))
                params["n_estimators"] = n_estimators
                clf = LGBMClassifier(**params)
                clf.fit(X_train, y_train)
                y_pred_test = clf.predict_proba(X_test)[:, 1]

                iter += 1
                estr[iter] = {n_estimators}

            # elif model == 'xgb':
            #     train_data = xgb.DMatrix(X_train, y_train, silent=False)
            #     cv_result = xgb.cv(params, train_data, num_boost_round=500, nfold=3, metrics='auc',
            #                        early_stopping_rounds=100)
            #     params['n_estimators'] = len(cv_result)
            #     clf = XGBClassifier(**params)
            #     clf.fit(X_train, y_train)
            #     y_pred_test = clf.predict_proba(X_test)[:, 1]
            # elif model == 'adaboost':
            #     clf = AdaBoostClassifier(DecisionTreeClassifier(**kwargs, min_samples_leaf=0.05))
            #     clf.fit(X_train, y_train)
            #     y_pred_test = clf.predict_proba(X_test)[:, 1]
            else:
                raise NotImplementedError("Not implemented!")

            ks_list.append(calc_ks(y_pred_test, y_test, method="crosstab"))
            # model.predict_proba(X_test)=
            # array([[0.1,0.9],   #代表[2,3,4,5]被判断为0的概率为0.1，被判断为1的概率为0.9
            #        [0.8,0.2]])  #代表[3,4,5,6]被判断为0的概率为0.8，被判断为1的概率为0.2
        ks_arr = np.asarray(ks_list)
        score = np.mean(ks_arr) - 1.96 * np.std(ks_arr) / np.sqrt(
            len(ks_arr)
        )  # 置信区间 https://www.shuxuele.com/data/confidence-interval.html

        loss = -score
        run_time = time.time() - starttime

        # Write to the csv file ('a' means append)
        of_connection = open(out_file, "a")
        writer = csv.writer(of_connection)
        writer.writerow([loss, params, estr, run_time, i])

        return {
            "loss": loss,
            "params": params,
            "estimators": i,
            "train_time": run_time,
            "status": STATUS_OK,
        }

    return ks_score


class LGBModelTuner(object):
    def __init__(self, lgbm, X, y, X_test, y_test, hit_indices):
        self.estimator = lgbm
        self.X, self.Y = X, y
        self.X_test, self.Y_test = X_test, y_test
        assert hit_indices is not None, "Hit indices are necessary to get model result!"
        self.hit_indices = hit_indices
        self.dataset_train = lgb.Dataset(X.values, label=y.values)
        self.dataset_test = lgb.Dataset(X_test.values, label=y_test.values)
        # TODO: add history
        self.history = []
        if isinstance(lgbm, lgb.Booster):
            assert (
                lgbm.params.get("seed") is not None
            ), "Better to set `seed` for lightgbm booster!"
            lgbm.params.update({"num_threads": -1})
            self.params = lgbm.params
        elif isinstance(lgbm, (LGBMClassifier, LGBMRegressor)):
            assert (
                lgbm.get_params().get("random_state") is not None
            ), "Better to set `random_state` for lightgbm booster!"
            lgbm.n_jobs = -1
            self.params = lgbm.get_params()
        else:
            raise TypeError(
                "Input model should be a `lgb.Booster` or `LGBMClassifier`/`LGBMRegressor`!"
            )

    def get_model_result(self, params: dict) -> dict:
        X, y = self.X, self.Y
        X_test, y_test = self.X_test, self.Y_test
        # X, y = self.X.values, self.Y.values
        # X_test, y_test = self.X_test.values, self.Y_test.values
        if isinstance(self.estimator, lgb.Booster):
            params["metric"] = "auc"
            estimator = lgb.train(params, self.dataset_train)
            pred_train = pd.Series(
                estimator.predict(self.dataset_train), index=self.X.index
            )
            pred_test = pd.Series(
                estimator.predict(self.dataset_test), index=self.X_test.index
            )
        elif isinstance(self.estimator, LGBMRegressor):
            estimator = LGBMRegressor(**params)
            estimator.fit(X, y, eval_metric="auc")
            pred_train = pd.Series(estimator.predict(X), index=self.X.index)
            pred_test = pd.Series(estimator.predict(X_test), index=self.X_test.index)
        elif isinstance(self.estimator, LGBMClassifier):
            estimator = LGBMClassifier(**params)
            estimator.fit(X, y, eval_metric="auc")
            pred_train = pd.Series(estimator.predict_proba(X)[:, 1], index=self.X.index)
            pred_test = pd.Series(
                estimator.predict_proba(X_test)[:, 1], index=self.X_test.index
            )
        else:
            raise TypeError(
                "Input model should be a `lgb.Booster` or `LGBMClassifier`/`LGBMRegressor`!"
            )
        # 置空得分
        pred_train.loc[~pred_train.index.isin(self.hit_indices)] = np.nan
        pred_test.loc[~pred_test.index.isin(self.hit_indices)] = np.nan
        # 计算模型评估指标
        ks_train, ks_test = calc_ks(-pred_train, y), calc_ks(-pred_test, y_test)
        auc_train, auc_test = calc_auc(pred_train, y), calc_auc(pred_test, y_test)
        # return {'train': (ks_train, auc_train), 'test': (ks_test, auc_test)}
        return {"ks": (ks_train, ks_test), "auc": (auc_train, auc_test)}

    def try_tune(self, param: str, space: list, plot: bool = True):
        params = {k: v for k, v in self.params.items()}
        ks_trains, ks_tests = [], []
        auc_trains, auc_tests = [], []
        for value in tqdm(space):
            params.update({param: value})
            result = self.get_model_result(params)
            ks_train, ks_test = result["ks"][0], result["ks"][1]
            auc_train, auc_test = result["auc"][0], result["auc"][1]
            ks_trains.append(ks_train)
            ks_tests.append(ks_test)
            auc_trains.append(auc_train)
            auc_tests.append(auc_test)
            logger.info(
                "While {}={}, model ks train={}, test={}, auc is train={}, test={}...".format(
                    param,
                    repr(value) if isinstance(value, str) else value,
                    ks_train,
                    ks_test,
                    auc_train,
                    auc_test,
                )
            )
        if plot:
            models_ks(param, space, ks_trains, ks_tests)
            models_auc(param, space, auc_trains, auc_tests)
        ks_auc_df = {
            "ks_trains": ks_trains,
            "ks_tests": ks_tests,
            "auc_trains": auc_trains,
            "auc_tests": auc_tests,
        }

        return ks_auc_df

    def tune(self, param: str, value: Union[str, int, float]) -> None:
        self.params.update({param: value})
        #        check_lgb_params(self.params)
        lgbm = self.estimator
        if isinstance(lgbm, lgb.Booster):
            lgbm.params = self.params
        elif isinstance(lgbm, (LGBMClassifier, LGBMRegressor)):
            lgbm.set_params(**self.params)
        else:
            raise TypeError(
                "Input model should be a `lgb.Booster` or `LGBMClassifier`/`LGBMRegressor`!"
            )
        logger.info(self.params)
        return None


def models_ks(hyper_param: str, space: list, ks_trains: list, ks_tests: list) -> None:
    """
    Plot train test data distribution curve.

    :param hyper_param: hyper parameter
    :param space: parameter value space
    :param ks_trains: train data ks
    :param ks_tests: test data ks
    :return: None
    """
    plt.figure(figsize=(20, 10))
    plt.plot(space, ks_trains, "s-", color="r", label="KS train")
    plt.plot(space, ks_tests, "o-", color="g", label="KS test")
    if len(space) <= 100:
        plt.xticks(space)
    else:
        pass
    plt.legend(loc="lower right")
    plt.ylabel("KS value")
    plt.xlabel(f"{hyper_param}")
    plt.title(f"KS value vs. Parameter tuning")
    plt.show()
    return None


def models_auc(
    hyper_param: str, space: list, auc_trains: list, auc_tests: list
) -> None:
    """
    Plot train test data distribution curve.

    :param hyper_param: hyper parameter
    :param space: parameter value space
    :param auc_trains: train data auc
    :param auc_tests: test data auc
    :return: None
    """
    plt.figure(figsize=(20, 10))
    plt.plot(space, auc_trains, "s-", color="r", label="AUC train")
    plt.plot(space, auc_tests, "o-", color="g", label="AUC test")

    if len(space) <= 100:
        plt.xticks(space)
    else:
        pass

    plt.legend(loc="lower right")
    plt.ylabel("AUC value")
    plt.xlabel(f"{hyper_param}")
    plt.title(f"AUC value vs. Parameter tuning")
    plt.show()
    return None


def calc_auc(
    preds: Union[np.ndarray, pd.Series], actual: Union[np.ndarray, pd.Series]
) -> float:
    """
    Calculate receiver operating characteristic.

    :param preds: predict result
    :param actual: actual label
    :return: false positive rate, true positive rate and auc value
    """
    if isinstance(actual, pd.Series):
        actual = actual.values
    if isinstance(preds, pd.Series):
        preds = preds.values

    notna_pos = ~np.isnan(preds)
    actual = actual[notna_pos]
    preds = preds[notna_pos]

    fpr, tpr, _ = roc_curve(actual, preds)
    auc_value = auc(fpr, tpr)
    logger.info("Current predict AUC is {}...".format(auc_value))
    return auc_value


def tune_n_estimators_learning_rate(lgbm_tuner, learning_rate, n_estimators):
    ks_trains = []
    ks_tests = []
    auc_trains = []
    auc_tests = []
    n_estimators = n_estimators
    learning_rate = learning_rate
    for i in n_estimators:
        for j in learning_rate:
            lgbm_tuner.tune("n_estimators", i)
            lgbm_tuner.tune("learning_rate", j)
            params = lgbm_tuner.params
            result_ks = lgbm_tuner.get_model_result(params)
            train_ks = result_ks["ks"][0]
            test_ks = result_ks["ks"][1]
            train_auc = result_ks["auc"][0]
            test_auc = result_ks["auc"][1]
            ks_trains.append(train_ks)
            ks_tests.append(test_ks)
            auc_trains.append(train_auc)
            auc_tests.append(test_auc)

    n_estimators_a = np.array(n_estimators)
    learning_rate_a = np.array(learning_rate)
    learning_rate_a, n_estimators_a = np.meshgrid(learning_rate_a, n_estimators_a)

    n_estimators_a.shape = (1, len(n_estimators) * len(learning_rate))
    learning_rate_a.shape = (1, len(n_estimators) * len(learning_rate))

    df = pd.DataFrame()
    df["n_estimators"] = n_estimators_a[0]
    df["learning_rate"] = learning_rate_a[0]
    df["auc_trains"] = auc_trains
    df["auc_tests"] = auc_tests

    df_pt = df.pivot_table(index="learning_rate", columns="n_estimators")
    df_pt.head()

    df_test = df_pt["auc_tests"]
    df_train = df_pt["auc_trains"]

    return df_train, df_test


def plot_n_estimators_learning_rate(df_train, df_test):
    f, ax = plt.subplots(figsize=(10, 4))
    cmap = sns.cubehelix_palette(start=1, rot=3, gamma=1, as_cmap=True,)
    sns.heatmap(
        df_train, cmap=cmap, linewidths=0.05, ax=ax, center=1.05 * df_train.max().max()
    )
    ax.set_title("Auc of Train cross by n_estimators and learning_rate")
    ax.set_xlabel("n_estimators")
    ax.set_ylabel("learning_rate")
    plt.show()

    f, ax = plt.subplots(figsize=(10, 4))
    cmap = sns.cubehelix_palette(start=1, rot=3, gamma=0.8, as_cmap=True)
    sns.heatmap(
        df_test, cmap=cmap, linewidths=0.05, ax=ax, center=1.05 * df_test.max().max()
    )
    ax.set_title("Auc of Test cross by n_estimators and learning_rate")
    ax.set_xlabel("n_estimators")
    ax.set_ylabel("learning_rate")
    plt.show()

    f, ax = plt.subplots(figsize=(10, 4))
    cmap = sns.cubehelix_palette(start=1, rot=3, gamma=0.8, as_cmap=True)
    sns.heatmap(
        df_train - df_test,
        cmap=cmap,
        linewidths=0.05,
        ax=ax,
        center=1.5 * (df_train - df_test).max().max(),
    )
    ax.set_title("Auc of (Train-test) cross by n_estimators and learning_rate")
    ax.set_xlabel("n_estimators")
    ax.set_ylabel("learning_rate")
    plt.show()
