from collections import namedtuple
from typing import Sequence, List, Optional
from logging import getLogger, INFO, FileHandler, StreamHandler, Formatter
import pandas as pd
import functions
import numpy as np

logger = getLogger("Bairong")
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
# logger.addHandler(log_file)


THREE_ELEMENT_DERIVE = ['id_province', 'id_city_level', 'cell_province', 'cell_city_level', 'cell_comp', 'gender',
                        'apply_age', 'id_region', 'cell_region', 'is_id_capital', 'is_cell_capital', 'person',
                        'level_cell_city', 'level_id_city', 'level_id_cell', 'is_province', 'is_city', 'province_city',
                        'id_city_level_blank', 'id_city_level_一线', 'id_city_level_三线', 'id_city_level_二线',
                        'id_city_level_五线', 'id_city_level_其他', 'id_city_level_四线',
                        'cell_city_level_blank', 'cell_city_level_一线', 'cell_city_level_三线', 'cell_city_level_二线',
                        'cell_city_level_五线', 'cell_city_level_其他', 'cell_city_level_四线',
                        'cell_comp_blank', 'cell_comp_电信', 'cell_comp_移动', 'cell_comp_联通', 'gender_blank', 'gender_女',
                        'gender_男',
                        'id_region_blank', 'id_region_东北地区', 'id_region_华东地区', 'id_region_华中地区', 'id_region_华北地区',
                        'id_region_华南地区', 'id_region_西北地区', 'id_region_西南地区',
                        'cell_region_blank', 'cell_region_东北地区', 'cell_region_华东地区', 'cell_region_华中地区',
                        'cell_region_华北地区', 'cell_region_华南地区', 'cell_region_西北地区', 'cell_region_西南地区',
                        'person_blank', 'person_中年女性', 'person_中年男性', 'person_青年女性', 'person_青年男性',
                        'level_id_cell_blank', 'level_id_cell_向上迁移', 'level_id_cell_向下迁移', 'level_id_cell_平行迁移',
                        'is_province_blank', 'is_province_一致', 'is_province_不一致', 'is_city_blank', 'is_city_一致',
                        'is_city_不一致', 'province_city_blank', 'province_city_城市与省份均迁移', 'province_city_城市迁移',
                        'province_city_未迁移', 'province_city_省份迁移']
ModuleMeta = namedtuple('ModuleMeta', ['flag_name', 'en_name', 'cn_name', 'description'])


def get_module_meta(prefix: str) -> Optional[ModuleMeta]:
    """
    Get module meta. (return None if module not found)

    :param prefix: module prefix, e.g. 'badinfo', 'ex', 'sl'
    :return: specific ModuleMeta
    """
    if prefix == 'badinfo':
        return ModuleMeta('flag_badinfo', 'badinfo', '自然人识别', '识别自然人信息')
    if prefix == 'ex':
        return ModuleMeta('flag_execution', 'ex', '法院被执行人—个人版', '查询个人的法院失信被执行人、被执行人的执行案件信息')
    if prefix == 'sl':
        return ModuleMeta('flag_specialList_c', 'sl', '特殊名单验证',
                          '用户本人、联系人、与用户有亲密关系的人（一度关系、二度关系-百融关系库定义）是否疑似命中中风险、一般风险、资信不佳、拒绝、高风险等百融特殊名单、以识别个体是否有虚假申请、欺诈等风险。')
    if prefix == 'bankfourpro':
        return ModuleMeta('flag_bankfourpro', 'bankfourpro', '银行卡四要素验证', '验证用户银行卡号、姓名、身份证号、手机号与银行预留信息是否一致')
    if prefix == 'idtwo':
        return ModuleMeta('flag_idtwo_z', 'idtwo_z', '身份证二要素验证', '核查身份证姓名是否一致')
    if prefix == 'idtwo_z':
        return ModuleMeta('flag_idtwo_z', 'idtwo_z', '身份证二要素验证', '核查身份证姓名是否一致')
    if prefix == 'telCheck':
        return ModuleMeta('flag_telCheck', 'telCheck', '手机三要素-移动联通电信', '验证移动、联通、电信手机号与绑定身份证号和姓名是否一致。')
    if prefix == 'bankthree':
        return ModuleMeta('flag_bankthree', 'bankthree', '银行卡三要素验证', '验证用户银行卡号、姓名、身份证号与银行预留信息是否一致。')
    if prefix == 'cv':
        return ModuleMeta('flag_companyver', 'cv', '单位验证', '单位名称和单位地址一致性验证。')
    if prefix == 'ka':
        return ModuleMeta('flag_keyattribution', 'ka', '身份证号手机号归属地', '？')
    if prefix == 'als':
        return ModuleMeta('flag_applyloanstr', 'als', '借贷意向验证（ApplyLoanStr_同评估报告）',
                          '用户近7/15天、1/3/6个月在百融的虚拟信贷联盟(银行、非银、非银细分类型)中的多次信贷申请情况。')
    if prefix == 'alf':
        return ModuleMeta('flag_ApplyFeature', 'alf', '借贷意向衍生特征', '根据用户过往申请记录生成反应借贷意向的衍生特征。')
    if prefix == 'tl':
        return ModuleMeta('flag_totalloan', 'tl', '借贷行为验证', '用户在百融的虚拟信贷联盟中的借贷行为情况。')
    if prefix == 'ir':
        return ModuleMeta('flag_inforelation', 'ir', '实名信息验证', '通过验证客户申请信息之间的关联关系、来判断客户的风险。')
    if prefix == 'frg':
        return ModuleMeta('flag_fraudrelation_g', 'frg', '团伙欺诈排查（通用版）', '团伙欺诈排查通用版是基于自有海量数据，通过算法挖掘用户的团伙欺诈行为。')
    if prefix == 'gl':
        return ModuleMeta('flag_graylist', '灰名单', 'gl', '根据用户过往的申请行为判断用户的团伙欺诈情况。')
    if prefix == 'drs':
        return ModuleMeta('flag_debtrepaystress', 'drs', '偿债压力指数', '用户本人当前偿债压力指数的情况。')
    if prefix == 'alu':
        return ModuleMeta('flag_applyloanusury', 'alu', '高风险借贷意向验证', '用户近 7/15 天、1/3/6/12 个月在超利贷机构中的多次申请情况。')
    if prefix == 'cons':
        return ModuleMeta('flag_consumption_c', 'cons', '商品消费指数', '商品消费产品查询用户商品消费行为、是对商品消费次数、金额和类目等维度的统计评估（自然月）。')
    if prefix == 'cf':
        return ModuleMeta('flag_ConsumptionFeature', 'cf', '商品消费衍生特征', '根据用户过往商品消费行为生成反应商品消费的衍生特征。')
    if prefix == 'location':
        return ModuleMeta('flag_location', 'location', '地址信息验证', '用户详细地址信息与百融地址信息库的一致性核查。')
    if prefix == 'stab':
        return ModuleMeta('flag_stability_c', 'stab', '稳定性指数', '用户查询信息与百融行为库中的信息是否匹配、来检验信息的关联性和一致性。')
    if prefix == 'media':
        return ModuleMeta('flag_media_c', 'media', '媒体阅览指数', '媒体阅览评估产品查询用户媒体阅览行为、是对阅览天数、类别等维度的统计评估（自然月）。')
    if prefix == 'ns':
        return ModuleMeta('flag_netshopping', 'ns', '消费指数', '查询个人的网购消费信息。')
    if prefix == 'pc':
        return ModuleMeta('flag_personalcre', 'pc', '个人资质-基础版', '查询用户消费、收入、资产、职业等信息，对用户消费等级、消费偏好、收入稳定性、职业等信息进行评估。')
    if prefix == 'pcp':
        return ModuleMeta('flag_personalcrepro', 'pcp', '个人资质-专业版',
                          '查询用户消费、收入、资产、职业等信息，对用户消费等级、消费偏好、收入稳定性、职业等信息进行评估，对银行卡消费情况的统计评估，展示1/3/6/12个月（自然月）支付行为情况（建议用户填其常用卡）。')
    return None


def get_flag(feature: str) -> str:
    """
    Get feature's flag.

    :param feature: feature name
    :return: flag of this feature
    :example:

    >>> get_flag('cons_m3_RYBH_pay')
    'flag_consumption_c'
    >>> get_flag('flag_consumption')
    'flag_consumption'
    """
    if feature.startswith('flag_'):
        return feature
    module = feature.split('_')[0]
    module_meta = get_module_meta(module)
    if module_meta is not None:
        return module_meta.flag_name
    else:
        raise ValueError(f"Unknown feature {repr(feature)} with no flag module!")


def get_flags(features: Sequence[str]) -> List[str]:
    """
    Get features' flags.

    :param features: list of feature names
    :return: flags of these features

    :example:

    >>> get_flags(['cons_m3_RYBH_pay', 'flag_consumption'])
    ['flag_consumption_c', 'flag_consumption']
    """
    flags = list(set(get_flag(f) for f in features if f not in THREE_ELEMENT_DERIVE))
    logger.info("Current features' corresponding flags are {}".format(flags))
    return flags


def get_hit_indices(X_raw: pd.DataFrame, final_flags: List[str]) -> np.ndarray:
    """
    Get sample hit indices.

    :param X_raw: raw data with all flag features
    :param final_flags: features flags
    :return: hit indices of all samples

    :example:

    >>> import pandas as pd
    >>> X = pd.DataFrame([[1, 0], [0, 1]], columns=list('AB'), index=list('CD'))
    >>> get_hit_indices(X, ['A', 'B'])
    array(['C', 'D'], dtype=object)
    """
    assert functions.subset(final_flags, X_raw.columns), "X_raw should contain all chosen feature of X!"
    flags = X_raw[final_flags]
    hit_pos = np.sum(flags.values, axis=1) >= 1
    hit_num, tot_num, hit_ratio = np.sum(hit_pos), len(hit_pos), np.average(hit_pos)
    logger.info("Current data hit result: hits={}, total={}, ratio={}...".format(hit_num, tot_num, hit_ratio))
    indices = flags.index.values[hit_pos]
    return indices
