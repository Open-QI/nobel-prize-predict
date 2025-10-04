import pandas as pd
import re
from typing import List, Dict, Any
import numpy as np

from mappings import institution_mapping, institution_onehot
from mappings import gender_map, country_map

import warnings
warnings.filterwarnings('ignore')

file_path = "data/nobel_dataset_wiki_sampled_100.csv"

df = pd.read_csv(file_path, header=0)

df["gender"] = df["gender"].map(gender_map)
df["origin_country"] = df["origin_country"].replace(country_map)
df["current_country"] = df["current_country"].replace(country_map)

df["award_age"] = df["award_year"] - df["birth_year"]
df["internet_weight"] = ((df["award_year"] - 1980) // 10).clip(lower=0)
df["total_citations_log"] = np.log1p(df["total_citations"])

df = df[df["data_parts_count"].isin([8, 6, 5, 4, 3, 2])]  
df = df[~pd.isna(df["award_age"])]  
print(df.shape)

cols = df.columns.tolist()

categorical_cols = ["origin_country", "current_country", "gender"]

citations_total_cols = ["total_citations", "hindex", "i10index"]

citations_year_cols = [
    c
    for c in cols
    if re.fullmatch(r"author_yearly_json_age_\d+", c)
    and 20 < int(re.search(r"\d+", c).group()) < 200
]

work_year_cols = [
    c
    for c in cols
    if re.fullmatch(r"top\d+_yearly_json_age_\d+", c)
    and 20 < int(re.search(r"age_(\d+)", c).group(1)) < 200
]


def extract_age_from_col(col_name: str) -> int:
    m = re.search(r"age_(\d+)$", col_name)
    return int(m.group(1)) if m else None


# 建立 age -> 列名 的映射（author 年度）
age_to_cit_col = {}
for c in citations_year_cols:
    age = extract_age_from_col(c)
    if age is not None:
        age_to_cit_col[age] = c


def extract_top_and_age(col_name: str):
    m = re.search(r"top(\d+)_yearly_json_age_(\d+)$", col_name)
    if not m:
        return None, None
    return int(m.group(1)), int(m.group(2))


top_to_age_to_col = {}
for c in work_year_cols:
    top_k, age = extract_top_and_age(c)
    if top_k is None or age is None:
        continue
    if top_k not in top_to_age_to_col:
        top_to_age_to_col[top_k] = {}
    top_to_age_to_col[top_k][age] = c


def generate_yearly_before_award_citations(df, max_years=50):
    """为每个人生成获奖前第1年到第n年的引用数据"""

    def get_citations_for_years_before_award(row):
        """获取获奖前指定年数的引用数据"""
        award_age = row["award_age"]
        result = {}

        # 生成获奖前第1年到第max_years年的引用数据
        for years_before in range(1, max_years + 1):
            target_age = award_age - years_before
            col_name = age_to_cit_col.get(target_age)

            if col_name and pd.notna(row[col_name]):
                result[f"citations_{years_before}_years_before_award"] = row[col_name]
            else:
                result[f"citations_{years_before}_years_before_award"] = 0

        return pd.Series(result)

    # 应用函数并展开结果
    yearly_citations = df.apply(get_citations_for_years_before_award, axis=1)

    # 将新列添加到原DataFrame
    for col in yearly_citations.columns:
        df[col] = yearly_citations[col]

    return df


df = generate_yearly_before_award_citations(df, max_years=50)


def generate_cumulative_citations_columns(df):
    """生成累积引用列：1-19年独立保留，20-25年聚合，26-35年聚合，36-50年聚合"""

    # 获取所有获奖前引用列
    citation_cols = [
        col
        for col in df.columns
        if col.startswith("citations_") and col.endswith("_years_before_award")
    ]

    # 按年数排序
    citation_cols_sorted = sorted(
        citation_cols,
        key=lambda x: int(re.search(r"(\d+)_years_before_award", x).group(1)),
    )

    # 分类列
    cols_1_to_19 = []
    cols_20_to_25 = []
    cols_26_to_35 = []
    cols_36_to_50 = []

    for col in citation_cols_sorted:
        years_before = int(re.search(r"(\d+)_years_before_award", col).group(1))
        if 1 <= years_before <= 19:
            cols_1_to_19.append(col)
        elif 20 <= years_before <= 25:
            cols_20_to_25.append(col)
        elif 26 <= years_before <= 35:
            cols_26_to_35.append(col)
        elif 36 <= years_before <= 50:
            cols_36_to_50.append(col)

    # 计算累积列
    df["citations_20_to_25_years_before_cumulative"] = df[cols_20_to_25].sum(axis=1)
    df["citations_26_to_35_years_before_cumulative"] = df[cols_26_to_35].sum(axis=1)
    df["citations_36_to_50_years_before_cumulative"] = df[cols_36_to_50].sum(axis=1)

    # 构建最终列列表
    final_columns = []

    # 添加获奖前1-19年的列（独立保留）
    final_columns.extend(cols_1_to_19)

    # 添加新生成的三个累积列
    final_columns.append(
        "citations_20_to_25_years_before_cumulative"
    )
    final_columns.append(
        "citations_26_to_35_years_before_cumulative"
    ) 
    final_columns.append(
        "citations_36_to_50_years_before_cumulative"
    )

    return final_columns, df


def generate_cumulative_growth_analysis(df):
    """生成累积引用增长分析：从50年开始按时间段累积到获奖前"""

    # 获取所有获奖前引用列
    citation_cols = [
        col
        for col in df.columns
        if col.startswith("citations_") and col.endswith("_years_before_award")
    ]

    # 按年数排序
    citation_cols_sorted = sorted(
        citation_cols,
        key=lambda x: int(re.search(r"(\d+)_years_before_award", x).group(1)),
    )

    # 定义时间段边界
    time_periods = [
        (50, 31, "citations_31_to_50_years_before_cumulative"),
        (30, 26, "citations_26_to_30_years_before_cumulative"), 
        (25, 21, "citations_21_to_25_years_before_cumulative"), 
        (20, 16, "citations_16_to_20_years_before_cumulative"), 
        (15, 11, "citations_11_to_15_years_before_cumulative"),
        (10, 1, "citations_1_to_10_years_before_cumulative"), 
    ]

    # 计算各时间段的累积引用
    for start_year, end_year, col_name in time_periods:
        period_cols = []
        for col in citation_cols_sorted:
            years_before = int(re.search(r"(\d+)_years_before_award", col).group(1))
            if end_year <= years_before <= start_year:
                period_cols.append(col)

        if period_cols:
            df[col_name] = df[period_cols].sum(axis=1)

    # 计算累积增长情况（从远期到近期）
    growth_columns = []

    # 50年总累积
    df["citations_50_years_total_cumulative"] = df[
        "citations_31_to_50_years_before_cumulative"
    ]

    # 30年累积（31-50年 + 26-30年）
    df["citations_30_years_cumulative"] = (
        df["citations_31_to_50_years_before_cumulative"]
        + df["citations_26_to_30_years_before_cumulative"]
    )

    # 25年累积（31-50年 + 26-30年 + 21-25年）
    df["citations_25_years_cumulative"] = (
        df["citations_31_to_50_years_before_cumulative"]
        + df["citations_26_to_30_years_before_cumulative"]
        + df["citations_21_to_25_years_before_cumulative"]
    )

    # 20年累积（31-50年 + 26-30年 + 21-25年 + 16-20年）
    df["citations_20_years_cumulative"] = (
        df["citations_31_to_50_years_before_cumulative"]
        + df["citations_26_to_30_years_before_cumulative"]
        + df["citations_21_to_25_years_before_cumulative"]
        + df["citations_16_to_20_years_before_cumulative"]
    )

    # 15年累积（31-50年 + 26-30年 + 21-25年 + 16-20年 + 11-15年）
    df["citations_15_years_cumulative"] = (
        df["citations_31_to_50_years_before_cumulative"]
        + df["citations_26_to_30_years_before_cumulative"]
        + df["citations_21_to_25_years_before_cumulative"]
        + df["citations_16_to_20_years_before_cumulative"]
        + df["citations_11_to_15_years_before_cumulative"]
    )

    # 10年累积（31-50年 + 26-30年 + 21-25年 + 16-20年 + 11-15年 + 1-10年）
    df["citations_10_years_cumulative"] = (
        df["citations_31_to_50_years_before_cumulative"]
        + df["citations_26_to_30_years_before_cumulative"]
        + df["citations_21_to_25_years_before_cumulative"]
        + df["citations_16_to_20_years_before_cumulative"]
        + df["citations_11_to_15_years_before_cumulative"]
        + df["citations_1_to_10_years_before_cumulative"]
    )

    # 计算增长率和增长量
    df["growth_30_vs_50"] = (
        df["citations_30_years_cumulative"] - df["citations_50_years_total_cumulative"]
    )
    df["growth_rate_30_vs_50"] = df["growth_30_vs_50"] / df[
        "citations_50_years_total_cumulative"
    ].replace(0, 1)

    df["growth_25_vs_30"] = (
        df["citations_25_years_cumulative"] - df["citations_30_years_cumulative"]
    )
    df["growth_rate_25_vs_30"] = df["growth_25_vs_30"] / df[
        "citations_30_years_cumulative"
    ].replace(0, 1)

    df["growth_20_vs_25"] = (
        df["citations_20_years_cumulative"] - df["citations_25_years_cumulative"]
    )
    df["growth_rate_20_vs_25"] = df["growth_20_vs_25"] / df[
        "citations_25_years_cumulative"
    ].replace(0, 1)

    df["growth_15_vs_20"] = (
        df["citations_15_years_cumulative"] - df["citations_20_years_cumulative"]
    )
    df["growth_rate_15_vs_20"] = df["growth_15_vs_20"] / df[
        "citations_20_years_cumulative"
    ].replace(0, 1)

    df["growth_10_vs_15"] = (
        df["citations_10_years_cumulative"] - df["citations_15_years_cumulative"]
    )
    df["growth_rate_10_vs_15"] = df["growth_10_vs_15"] / df[
        "citations_15_years_cumulative"
    ].replace(0, 1)

    # 构建分析结果列
    growth_columns = [
        "growth_rate_30_vs_50",
        "growth_rate_25_vs_30",
        "growth_rate_20_vs_25",
        "growth_rate_15_vs_20",
        "growth_rate_10_vs_15",
    ]

    return growth_columns, df


growth_analysis_columns, df = generate_cumulative_growth_analysis(df)

years_before_award_columns, df = generate_cumulative_citations_columns(df)


def generate_yearly_before_award_work_citations(df, max_years=50):
    """为每个人的top1-top5作品生成获奖前第1年到第n年的引用数据"""

    def get_work_citations_for_years_before_award(row):
        """获取获奖前指定年数的作品引用数据"""
        award_age = row["award_age"]
        result = {}

        # 为每个top_k (1-5) 生成获奖前第1年到第max_years年的引用数据
        for top_k in range(1, 6):
            top_age_to_col = top_to_age_to_col.get(top_k, {})

            for years_before in range(1, max_years + 1):
                target_age = award_age - years_before
                col_name = top_age_to_col.get(target_age)

                if col_name and pd.notna(row[col_name]):
                    result[
                        f"top{top_k}_citations_{years_before}_years_before_award"
                    ] = row[col_name]
                else:
                    result[
                        f"top{top_k}_citations_{years_before}_years_before_award"
                    ] = 0

        return pd.Series(result)

    yearly_work_citations = df.apply(get_work_citations_for_years_before_award, axis=1)

    for col in yearly_work_citations.columns:
        df[col] = yearly_work_citations[col]

    return df


def generate_cumulative_work_citations_columns(df):
    """为work_year_cols生成累积引用列：1-19年独立保留，20-25年聚合，26-35年聚合，36-50年聚合"""

    # 获取所有获奖前作品引用列
    work_citation_cols = [
        col
        for col in df.columns
        if re.match(r"top\d+_citations_\d+_years_before_award$", col)
    ]

    # 按top_k和年数排序
    def sort_key(col):
        top_match = re.search(r"top(\d+)_citations_(\d+)_years_before_award", col)
        return (int(top_match.group(1)), int(top_match.group(2)))

    work_citation_cols_sorted = sorted(work_citation_cols, key=sort_key)

    final_columns = []

    # 为每个top_k处理
    for top_k in range(1, 6):
        # 获取该top_k的所有列
        top_cols = [
            col
            for col in work_citation_cols_sorted
            if col.startswith(f"top{top_k}_citations_")
        ]

        # 分类列
        cols_1_to_19 = []
        cols_20_to_25 = []
        cols_26_to_35 = []
        cols_36_to_50 = []

        for col in top_cols:
            years_before = int(re.search(r"(\d+)_years_before_award", col).group(1))
            if 1 <= years_before <= 19:
                cols_1_to_19.append(col)
            elif 20 <= years_before <= 25:
                cols_20_to_25.append(col)
            elif 26 <= years_before <= 35:
                cols_26_to_35.append(col)
            elif 36 <= years_before <= 50:
                cols_36_to_50.append(col)

        # 计算累积列
        if cols_20_to_25:
            df[f"top{top_k}_citations_20_to_25_years_before_cumulative"] = df[
                cols_20_to_25
            ].sum(axis=1)
            final_columns.append(
                f"top{top_k}_citations_20_to_25_years_before_cumulative"
            )

        if cols_26_to_35:
            df[f"top{top_k}_citations_26_to_35_years_before_cumulative"] = df[
                cols_26_to_35
            ].sum(axis=1)
            final_columns.append(
                f"top{top_k}_citations_26_to_35_years_before_cumulative"
            )

        if cols_36_to_50:
            df[f"top{top_k}_citations_36_to_50_years_before_cumulative"] = df[
                cols_36_to_50
            ].sum(axis=1)
            final_columns.append(
                f"top{top_k}_citations_36_to_50_years_before_cumulative"
            )

        # 添加获奖前1-19年的列（独立保留）
        final_columns.extend(cols_1_to_19)

    return final_columns, df


def generate_cumulative_work_growth_analysis(df):
    """为work_year_cols生成累积引用增长分析：从50年开始按时间段累积到获奖前"""

    # 获取所有获奖前作品引用列
    work_citation_cols = [
        col
        for col in df.columns
        if re.match(r"top\d+_citations_\d+_years_before_award$", col)
    ]

    # 按top_k和年数排序
    def sort_key(col):
        top_match = re.search(r"top(\d+)_citations_(\d+)_years_before_award", col)
        return (int(top_match.group(1)), int(top_match.group(2)))

    work_citation_cols_sorted = sorted(work_citation_cols, key=sort_key)

    growth_columns = []

    # 为每个top_k处理
    for top_k in range(1, 6):
        # 获取该top_k的所有列
        top_cols = [
            col
            for col in work_citation_cols_sorted
            if col.startswith(f"top{top_k}_citations_")
        ]

        # 定义时间段边界
        time_periods = [
            (
                50,
                31,
                f"top{top_k}_citations_31_to_50_years_before_cumulative",
            ),  # 50-31年累积
            (
                30,
                26,
                f"top{top_k}_citations_26_to_30_years_before_cumulative",
            ),  # 30-26年累积
            (
                25,
                21,
                f"top{top_k}_citations_21_to_25_years_before_cumulative",
            ),  # 25-21年累积
            (
                20,
                16,
                f"top{top_k}_citations_16_to_20_years_before_cumulative",
            ),  # 20-16年累积
            (
                15,
                11,
                f"top{top_k}_citations_11_to_15_years_before_cumulative",
            ),  # 15-11年累积
            (
                10,
                1,
                f"top{top_k}_citations_1_to_10_years_before_cumulative",
            ),  # 10-1年累积
        ]

        # 计算各时间段的累积引用
        for start_year, end_year, col_name in time_periods:
            period_cols = []
            for col in top_cols:
                years_before = int(re.search(r"(\d+)_years_before_award", col).group(1))
                if end_year <= years_before <= start_year:
                    period_cols.append(col)

            if period_cols:
                df[col_name] = df[period_cols].sum(axis=1)

        # 计算累积增长情况（从远期到近期）
        # 50年总累积
        df[f"top{top_k}_citations_50_years_total_cumulative"] = df[
            f"top{top_k}_citations_31_to_50_years_before_cumulative"
        ]

        # 30年累积（31-50年 + 26-30年）
        df[f"top{top_k}_citations_30_years_cumulative"] = (
            df[f"top{top_k}_citations_31_to_50_years_before_cumulative"]
            + df[f"top{top_k}_citations_26_to_30_years_before_cumulative"]
        )

        # 25年累积（31-50年 + 26-30年 + 21-25年）
        df[f"top{top_k}_citations_25_years_cumulative"] = (
            df[f"top{top_k}_citations_31_to_50_years_before_cumulative"]
            + df[f"top{top_k}_citations_26_to_30_years_before_cumulative"]
            + df[f"top{top_k}_citations_21_to_25_years_before_cumulative"]
        )

        # 20年累积（31-50年 + 26-30年 + 21-25年 + 16-20年）
        df[f"top{top_k}_citations_20_years_cumulative"] = (
            df[f"top{top_k}_citations_31_to_50_years_before_cumulative"]
            + df[f"top{top_k}_citations_26_to_30_years_before_cumulative"]
            + df[f"top{top_k}_citations_21_to_25_years_before_cumulative"]
            + df[f"top{top_k}_citations_16_to_20_years_before_cumulative"]
        )

        # 15年累积（31-50年 + 26-30年 + 21-25年 + 16-20年 + 11-15年）
        df[f"top{top_k}_citations_15_years_cumulative"] = (
            df[f"top{top_k}_citations_31_to_50_years_before_cumulative"]
            + df[f"top{top_k}_citations_26_to_30_years_before_cumulative"]
            + df[f"top{top_k}_citations_21_to_25_years_before_cumulative"]
            + df[f"top{top_k}_citations_16_to_20_years_before_cumulative"]
            + df[f"top{top_k}_citations_11_to_15_years_before_cumulative"]
        )

        # 10年累积（31-50年 + 26-30年 + 21-25年 + 16-20年 + 11-15年 + 1-10年）
        df[f"top{top_k}_citations_10_years_cumulative"] = (
            df[f"top{top_k}_citations_31_to_50_years_before_cumulative"]
            + df[f"top{top_k}_citations_26_to_30_years_before_cumulative"]
            + df[f"top{top_k}_citations_21_to_25_years_before_cumulative"]
            + df[f"top{top_k}_citations_16_to_20_years_before_cumulative"]
            + df[f"top{top_k}_citations_11_to_15_years_before_cumulative"]
            + df[f"top{top_k}_citations_1_to_10_years_before_cumulative"]
        )

        # 计算增长率和增长量
        df[f"top{top_k}_growth_30_vs_50"] = (
            df[f"top{top_k}_citations_30_years_cumulative"]
            - df[f"top{top_k}_citations_50_years_total_cumulative"]
        )
        df[f"top{top_k}_growth_rate_30_vs_50"] = df[f"top{top_k}_growth_30_vs_50"] / df[
            f"top{top_k}_citations_50_years_total_cumulative"
        ].replace(0, 1)

        df[f"top{top_k}_growth_25_vs_30"] = (
            df[f"top{top_k}_citations_25_years_cumulative"]
            - df[f"top{top_k}_citations_30_years_cumulative"]
        )
        df[f"top{top_k}_growth_rate_25_vs_30"] = df[f"top{top_k}_growth_25_vs_30"] / df[
            f"top{top_k}_citations_30_years_cumulative"
        ].replace(0, 1)

        df[f"top{top_k}_growth_20_vs_25"] = (
            df[f"top{top_k}_citations_20_years_cumulative"]
            - df[f"top{top_k}_citations_25_years_cumulative"]
        )
        df[f"top{top_k}_growth_rate_20_vs_25"] = df[f"top{top_k}_growth_20_vs_25"] / df[
            f"top{top_k}_citations_25_years_cumulative"
        ].replace(0, 1)

        df[f"top{top_k}_growth_15_vs_20"] = (
            df[f"top{top_k}_citations_15_years_cumulative"]
            - df[f"top{top_k}_citations_20_years_cumulative"]
        )
        df[f"top{top_k}_growth_rate_15_vs_20"] = df[f"top{top_k}_growth_15_vs_20"] / df[
            f"top{top_k}_citations_20_years_cumulative"
        ].replace(0, 1)

        df[f"top{top_k}_growth_10_vs_15"] = (
            df[f"top{top_k}_citations_10_years_cumulative"]
            - df[f"top{top_k}_citations_15_years_cumulative"]
        )
        df[f"top{top_k}_growth_rate_10_vs_15"] = df[f"top{top_k}_growth_10_vs_15"] / df[
            f"top{top_k}_citations_15_years_cumulative"
        ].replace(0, 1)

        # 添加增长率列到结果
        growth_columns.extend(
            [
                f"top{top_k}_growth_rate_30_vs_50",
                f"top{top_k}_growth_rate_25_vs_30",
                f"top{top_k}_growth_rate_20_vs_25",
                f"top{top_k}_growth_rate_15_vs_20",
                f"top{top_k}_growth_rate_10_vs_15",
            ]
        )

    return growth_columns, df


# 为work_year_cols生成获奖前引用数据
df = generate_yearly_before_award_work_citations(df, max_years=50)

# 生成work_year_cols的累积增长分析
work_growth_analysis_columns, df = generate_cumulative_work_growth_analysis(df)

# 生成work_year_cols的累积列并获取列列表
work_years_before_award_columns, df = generate_cumulative_work_citations_columns(df)


def extract_institutions(row: Dict[str, Any]) -> List[str]:
    """
    从单行数据中提取所有机构名称（来自education和career字段）
    """
    institutions = set()

    # 从education字段提取机构
    if "extraction" in row and "education" in row["extraction"]:
        education_data = row["extraction"]["education"]
        if isinstance(education_data, list):
            for edu in education_data:
                if isinstance(edu, dict) and "institution" in edu:
                    institution = edu["institution"]
                    if institution and institution.strip():
                        institutions.add(institution.strip())

    # 从career字段提取机构
    if "extraction" in row and "career" in row["extraction"]:
        career_data = row["extraction"]["career"]
        if isinstance(career_data, list):
            for career in career_data:
                if isinstance(career, dict) and "institution" in career:
                    institution = career["institution"]
                    if institution and institution.strip():
                        institutions.add(institution.strip())

    return list(institutions)


wiki_df = pd.read_json("data/nobel_dataset_wiki_sampled_100.jsonl", lines=True)

wiki_df["institutions"] = wiki_df.apply(extract_institutions, axis=1)


def disambiguate_and_onehot_institutions(
    institutions_list: List[str],
) -> Dict[str, int]:
    """
    对机构列表进行消歧义和onehot编码
    返回字典，键为机构名称，值为出现次数
    """
    # 先通过institution_mapping进行消歧义
    disambiguated = [institution_mapping.get(inst, inst) for inst in institutions_list]

    # 计算每个机构出现的次数
    institution_counts = {}
    for inst in disambiguated:
        institution_counts[inst] = institution_counts.get(inst, 0) + 1

    # 创建onehot向量，基于institution_onehot中的机构
    onehot_result = {}
    for inst in institution_onehot:
        onehot_result[inst] = institution_counts.get(inst, 0)

    return onehot_result


# 对每行的机构列表进行消歧义和onehot编码
wiki_df["institution_onehot"] = wiki_df["institutions"].apply(
    disambiguate_and_onehot_institutions
)

# 将onehot结果展开为单独的列
onehot_df = pd.DataFrame(wiki_df["institution_onehot"].tolist())
wiki_df = pd.concat([wiki_df, onehot_df], axis=1)

# df与wiki_df通过名字merge
df = pd.merge(
    df,
    wiki_df[["original_name"] + onehot_df.columns.tolist()],
    on="original_name",
    how="left",
)
