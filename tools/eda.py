import pandas as pd
import re
import numpy as np
import os
import matplotlib.pyplot as plt
from pandas.api.types import is_numeric_dtype

from features import (df, 
    citations_total_cols,
    years_before_award_columns,
    growth_analysis_columns,
    work_growth_analysis_columns,
    work_years_before_award_columns)  

try:
    from features import categorical_cols 
except Exception:
    categorical_cols = []
try:
    from features import citations_year_cols
except Exception:
    citations_year_cols = []
try:
    from features import work_year_cols 
except Exception:
    work_year_cols = []

import warnings
warnings.filterwarnings('ignore')

df_full = df.copy()

df = df[~df['list_lable'].isin([2])]
def plot_columns_by_groups_compare(
    df_all: pd.DataFrame,
    groups: dict,
    base_out_dir: str = 'eda_compare',
    top_n_cats: int = 30
) -> None:
    """
    仅针对传入的列分组绘图。每个分组输出到 base_out_dir/<group_name>/ 下：
      - 数值列: 叠加直方图（密度标准化）
      - 分类型/文本列: 并列柱状图（Top N 类按比例）
    """
    os.makedirs(base_out_dir, exist_ok=True)


    df_A = df_all[~df_all['list_lable'].isin([2])].copy() 
    df_B = df_all[~df_all['list_lable'].isin([1, 2])].copy() 

    for group_name, cols in groups.items():
        out_dir = os.path.join(base_out_dir, group_name)
        os.makedirs(out_dir, exist_ok=True)
        for col in cols:
            if col not in df_all.columns:
                continue
            try:
                series_A = df_A[col]
                series_B = df_B[col]

                col_is_numeric = is_numeric_dtype(series_A) and is_numeric_dtype(series_B)
                if not col_is_numeric:
                    sA_num = pd.to_numeric(series_A, errors='coerce')
                    sB_num = pd.to_numeric(series_B, errors='coerce')
                    valid_ratio = ((sA_num.notna().mean() + sB_num.notna().mean()) / 2.0)
                    if valid_ratio >= 0.7:
                        col_is_numeric = True
                        series_A, series_B = sA_num, sB_num

                if col_is_numeric:
                    A_vals = pd.to_numeric(series_A, errors='coerce').replace([np.inf, -np.inf], np.nan).dropna()
                    B_vals = pd.to_numeric(series_B, errors='coerce').replace([np.inf, -np.inf], np.nan).dropna()
                    if len(A_vals) == 0 and len(B_vals) == 0:
                        plt.close()
                        continue
                    combined = pd.concat([A_vals, B_vals])
                    try:
                        bins = np.histogram_bin_edges(combined, bins=50)
                    except Exception:
                        bins = 50
                    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True, sharey=True)
                    axes[0].hist(A_vals, bins=bins, alpha=0.8, density=True, color="#1f77b4")
                    axes[0].set_title('~isin([2])')
                    axes[0].set_ylabel('Density')
                    axes[0].set_xlabel(col)
                    axes[1].hist(B_vals, bins=bins, alpha=0.8, density=True, color="#d62728")
                    axes[1].set_title('~isin([1,2])')
                    axes[1].set_xlabel(col)
                    fig.suptitle(f'[{group_name}] Distribution (numeric): {col}', y=1.02, fontsize=11)
                    fig.tight_layout()
                else:
                    A_c = series_A.astype('object').fillna('NaN').astype(str).value_counts(normalize=True)
                    B_c = series_B.astype('object').fillna('NaN').astype(str).value_counts(normalize=True)
                    all_c = (series_A.astype('object').fillna('NaN').astype(str)
                             .append(series_B.astype('object').fillna('NaN').astype(str)))
                    top_cats = all_c.value_counts().head(top_n_cats).index
                    A_plot = A_c.reindex(top_cats, fill_value=0.0)
                    B_plot = B_c.reindex(top_cats, fill_value=0.0)
                    x = np.arange(len(top_cats))
                    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
                    axes[0].bar(x, A_plot.values, color="#1f77b4")
                    axes[0].set_title('~isin([2])')
                    axes[0].set_ylabel('Proportion')
                    axes[0].set_xticks(x)
                    axes[0].set_xticklabels(top_cats, rotation=45, ha='right')
                    axes[1].bar(x, B_plot.values, color="#d62728")
                    axes[1].set_title('~isin([1,2])')
                    axes[1].set_xticks(x)
                    axes[1].set_xticklabels(top_cats, rotation=45, ha='right')
                    ymax = max(axes[0].get_ylim()[1], axes[1].get_ylim()[1])
                    axes[0].set_ylim(0, ymax)
                    axes[1].set_ylim(0, ymax)
                    fig.suptitle(f'[{group_name}] Distribution (categorical): {col}', y=1.02, fontsize=11)
                    fig.tight_layout()

                safe_name = re.sub(r'[^\w\-\.]+', '_', str(col))
                safe_out_path = os.path.join(out_dir, f"compare_{safe_name}.png")
                if 'fig' in locals():
                    fig.savefig(safe_out_path, dpi=160)
                    plt.close(fig)
                else:
                    plt.savefig(safe_out_path, dpi=160)
                    plt.close()
            except Exception as e:
                print(f"[WARN] 列 {col} 绘图失败: {e}")

# 运行对比绘图
print("\n开始生成指定列分组的分布对比图：~isin([2]) vs ~isin([1,2]) ...")
groups_to_plot = {
    'citations_total_cols': citations_total_cols,
    'years_before_award_columns': years_before_award_columns,
    'growth_analysis_columns': growth_analysis_columns,
    'work_growth_analysis_columns': work_growth_analysis_columns,
    'work_years_before_award_columns': work_years_before_award_columns,
}
plot_columns_by_groups_compare(df_full, groups_to_plot, base_out_dir='eda_compare')
print("分布对比图已生成到目录: eda_compare/<group_name>/")

# =========================
# Top1-Top5前20年引用空值分析
# =========================

print("\n[Top1-Top5前20年引用空值分析] 统计每行前20年引用都为空的情况:")

# 获取所有work_citation列
work_citation_cols = [col for col in df.columns if re.match(r'top\d+_citations_\d+_years_before_award$', col)]

# 按top_k和年数排序
def sort_key(col):
    top_match = re.search(r'top(\d+)_citations_(\d+)_years_before_award', col)
    return (int(top_match.group(1)), int(top_match.group(2)))

work_citation_cols_sorted = sorted(work_citation_cols, key=sort_key)

# 为每个top_k处理
empty_analysis_results = []

for top_k in range(1, 6):
    # 获取该top_k的所有列
    top_cols = [col for col in work_citation_cols_sorted if col.startswith(f'top{top_k}_citations_')]
    
    # 分类列
    cols_1_to_19 = []
    cols_20_to_25 = []
    cols_26_to_35 = []
    cols_36_to_50 = [] 
    
    for col in top_cols:
        years_before = int(re.search(r'(\d+)_years_before_award', col).group(1))
        if 1 <= years_before <= 19:
            cols_1_to_19.append(col)
        elif 20 <= years_before <= 25:
            cols_20_to_25.append(col)
        elif 26 <= years_before <= 35:
            cols_26_to_35.append(col)
        elif 36 <= years_before <= 50:
            cols_36_to_50.append(col)
    
    # 检查前20年（1-19年）的列是否都为空
    if cols_1_to_19:
        # 检查每行前20年列是否都为空（0或NaN）
        empty_mask = (df[cols_1_to_19] == 0) | df[cols_1_to_19].isna()
        all_empty_mask = empty_mask.all(axis=1)
        
        # 统计空值情况
        total_rows = len(df)
        empty_rows = all_empty_mask.sum()
        empty_percentage = (empty_rows / total_rows) * 100
        
        empty_analysis_results.append({
            'top_k': top_k,
            'total_rows': total_rows,
            'empty_rows': empty_rows,
            'empty_percentage': empty_percentage,
            'cols_1_to_19_count': len(cols_1_to_19)
        })
        
        print(f"\nTop{top_k} 前20年引用空值统计:")
        print(f"  总行数: {total_rows}")
        print(f"  前20年都为空的行数: {empty_rows}")
        print(f"  空值比例: {empty_percentage:.2f}%")
        print(f"  前20年列数: {len(cols_1_to_19)}")

# 创建空值分析结果DataFrame
empty_analysis_df = pd.DataFrame(empty_analysis_results)
print("\n[空值分析汇总] Top1-Top5前20年引用空值统计:")
print(empty_analysis_df)

# 找出所有top1-top5前20年都为空的行
print("\n[详细分析] 所有Top1-Top5前20年引用都为空的行:")

# 收集所有top_k的前20年列
all_cols_1_to_19 = []
for top_k in range(1, 6):
    top_cols = [col for col in work_citation_cols_sorted if col.startswith(f'top{top_k}_citations_')]
    for col in top_cols:
        years_before = int(re.search(r'(\d+)_years_before_award', col).group(1))
        if 1 <= years_before <= 19:
            all_cols_1_to_19.append(col)

# 检查所有前20年列是否都为空
if all_cols_1_to_19:
    all_empty_mask = ((df[all_cols_1_to_19] == 0) | df[all_cols_1_to_19].isna()).all(axis=1)
    
    # 获取所有前20年都为空的行
    empty_rows_df = df[all_empty_mask].copy()
    
    if not empty_rows_df.empty:
        # 创建结果DataFrame，包含名字，国家，生日，得奖年龄，以及5个空值情况
        result_df = empty_rows_df[['name', 'origin_country', 'birth_year', 'award_age']].copy()
        
        # 为每个top_k添加空值标记
        for top_k in range(1, 6):
            top_cols = [col for col in work_citation_cols_sorted if col.startswith(f'top{top_k}_citations_')]
            cols_1_to_19 = []
            for col in top_cols:
                years_before = int(re.search(r'(\d+)_years_before_award', col).group(1))
                if 1 <= years_before <= 19:
                    cols_1_to_19.append(col)
            
            if cols_1_to_19:
                empty_mask = (df[cols_1_to_19] == 0) | df[cols_1_to_19].isna()
                all_empty_mask = empty_mask.all(axis=1)
                result_df[f'top{top_k}_empty'] = all_empty_mask[all_empty_mask].astype(int)
        
        print(f"找到 {len(result_df)} 行所有Top1-Top5前20年引用都为空:")
        print(result_df.head(20)) 
        
    else:
        print("没有找到所有Top1-Top5前20年引用都为空的行")
else:
    print("没有找到前20年的引用列")

# =========================
# Top5中3个前20年都为空的行数统计
# =========================

print("\n[Top5中3个前20年都为空统计] 统计同一行中Top5里面有3个前20年都为空的行数:")

# 为每个top_k计算前20年都为空的情况
top_empty_status = {}
for top_k in range(1, 6):
    top_cols = [col for col in work_citation_cols_sorted if col.startswith(f'top{top_k}_citations_')]
    cols_1_to_19 = []
    for col in top_cols:
        years_before = int(re.search(r'(\d+)_years_before_award', col).group(1))
        if 1 <= years_before <= 19:
            cols_1_to_19.append(col)
    
    if cols_1_to_19:
        empty_mask = (df[cols_1_to_19] == 0) | df[cols_1_to_19].isna()
        all_empty_mask = empty_mask.all(axis=1)
        top_empty_status[f'top{top_k}'] = all_empty_mask

# 创建DataFrame来统计每行的空值情况
empty_status_df = pd.DataFrame(top_empty_status)

# 计算每行有多少个top_k前20年都为空
empty_count_per_row = empty_status_df.sum(axis=1)

# 统计不同空值个数的行数
empty_count_stats = empty_count_per_row.value_counts().sort_index()

print("\n[空值个数分布] 每行Top5中前20年都为空的数量分布:")
for count, rows in empty_count_stats.items():
    print(f"  {count}个前20年都为空: {rows} 行")

# 特别关注3个前20年都为空的行
three_empty_rows = empty_count_per_row >= 4
three_empty_count = three_empty_rows.sum()

print(f"\n[重点统计] Top5中有3个前20年都为空的行数: {three_empty_count}")

if three_empty_count > 0:
    # 显示这些行的详细信息
    three_empty_indices = empty_count_per_row[three_empty_rows].index
    three_empty_details = df.loc[three_empty_indices, ['original_name', 'origin_country', 'birth_year', 'award_age']].copy()
    
    # 添加每个top_k的空值状态
    for top_k in range(1, 6):
        three_empty_details[f'top{top_k}_empty'] = top_empty_status[f'top{top_k}'][three_empty_indices].astype(int)
    
    print("\n[详细信息] 3个前20年都为空的行详情 (前20行):")
    print(three_empty_details.head(20))
    
    # 统计这3个空值在top1-top5中的分布
    print(f"\n[空值分布] 在{three_empty_count}行中，各top_k前20年都为空的情况:")
    for top_k in range(1, 6):
        count = three_empty_details[f'top{top_k}_empty'].sum()
        print(f"  Top{top_k}: {count} 行")




# =========================
# 国家分布分析
# =========================

# =========================
# 性别分布分析
# =========================

print("=== label=0和1在不同性别的分布 ===\n")

# 性别分布统计
gender_ct = pd.crosstab(df['gender'], df['label'])
# 确保label=0/1两列都存在
for lbl in [0, 1]:
    if lbl not in gender_ct.columns:
        gender_ct[lbl] = 0
gender_ct = gender_ct[[0, 1]]  # 确保列顺序为0,1
print("[性别分布] gender:")
print(gender_ct)


print("=== label=0和1在不同国家的分布 ===\n")

# 1. 原始国家分布
print("[原始国家分布] origin_country:")
origin_ct = pd.crosstab(df['origin_country'], df['label'])
# 确保label=0/1两列都存在
for lbl in [0, 1]:
    if lbl not in origin_ct.columns:
        origin_ct[lbl] = 0
origin_ct = origin_ct[[0, 1]]  # 确保列顺序为0,1
print(origin_ct)

# 2. 当前国家分布  
print("\n[当前国家分布] current_country:")
current_ct = pd.crosstab(df['current_country'], df['label'])
# 确保label=0/1两列都存在
for lbl in [0, 1]:
    if lbl not in current_ct.columns:
        current_ct[lbl] = 0
current_ct = current_ct[[0, 1]]  # 确保列顺序为0,1
print(current_ct)



# =========================
# Exploratory Data Analysis (tabular only, no plots)
# =========================

print('\n[人数统计] 按 label 分组与总人数:')
label_counts = df['label'].value_counts()
total_count = len(df)
count_summary = pd.DataFrame({
    '总体': total_count,
    '得奖(label=1)': label_counts.get(1, 0),
    '不得奖(label=0)': label_counts.get(0, 0)
}, index=['count'])
print(count_summary)


print('\n[award_age] 统计特征汇总:')

total_count = len(df['award_age'])
inf_count = np.sum(np.isinf(df['award_age']))
nan_count = np.sum(pd.isna(df['award_age']))

label_1_inf_count = np.sum(np.isinf(df[df['label']==1]['award_age']))
label_1_nan_count = np.sum(pd.isna(df[df['label']==1]['award_age']))
label_0_inf_count = np.sum(np.isinf(df[df['label']==0]['award_age'])) 
label_0_nan_count = np.sum(pd.isna(df[df['label']==0]['award_age']))

award_age_clean = df['award_age'].replace([np.inf, -np.inf], np.nan)

# 计算总体统计
overall_stats = award_age_clean.describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95])

# 计算按label分组的统计
label_1_stats = df[df['label'] == 1]['award_age'].replace([np.inf, -np.inf], np.nan).describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95])
label_0_stats = df[df['label'] == 0]['award_age'].replace([np.inf, -np.inf], np.nan).describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95])
label_1_stats['inf_count'] = label_1_inf_count
label_1_stats['nan_count'] = label_1_nan_count
label_0_stats['inf_count'] = label_0_inf_count
label_0_stats['nan_count'] = label_0_nan_count


overall_stats['inf_count'] = inf_count
overall_stats['nan_count'] = nan_count 
award_age_summary = pd.DataFrame({
    'overall': overall_stats,
    '(label=1)': label_1_stats,
    '(label=0)': label_0_stats
})

print(award_age_summary)

print('\n[指数指标] 统计特征汇总:')
cit_df_clean = df[citations_total_cols].copy()
for c in citations_total_cols:
    if c in cit_df_clean.columns:
        cit_df_clean[c] = pd.to_numeric(cit_df_clean[c], errors='coerce')
cit_df_clean = cit_df_clean.replace([np.inf, -np.inf], np.nan)

# 计算总体统计
overall_cit_stats = cit_df_clean.describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95])

# 计算按label分组的统计
label_1_cit_stats = df[df['label'] == 1][citations_total_cols].apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan).describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95])
label_0_cit_stats = df[df['label'] == 0][citations_total_cols].apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan).describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95])

# 为每个指标分别创建统计汇总，然后横向拼接
cit_summaries = []
for col in citations_total_cols:
    if col in overall_cit_stats.columns:
        col_summary = pd.DataFrame({
            f'overall_{col}': overall_cit_stats[col],
            f'(label=1)_{col}': label_1_cit_stats[col],
            f'(label=0)_{col}': label_0_cit_stats[col]
        })
        cit_summaries.append(col_summary)
# 横向合并所有指标的统计结果
cit_summary = pd.concat(cit_summaries, axis=1)
print(cit_summary)


# 对 citations_year_cols（如 author_yearly_json_age_78）基于 award_age 对齐到相对年（0 表示获奖年，- 表示获奖前年，+ 表示获奖后年），
# 并以每隔 5 年为一个区间，计算5年增长率 (later - earlier) / |earlier|。
print('\n[逐年引文相关统计] 基于相对获奖年，每5年增长率:')

def extract_age_from_col(col_name: str) -> int:
    m = re.search(r'age_(\d+)$', col_name)
    return int(m.group(1)) if m else None

# 建立 age -> 列名 的映射，只对 citations_year_cols
age_to_col = {}
for c in citations_year_cols:
    age = extract_age_from_col(c)
    if age is not None:
        age_to_col[age] = c

growth_records = []

for idx, row in df.iterrows():
    award_age_value = row.get('award_age', np.nan)
    if pd.isna(award_age_value):
        continue
    try:
        award_age_int = int(award_age_value)
    except Exception:
        continue

    # 组装相对年 -> 引文值 的序列
    rel_to_value = {}
    for age, col_name in age_to_col.items():
        val = pd.to_numeric(row.get(col_name, np.nan), errors='coerce')
        if pd.isna(val):
            continue
        rel_year = age - award_age_int 
        rel_to_value[rel_year] = val

    if not rel_to_value:
        continue

    available_rels = sorted(rel_to_value.keys())
    min_rel, max_rel = available_rels[0], available_rels[-1]

    # 当前行的标签
    row_label = row.get('label', np.nan)

    # 向前（负方向）：(-5 -> 0), (-10 -> -5), ...
    rel_end = 0
    while rel_end - 5 >= min_rel:
        rel_start = rel_end - 5
        v_start = rel_to_value.get(rel_start, np.nan)
        v_end = rel_to_value.get(rel_end, np.nan)
        rate = np.nan
        if not pd.isna(v_start) and not pd.isna(v_end) and v_start != 0:
            rate = (v_end - v_start) / abs(v_start)
        growth_records.append({
            'interval': f'{rel_start}->{rel_end}',
            'rel_start': rel_start,
            'rel_end': rel_end,
            'rate': rate,
            'label': row_label
        })
        rel_end -= 5

    # 向后（正方向）：(0 -> +5), (+5 -> +10), ...
    rel_start = 0
    while rel_start + 5 <= max_rel:
        rel_end = rel_start + 5
        v_start = rel_to_value.get(rel_start, np.nan)
        v_end = rel_to_value.get(rel_end, np.nan)
        rate = np.nan
        if not pd.isna(v_start) and not pd.isna(v_end) and v_start != 0:
            rate = (v_end - v_start) / abs(v_start)
        growth_records.append({
            'interval': f'{rel_start}->{rel_end}',
            'rel_start': rel_start,
            'rel_end': rel_end,
            'rate': rate,
            'label': row_label
        })
        rel_start += 5

yearly_growth_5y = pd.DataFrame(growth_records)
if not yearly_growth_5y.empty:
    print('样例（前20行）：')
    print(yearly_growth_5y.head(20))
    growth_mean_by_interval = yearly_growth_5y.groupby('interval', as_index=False)['rate'].mean().sort_values(
        by=['interval'], key=lambda s: s.str.extract(r'(-?\d+)->(-?\d+)').astype(int).apply(tuple, axis=1)
    )
    print('\n各区间平均增长率:')
    print(growth_mean_by_interval)

    # 按 label 拆分统计：均值、最大值、75分位、空值率
    def p75(series: pd.Series) -> float:
        return series.quantile(0.75)

    stats_by_label = yearly_growth_5y.groupby(['interval', 'label']).agg(
        mean_rate=('rate', 'mean'),
        max_rate=('rate', 'max'),
        p75_rate=('rate', p75),
        null_rate=('rate', lambda s: s.isna().mean())
    ).reset_index()

    # 透视成每个 interval 一行，label=0/1 各出4列
    stats_pivot = stats_by_label.pivot(index='interval', columns='label', values=['mean_rate', 'max_rate', 'p75_rate', 'null_rate'])
    # 扁平列名：('mean_rate', 0) -> label_0_mean_rate
    stats_pivot.columns = [f"label_{int(lbl)}_{metric}" for metric, lbl in stats_pivot.columns]
    stats_pivot = stats_pivot.reset_index().sort_values(
        by=['interval'], key=lambda s: s.str.extract(r'(-?\d+)->(-?\d+)').astype(int).apply(tuple, axis=1)
    )
    print('\n按 label 拆分的区间统计（均值/最大/75分位/空值率）：')
    print(stats_pivot)
else:
    print('无法计算增长率（缺少连续5年数据或有效数值）')

# 4.1) Top1~Top5 作品逐年统计（work_year_cols）基于相对获奖年，每5年增长率，长表输出
print('\n[逐年作品相关统计] 基于相对获奖年，每5年增长率:')

def extract_top_and_age(col_name: str):
    m = re.search(r'top(\d+)_yearly_json_age_(\d+)$', col_name)
    if not m:
        return None, None
    return int(m.group(1)), int(m.group(2))

# 构建 top_k -> (age -> col) 的映射
top_to_age_to_col = {}
for c in work_year_cols:
    top_k, age = extract_top_and_age(c)
    if top_k is None or age is None:
        continue
    if top_k not in top_to_age_to_col:
        top_to_age_to_col[top_k] = {}
    top_to_age_to_col[top_k][age] = c

work_growth_records = []

for idx, row in df.iterrows():
    award_age_value = row.get('award_age', np.nan)
    if pd.isna(award_age_value):
        continue
    try:
        award_age_int = int(award_age_value)
    except Exception:
        continue

    row_label = row.get('label', np.nan)

    # 对每个 top_k 单独计算相对年序列与区间增长率
    for top_k, age_to_col_map in top_to_age_to_col.items():
        rel_to_value = {}
        for age, col_name in age_to_col_map.items():
            val = pd.to_numeric(row.get(col_name, np.nan), errors='coerce')
            if pd.isna(val):
                continue
            rel_year = age - award_age_int
            rel_to_value[rel_year] = val

        if not rel_to_value:
            continue

        available_rels = sorted(rel_to_value.keys())
        min_rel, max_rel = available_rels[0], available_rels[-1]

        # 负向区间：(-5 -> 0), (-10 -> -5), ...
        rel_end = 0
        while rel_end - 5 >= min_rel:
            rel_start = rel_end - 5
            v_start = rel_to_value.get(rel_start, np.nan)
            v_end = rel_to_value.get(rel_end, np.nan)
            rate = np.nan
            if not pd.isna(v_start) and not pd.isna(v_end) and v_start != 0:
                rate = (v_end - v_start) / abs(v_start)
            work_growth_records.append({
                'top_k': top_k,
                'interval': f'{rel_start}->{rel_end}',
                'rel_start': rel_start,
                'rel_end': rel_end,
                'rate': rate,
                'label': row_label
            })
            rel_end -= 5

        # 正向区间：(0 -> +5), (+5 -> +10), ...
        rel_start = 0
        while rel_start + 5 <= max_rel:
            rel_end = rel_start + 5
            v_start = rel_to_value.get(rel_start, np.nan)
            v_end = rel_to_value.get(rel_end, np.nan)
            rate = np.nan
            if not pd.isna(v_start) and not pd.isna(v_end) and v_start != 0:
                rate = (v_end - v_start) / abs(v_start)
            work_growth_records.append({
                'top_k': top_k,
                'interval': f'{rel_start}->{rel_end}',
                'rel_start': rel_start,
                'rel_end': rel_end,
                'rate': rate,
                'label': row_label
            })
            rel_start += 5

work_yearly_growth_5y = pd.DataFrame(work_growth_records)
if not work_yearly_growth_5y.empty:
    print('样例（前20行）：')
    print(work_yearly_growth_5y.head(20))
    # 可按需汇总：例如按 interval、top_k、label 统计
    work_stats = work_yearly_growth_5y.groupby(['interval', 'top_k', 'label'], as_index=False)['rate'].mean()
    work_stats = work_stats.sort_values(
        by=['interval', 'top_k'],
        key=lambda s: s.str.extract(r'(-?\d+)->(-?\d+)').astype(int).apply(tuple, axis=1) if s.name == 'interval' else s
    )
    print('\n各区间×TopK×label 的平均增长率:')
    print(work_stats)
    # 生成 5*2*4 列的透视表：top1..top5 × label(0/1) × (mean,max,p75,null)
    def p75(series: pd.Series) -> float:
        return series.quantile(0.75)

    work_stats_full = work_yearly_growth_5y.groupby(['interval', 'top_k', 'label']).agg(
        mean_rate=('rate', 'mean'),
        max_rate=('rate', 'max'),
        p75_rate=('rate', p75),
        null_rate=('rate', lambda s: s.isna().mean())
    ).reset_index()

    work_stats_pivot = work_stats_full.pivot(index='interval', columns=['top_k', 'label'], values=['mean_rate', 'max_rate', 'p75_rate', 'null_rate'])
    work_stats_pivot.columns = [f"top{int(top_k)}_label_{int(lbl)}_{metric}" for metric, top_k, lbl in work_stats_pivot.columns]
    work_stats_pivot = work_stats_pivot.reset_index()
    _iv = work_stats_pivot['interval'].astype(str).str.extract(r'(-?\\d+)->(-?\\d+)')
    work_stats_pivot['__start__'] = pd.to_numeric(_iv[0], errors='coerce')
    work_stats_pivot['__end__'] = pd.to_numeric(_iv[1], errors='coerce')
    work_stats_pivot = work_stats_pivot.sort_values(by=['__start__', '__end__']).drop(columns=['__start__', '__end__'])
    print('\nTop1~Top5 × label × (mean/max/p75/null) 统计（每行一个 interval）：')
    print(work_stats_pivot)
else:
    print('无法计算作品增长率（缺少连续5年数据或有效数值）')


# 5) 区分得奖/不得奖，在不同类别下的人数（交叉表）
categorical_ct_list = []
for col in categorical_cols:
    print(f"\n[分类人数交叉表] {col} × label (显示前20类):")
    ct = pd.crosstab(df[col], df['label'])
    # 确保 label=0/1 两列都存在
    for lbl in [0, 1]:
        if lbl not in ct.columns:
            ct[lbl] = 0
    # 排列列顺序并计算 total
    ct = ct[[0, 1]]
    ct['total'] = ct.sum(axis=1)
    ct = ct.sort_values('total', ascending=False)
    # 整理为长表并汇总
    ct_out = ct.reset_index().rename(columns={col: 'category', 0: 'label_0', 1: 'label_1'})
    ct_out.insert(0, 'feature', col)
    categorical_ct_list.append(ct_out[['feature', 'category', 'label_0', 'label_1', 'total']])

categorical_ct_all = pd.concat(categorical_ct_list, ignore_index=True) if categorical_ct_list else pd.DataFrame()
print("\n[分类人数交叉表汇总] 所有类别 × label:")
print(categorical_ct_all)
