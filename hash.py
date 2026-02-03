import pandas as pd
import numpy as np
import json

def calculate_competition_advantage(file_path, output_file='advantage_dict.json'):
    # 1. 读取数据
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        return f"读取文件失败: {e}"

    # 2. 数据预处理
    # 填充缺失的分类数据
    cat_cols = ['ballroom_partner', 'celebrity_industry', 'celebrity_homestate']
    for col in cat_cols:
        df[col] = df[col].fillna('Unknown')

    # 计算每行的平均分 (排除空值和0.0分)
    def get_row_avg_score(row):
        scores = []
        # 假设分数存储在 weekX_judgeY_score 格式的列中
        score_cols = [c for c in row.index if 'judge' in c and 'score' in c]
        for col in score_cols:
            val = row[col]
            try:
                f_val = float(val)
                if f_val > 0.0:  # 0.0 表示淘汰，不计入平均分计算
                    scores.append(f_val)
            except (ValueError, TypeError):
                continue
        return np.mean(scores) if scores else 0.0

    df['avg_score'] = df.apply(get_row_avg_score, axis=1)

    # 计算赛季统计量：参赛人数 和 选手平均分之和
    season_stats = df.groupby('season').agg(
        n_participants=('celebrity_name', 'count'),
        sum_avg_scores=('avg_score', 'sum')
    ).to_dict('index')

    # 3. 计算单条数据的“比赛优势”
    def calc_advantage_metric(row):
        stats = season_stats.get(row['season'])
        if not stats or stats['n_participants'] == 0 or stats['sum_avg_scores'] == 0:
            return 0.0
        
        # 公式：(-1) * placement / 参赛人数 - avg_score / 参赛选手的avg_score之和
        term1 = (-1) * row['placement'] / stats['n_participants']
        term2 = row['avg_score'] / stats['sum_avg_scores']
        return term1 - term2

    df['raw_advantage'] = df.apply(calc_advantage_metric, axis=1)

    # 构造三元组列
    df['triplet'] = list(zip(df['ballroom_partner'], df['celebrity_industry'], df['celebrity_homestate']))

    # ---------------------------------------------------------
    # 4. 核心逻辑：统计与归一化函数
    # ---------------------------------------------------------
    final_dict = {
        "triplets": {},
        "single_features": {
            "ballroom_partner": {},
            "celebrity_industry": {},
            "celebrity_homestate": {}
        }
    }

    # 设定频次阈值 (思考决定：设为 2，因为三元组完全重复的概率较低，设太高会导致数据稀疏)
    TRIPLET_THRESHOLD = 2
    FEATURE_THRESHOLD = 3

    # --- 处理三元组 ---
    # 统计频次
    triplet_counts = df['triplet'].value_counts()
    
    # 筛选高频三元组并求优势和
    valid_triplets = triplet_counts[triplet_counts >= TRIPLET_THRESHOLD].index
    triplet_sums = df[df['triplet'].isin(valid_triplets)].groupby('triplet')['raw_advantage'].sum()
    
    # 归一化：除以和的绝对值
    total_abs_sum = triplet_sums.abs().sum()
    if total_abs_sum != 0:
        norm_triplets = triplet_sums / total_abs_sum
    else:
        norm_triplets = triplet_sums # 避免除以0
        
    # 存入字典 (Tuple 转 string 以便 JSON 序列化)
    final_dict["triplets"] = {str(k): v for k, v in norm_triplets.items()}

    # --- 处理三大类离散量 (包含 Others 逻辑) ---
    for col in cat_cols:
        # 统计频次
        counts = df[col].value_counts()
        
        # 区分高频和低频 (Others)
        high_freq_keys = counts[counts >= FEATURE_THRESHOLD].index
        
        # 计算高频值的优势和
        high_freq_sum = df[df[col].isin(high_freq_keys)].groupby(col)['raw_advantage'].sum()
        
        # 计算 Others 的优势和
        others_sum = df[~df[col].isin(high_freq_keys)]['raw_advantage'].sum()
        
        # 合并 Series
        combined_sums = high_freq_sum.copy()
        if not df[~df[col].isin(high_freq_keys)].empty:
            combined_sums['others'] = others_sum
            
        # 归一化
        feat_abs_sum = combined_sums.abs().sum()
        if feat_abs_sum != 0:
            norm_feat = combined_sums / feat_abs_sum
        else:
            norm_feat = combined_sums
            
        final_dict["single_features"][col] = norm_feat.to_dict()

    # 保存文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_dict, f, ensure_ascii=False, indent=4)
    
    print(f"字典构建完成，已保存至 {output_file}")
    return final_dict

# ---------------------------------------------------------
# 5. 调用函数 (查询逻辑)
# ---------------------------------------------------------
def get_triplet_advantage(triplet, dict_data):
    """
    输入: triplet (tuple) -> (partner, industry, homestate)
    输入: dict_data (dict) -> 加载的字典数据
    返回: 比赛优势值
    """
    partner, industry, homestate = triplet
    triplet_key = str(triplet)
    
    # 1. 查询三元组是否直接存在
    if triplet_key in dict_data["triplets"]:
        return dict_data["triplets"][triplet_key]
    
    # 2. 若不存在，查询单项强度并求均值
    def get_feature_val(category, key):
        feat_dict = dict_data["single_features"].get(category, {})
        if key in feat_dict:
            return feat_dict[key]
        else:
            return feat_dict.get('others', 0.0)

    val1 = get_feature_val("ballroom_partner", partner)
    val2 = get_feature_val("celebrity_industry", industry)
    val3 = get_feature_val("celebrity_homestate", homestate)
    
    return (val1 + val2 + val3) / 3.0

# ==========================================
# 执行代码示例
# ==========================================
if __name__ == "__main__":
    # 假设文件名为 season_3_27.csv
    file_name = '/Users/xuyunpeng/Desktop/美赛c题2/season_3_27.csv'
    
    # 1. 构建字典
    advantage_data = calculate_competition_advantage(file_name)
    
    # 2. 测试调用
    if isinstance(advantage_data, dict):
        # 测试用例 1: 数据集中可能存在的组合
        test_triplet_1 = ('Cheryl Burke', 'Athlete', 'Florida') 
        score_1 = get_triplet_advantage(test_triplet_1, advantage_data)
        print(f"三元组 {test_triplet_1} 的比赛优势: {score_1}")

        # 测试用例 2: 一个完全陌生的组合 (测试均值逻辑)
        test_triplet_2 = ('Unknown Dancer', 'Scientist', 'Mars')
        score_2 = get_triplet_advantage(test_triplet_2, advantage_data)
        print(f"三元组 {test_triplet_2} 的比赛优势: {score_2}")