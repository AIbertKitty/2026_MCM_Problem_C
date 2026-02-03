import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import warnings
import re

# 忽略警告
warnings.filterwarnings('ignore')

# ==========================================
# 1. 配置与数据加载
# ==========================================
INPUT_FILE = ''
OUTPUT_FILE = ''

def load_and_preprocess(file_path):
    print(f">>> [Step 1] 正在读取原始文件: {file_path}")
    df_raw = pd.read_csv(file_path)
    df_original = df_raw.copy()
    
    # 清洗
    df_clean = df_raw.dropna(subset=['celebrity_name', 'season', 'placement']).copy()
    
    # 识别评委分数列
    judge_cols = [c for c in df_clean.columns if 'judge' in c and 'score' in c]
    for col in judge_cols:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        
    return df_original, df_clean, judge_cols

# ==========================================
# 2. 转换为长表 (Wide -> Long)
# ==========================================
def convert_to_long_format(df, judge_cols):
    print(">>> [Step 2] 正在转换数据结构...")
    long_data = []
    
    for idx, row in df.iterrows():
        results_str = str(row['results'])
        elim_week = 99 
        # 解析实际淘汰周
        match = re.search(r'Week\s+(\d+)', results_str, re.IGNORECASE)
        if 'Eliminated' in results_str and match:
            elim_week = int(match.group(1))
            
        for w in range(1, 12):
            cols_this_week = [c for c in judge_cols if f'week{w}_' in c]
            if not cols_this_week: continue
            scores = row[cols_this_week]
            
            if scores.isna().all(): continue
            week_avg_score = np.nanmean(scores)
            if week_avg_score == 0: continue 

            long_data.append({
                'original_index': idx, 
                'season': row['season'],
                'week': w,
                'celebrity_name': row['celebrity_name'],
                'celebrity_industry': row['celebrity_industry'],
                'celebrity_homestate': row['celebrity_homestate'],
                'celebrity_age': row['celebrity_age_during_season'],
                'placement': row['placement'], 
                'elim_week': elim_week,
                'judge_avg_score': week_avg_score
            })
            
    return pd.DataFrame(long_data)

# ==========================================
# 3. 特征工程
# ==========================================
def feature_engineering(df_long):
    print(">>> [Step 3] 正在进行特征工程...")
    # 1. 评委分占比
    df_long['week_total_avg'] = df_long.groupby(['season', 'week'])['judge_avg_score'].transform('sum')
    df_long['judge_share'] = df_long['judge_avg_score'] / df_long['week_total_avg']
    
    # 2. 构造训练目标 (Target Strength)
    max_place = df_long.groupby('season')['placement'].transform('max')
    df_long['target_strength'] = 1 - (df_long['placement'] - 1) / max_place
    
    # 3. 编码
    le = LabelEncoder()
    for col in ['celebrity_industry', 'celebrity_homestate']:
        df_long[col] = df_long[col].fillna('Unknown').astype(str)
        df_long[f'{col}_enc'] = le.fit_transform(df_long[col])
        
    return df_long

# ==========================================
# 4. 置信度计算方法 (Confidence)
# ==========================================
def calculate_confidence(scores_series, is_elimination=True):
    """
    计算模型预测的置信度。
    原理：基于 Margin (差距)。
    - 淘汰周：(倒数第二名分数 - 倒数第一名分数) 的 Sigmoid 映射。
    - 决赛周：(第一名分数 - 第二名分数) 的 Sigmoid 映射。
    """
    sorted_scores = sorted(scores_series)
    if len(sorted_scores) < 2:
        return 0.5 # 无法比较
    
    if is_elimination:
        # 淘汰：关注最低分和次低分的差距
        margin = sorted_scores[1] - sorted_scores[0]
    else:
        # 决赛：关注冠军和亚军的差距
        margin = sorted_scores[-1] - sorted_scores[-2]
    
    # 使用 Sigmoid 函数将差距映射到 0.5 - 1.0 之间
    # 系数 10.0 是根据 target_strength (0-1) 的分布经验设定的
    confidence = 1 / (1 + np.exp(-10.0 * margin))
    return confidence

# ==========================================
# 5. 高级统计量评估 (Consistency & Accuracy)
# ==========================================
def evaluate_advanced_metrics(df_subset, dataset_name="Set"):
    """
    计算：
    1. 基础准确率
    2. 完全正确赛季数
    3. 连续正确数 (Streak) 及其 RMS 评价指标
    4. 平均置信度
    """
    seasons = sorted(df_subset['season'].unique())
    
    total_weeks_checked = 0
    correct_weeks_count = 0
    
    perfect_seasons_count = 0
    streak_squares_sum = 0 # 用于计算 RMS
    
    all_confidences = []
    
    print(f"\n   --- [{dataset_name}] 高级一致性评估 ---")
    
    for s in seasons:
        s_data = df_subset[df_subset['season'] == s]
        weeks = sorted(s_data['week'].unique())
        final_week = max(weeks)
        
        current_streak = 0
        streak_broken = False
        season_is_perfect = True
        
        for w in weeks:
            w_data = s_data[s_data['week'] == w].copy()
            
            # 计算当周模拟总分 (Judge + Vote)
            # 这里简化处理：直接用 pred_strength 近似 (因为 vote 是由 strength 导出的)
            # 为了计算置信度，我们需要具体的分数
            w_data['sim_score'] = w_data['pred_strength_raw']
            
            # --- 分支 A: 决赛周 (Finals) ---
            if w == final_week:
                # 决赛逻辑：比较排名
                # 实际排名
                actual_ranking = w_data.sort_values('placement')['celebrity_name'].tolist()
                # 预测排名 (分数降序)
                pred_ranking = w_data.sort_values('sim_score', ascending=False)['celebrity_name'].tolist()
                
                # 计算置信度
                conf = calculate_confidence(w_data['sim_score'].values, is_elimination=False)
                all_confidences.append(conf)
                
                # 计算决赛中位置正确的个数
                finals_correct_count = 0
                for rank_idx, name in enumerate(actual_ranking):
                    if rank_idx < len(pred_ranking) and name == pred_ranking[rank_idx]:
                        finals_correct_count += 1
                
                # 统计
                if finals_correct_count == len(actual_ranking):
                    # 决赛完全正确
                    pass 
                else:
                    season_is_perfect = False
                
                # 连续性逻辑：如果之前没断，加上决赛正确的个数
                if not streak_broken:
                    current_streak += finals_correct_count
                
            # --- 分支 B: 淘汰周 (Elimination) ---
            else:
                # 找出实际淘汰者
                actual_elim = w_data[w_data['elim_week'] == w]
                
                # 如果这周没人淘汰(比如退赛导致)，跳过不计入 streak 断裂，也不加分
                if actual_elim.empty:
                    continue
                
                total_weeks_checked += 1
                
                # 预测淘汰者 (分数最低)
                pred_loser = w_data.sort_values('sim_score').iloc[0]
                
                # 计算置信度
                conf = calculate_confidence(w_data['sim_score'].values, is_elimination=True)
                all_confidences.append(conf)
                
                is_correct = pred_loser['celebrity_name'] in actual_elim['celebrity_name'].values
                
                if is_correct:
                    correct_weeks_count += 1
                    if not streak_broken:
                        current_streak += 1
                else:
                    season_is_perfect = False
                    streak_broken = True # 连胜中断
        
        # 赛季结束统计
        if season_is_perfect:
            perfect_seasons_count += 1
        
        streak_squares_sum += (current_streak ** 2)
        # print(f"     Season {s}: Streak = {current_streak}") # 调试用
        
    # --- 汇总计算 ---
    avg_acc = correct_weeks_count / total_weeks_checked if total_weeks_checked > 0 else 0
    rms_streak = np.sqrt(streak_squares_sum / len(seasons)) if len(seasons) > 0 else 0
    avg_confidence = np.mean(all_confidences) if all_confidences else 0
    
    print(f"   1. 基础淘汰预测准确率: {avg_acc:.2%}")
    print(f"   2. 完全正确赛季数: {perfect_seasons_count} / {len(seasons)}")
    print(f"   3. 连续正确性奖励指标 (RMS Streak): {rms_streak:.4f}")
    print(f"      (该指标越高，说明模型越能从第1周开始连续准确预测)")
    print(f"   4. 平均预测置信度: {avg_confidence:.4f}")
    
    return avg_acc, rms_streak

# ==========================================
# 6. 训练、评估与全量预测
# ==========================================
def train_evaluate_and_predict(df_long):
    print("\n>>> [Step 4] 模型训练与多维评估")
    
    # --- A. 划分训练/测试集 ---
    seasons = sorted(df_long['season'].unique())
    split_idx = int(len(seasons) * 0.8)
    train_seasons = seasons[:split_idx]
    test_seasons = seasons[split_idx:]
    
    print(f"   训练集: {train_seasons[0]}-{train_seasons[-1]} ({len(train_seasons)}季)")
    print(f"   测试集: {test_seasons[0]}-{test_seasons[-1]} ({len(test_seasons)}季)")
    
    df_train = df_long[df_long['season'].isin(train_seasons)].copy()
    df_test = df_long[df_long['season'].isin(test_seasons)].copy()
    
    features = ['judge_share', 'judge_avg_score', 'celebrity_age', 
                'celebrity_industry_enc', 'celebrity_homestate_enc']
    target = 'target_strength'
    
    # --- B. 训练 ---
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(df_train[features], df_train[target], verbose=False)
    
    # --- C. 预测并保存原始分数 ---
    df_train['pred_strength_raw'] = model.predict(df_train[features])
    df_test['pred_strength_raw'] = model.predict(df_test[features])
    
    # --- D. 执行高级评估 ---
    evaluate_advanced_metrics(df_train, "训练集 (拟合度)")
    evaluate_advanced_metrics(df_test, "测试集 (泛化性)")
    
    # --- E. 全量预测 (Backfill) ---
    print("\n>>> [Step 5] 对全量数据进行得票率模拟与回填...")
    df_long['pred_strength'] = model.predict(df_long[features])
    
    # --- F. 计算观众得票率 (Softmax) ---
    def calculate_vote_share(group):
        # 分离观众因素：总实力 - 0.5 * 评委因素
        proxy_vote = group['pred_strength'] - 0.5 * group['judge_share']
        # 温度系数 8.0
        exp_score = np.exp(proxy_vote * 8.0)
        return exp_score / exp_score.sum()

    df_long['predicted_vote_share'] = df_long.groupby(['season', 'week'], group_keys=False).apply(calculate_vote_share)
    
    return df_long

# ==========================================
# 7. 回填至原始 CSV 结构
# ==========================================
def backfill_to_original(df_original, df_long_predicted):
    print(">>> [Step 6] 执行回填操作...")
    
    new_cols = [f'week{w}_vote_share' for w in range(1, 12)]
    for col in new_cols:
        df_original[col] = np.nan
        
    pivot_votes = df_long_predicted.pivot(index='original_index', columns='week', values='predicted_vote_share')
    
    for w in range(1, 12):
        if w in pivot_votes.columns:
            df_original[f'week{w}_vote_share'] = pivot_votes[w]
            
    return df_original

# ==========================================
# 主程序执行
# ==========================================
try:
    # 1. 加载
    df_orig, df_clean, judge_cols = load_and_preprocess(INPUT_FILE)
    
    # 2. 转换
    df_long = convert_to_long_format(df_clean, judge_cols)
    
    # 3. 特征
    df_features = feature_engineering(df_long)
    
    # 4. 训练、高级评估并全量预测
    df_preds = train_evaluate_and_predict(df_features)
    
    # 5. 回填
    df_final = backfill_to_original(df_orig, df_preds)
    
    # 6. 保存
    df_final.to_csv(OUTPUT_FILE, index=False)
    
    print(f"\n>>> [Success] 任务完成!")
    print(f"    文件已保存至: {OUTPUT_FILE}")

except FileNotFoundError:
    print(f"错误: 找不到文件 {INPUT_FILE}")
except Exception as e:
    import traceback
    traceback.print_exc()