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
INPUT_FILE = '/Users/xuyunpeng/Desktop/美赛c题2/season_28_34.csv'
OUTPUT_FILE = 'season_28_30_with_votes.csv'

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
    
    # 2. 构造训练目标
    max_place = df_long.groupby('season')['placement'].transform('max')
    df_long['target_strength'] = 1 - (df_long['placement'] - 1) / max_place
    
    # 3. 编码
    le = LabelEncoder()
    for col in ['celebrity_industry', 'celebrity_homestate']:
        df_long[col] = df_long[col].fillna('Unknown').astype(str)
        df_long[f'{col}_enc'] = le.fit_transform(df_long[col])
        
    return df_long

# ==========================================
# 4. 动态置信度计算 (适配两种规则)
# ==========================================
def calculate_dynamic_confidence(candidates, rule_type='rank_sum'):
    """
    根据不同的淘汰规则计算置信度。
    """
    if rule_type == 'rank_sum':
        # 旧规则：直接看 Rank Sum 的差距
        # 候选人已按 Rank Sum 降序排列 (最差的在前)
        # Gap = Worst - 2nd_Worst
        scores = candidates['rank_sum'].values
        if len(scores) < 2: return 0.5
        gap = scores[0] - scores[1] # 越大约好
        # Rank Sum 是整数，差距通常在 1-5 之间
        return 1 / (1 + np.exp(-0.5 * gap))
        
    elif rule_type == 'judges_save':
        # 新规则：看 Bottom 2 之间评委分的差距
        # 假设 candidates 是 Bottom 2，且已经按 judge_score 排序
        # Gap = Saved_Score - Eliminated_Score
        scores = candidates['judge_avg_score'].values
        if len(scores) < 2: return 0.5
        gap = abs(scores[0] - scores[1])
        # 评委分差距通常在 0-5 分之间
        return 1 / (1 + np.exp(-2.0 * gap))
    
    elif rule_type == 'finals':
        # 决赛：看 Rank Sum 差距 (越小越好)
        scores = sorted(candidates['rank_sum'].values)
        if len(scores) < 2: return 0.5
        gap = scores[1] - scores[0] # 2nd Best - Best
        return 1 / (1 + np.exp(-0.5 * gap))
        
    return 0.5

# ==========================================
# 5. 高级统计量评估 (引入第28季规则变更)
# ==========================================
def evaluate_advanced_metrics(df_subset, dataset_name="Set"):
    seasons = sorted(df_subset['season'].unique())
    
    total_weeks_checked = 0
    correct_weeks_count = 0
    perfect_seasons_count = 0
    streak_squares_sum = 0 
    all_confidences = []
    
    print(f"\n   --- [{dataset_name}] 高级一致性评估 (含 Season 28+ 评委拯救机制) ---")
    
    for s in seasons:
        s_data = df_subset[df_subset['season'] == s]
        weeks = sorted(s_data['week'].unique())
        final_week = max(weeks)
        
        current_streak = 0
        streak_broken = False
        season_is_perfect = True
        
        for w in weeks:
            w_data = s_data[s_data['week'] == w].copy()
            
            # 1. 模拟观众得票率
            raw_preds = w_data['pred_strength_raw'] - 0.5 * w_data['judge_share']
            exp_preds = np.exp(raw_preds * 8.0)
            w_data['sim_vote_share'] = exp_preds / exp_preds.sum()
            
            # 2. 计算排名 (Rank) - 降序 (分数高=Rank 1)
            w_data['judge_rank'] = w_data['judge_avg_score'].rank(ascending=False, method='min')
            w_data['vote_rank'] = w_data['sim_vote_share'].rank(ascending=False, method='min')
            
            # 3. 计算排名和 (Rank Sum) - 值越小越好
            w_data['rank_sum'] = w_data['judge_rank'] + w_data['vote_rank']
            
            # --- 分支 A: 决赛周 (Finals) ---
            # 决赛规则不变：排名和决定最终名次
            if w == final_week:
                actual_ranking = w_data.sort_values('placement')['celebrity_name'].tolist()
                
                # 预测排名 (Rank Sum 升序)
                pred_ranking = w_data.sort_values(['rank_sum', 'sim_vote_share'], ascending=[True, False])['celebrity_name'].tolist()
                
                # 决赛置信度
                conf = calculate_dynamic_confidence(w_data, 'finals')
                all_confidences.append(conf)
                
                finals_correct = 0
                for rank_idx, name in enumerate(actual_ranking):
                    if rank_idx < len(pred_ranking) and name == pred_ranking[rank_idx]:
                        finals_correct += 1
                
                if finals_correct != len(actual_ranking):
                    season_is_perfect = False
                
                if not streak_broken:
                    current_streak += finals_correct
                
            # --- 分支 B: 淘汰周 (Elimination) ---
            else:
                actual_elim = w_data[w_data['elim_week'] == w]
                if actual_elim.empty: continue # 无人淘汰
                
                total_weeks_checked += 1
                
                # === 核心逻辑变更点 ===
                pred_loser_name = None
                conf = 0.5
                
                # 规则判定：Season 28 之前 vs 之后
                if s < 28:
                    # [旧规则] 直接淘汰 Rank Sum 最大者
                    # 排序：Rank Sum 降序 (最大的在最前)
                    candidates = w_data.sort_values(['rank_sum', 'sim_vote_share'], ascending=[False, True])
                    pred_loser = candidates.iloc[0]
                    pred_loser_name = pred_loser['celebrity_name']
                    
                    conf = calculate_dynamic_confidence(candidates, 'rank_sum')
                    
                else:
                    # [新规则] Season 28+: 评委拯救机制
                    # 1. 找出 Bottom 2 (Rank Sum 最大的两人)
                    bottom_two = w_data.sort_values(['rank_sum', 'sim_vote_share'], ascending=[False, True]).head(2)
                    
                    if len(bottom_two) < 2:
                        # 异常处理：如果只剩1人(不可能，但防守性编程)
                        pred_loser_name = bottom_two.iloc[0]['celebrity_name']
                    else:
                        # 2. 评委投票：比较两人的 judge_avg_score
                        # 评委通常会救分数高的，淘汰分数低的
                        # 按 judge_score 升序排列，第一个就是分数最低的(被淘汰)
                        decision_order = bottom_two.sort_values('judge_avg_score', ascending=True)
                        pred_loser = decision_order.iloc[0]
                        pred_loser_name = pred_loser['celebrity_name']
                        
                        conf = calculate_dynamic_confidence(bottom_two, 'judges_save')

                all_confidences.append(conf)
                
                # 判定是否正确
                is_correct = pred_loser_name in actual_elim['celebrity_name'].values
                
                if is_correct:
                    correct_weeks_count += 1
                    if not streak_broken:
                        current_streak += 1
                else:
                    season_is_perfect = False
                    streak_broken = True
        
        if season_is_perfect:
            perfect_seasons_count += 1
        streak_squares_sum += (current_streak ** 2)
        
    avg_acc = correct_weeks_count / total_weeks_checked if total_weeks_checked > 0 else 0
    rms_streak = np.sqrt(streak_squares_sum / len(seasons)) if len(seasons) > 0 else 0
    avg_confidence = np.mean(all_confidences) if all_confidences else 0
    
    print(f"   1. 淘汰预测准确率: {avg_acc:.2%}")
    print(f"   2. 完全正确赛季数: {perfect_seasons_count} / {len(seasons)}")
    print(f"   3. 连续正确性 (RMS Streak): {rms_streak:.4f}")
    print(f"   4. 平均置信度: {avg_confidence:.4f}")
    
    return avg_acc

# ==========================================
# 6. 训练、评估与全量预测
# ==========================================
def train_evaluate_and_predict(df_long):
    print("\n>>> [Step 4] 模型训练与多维评估")
    
    seasons = sorted(df_long['season'].unique())
    split_idx = int(len(seasons) * 0.8)
    train_seasons = seasons[:split_idx]
    test_seasons = seasons[split_idx:]
    
    print(f"   训练集: {train_seasons[0]}-{train_seasons[-1]}")
    print(f"   测试集: {test_seasons[0]}-{test_seasons[-1]}")
    
    df_train = df_long[df_long['season'].isin(train_seasons)].copy()
    df_test = df_long[df_long['season'].isin(test_seasons)].copy()
    
    features = ['judge_share', 'judge_avg_score', 'celebrity_age', 
                'celebrity_industry_enc', 'celebrity_homestate_enc']
    target = 'target_strength'
    
    # 训练
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
    
    # 预测原始分数
    df_train['pred_strength_raw'] = model.predict(df_train[features])
    df_test['pred_strength_raw'] = model.predict(df_test[features])
    
    # 评估 (应用 Season 28 规则变更)
    evaluate_advanced_metrics(df_train, "训练集")
    evaluate_advanced_metrics(df_test, "测试集")
    
    # 全量预测
    print("\n>>> [Step 5] 对全量数据进行回填...")
    df_long['pred_strength'] = model.predict(df_long[features])
    
    # 生成最终 Vote Share
    def calculate_vote_share(group):
        proxy_vote = group['pred_strength'] - 0.5 * group['judge_share']
        exp_score = np.exp(proxy_vote * 8.0)
        return exp_score / exp_score.sum()

    df_long['predicted_vote_share'] = df_long.groupby(['season', 'week'], group_keys=False).apply(calculate_vote_share)
    
    return df_long

# ==========================================
# 7. 回填至原始 CSV
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
# 主程序
# ==========================================
try:
    df_orig, df_clean, judge_cols = load_and_preprocess(INPUT_FILE)
    df_long = convert_to_long_format(df_clean, judge_cols)
    df_features = feature_engineering(df_long)
    df_preds = train_evaluate_and_predict(df_features)
    df_final = backfill_to_original(df_orig, df_preds)
    df_final.to_csv(OUTPUT_FILE, index=False)
    
    print(f"\n>>> [Success] 任务完成! 文件已保存至: {OUTPUT_FILE}")

except FileNotFoundError:
    print(f"错误: 找不到文件 {INPUT_FILE}")
except Exception as e:
    import traceback
    traceback.print_exc()