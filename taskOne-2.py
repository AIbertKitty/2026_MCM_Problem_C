import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import os
from collections import deque
from sklearn.preprocessing import StandardScaler
import json 

# ==========================================
# 0.预定义离散量
# ==========================================

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
# 1. 外部接口函数与工具
# ==========================================

def calculate_js_features(scores):
    """
    将变长的历史得分向量转化为定长的统计特征 (JS)。
    """
    if len(scores) == 0:
        return [0.0, 0.0, 0.0, 0.0]
    
    scores = np.array(scores)
    mean_score = np.mean(scores)
    last_score = scores[-1]
    
    if len(scores) > 1:
        x = np.arange(len(scores))
        slope, _ = np.polyfit(x, scores, 1)
        trend = slope
    else:
        trend = 0.0
        
    consistency = np.std(scores)
    
    return [mean_score, last_score, trend, consistency]

# ==========================================
# 2. 神经网络模型 (单隐层 + BatchNorm)
# ==========================================

class XSimpleDancingModel(nn.Module):
    def __init__(self, input_dim):
        super(XSimpleDancingModel, self).__init__()
        # 简化结构：Input(7) -> Linear(16) -> BN -> ReLU -> Output(1)
        self.hidden = nn.Linear(input_dim, 16)
        self.bn = nn.BatchNorm1d(16) # 批归一化
        self.relu = nn.ReLU()
        self.output = nn.Linear(16, 1) 

    def forward(self, x):
        x = self.hidden(x)
        x = self.bn(x)
        x = self.relu(x)
        r = self.output(x)
        return r

# ==========================================
# 3. 数据预处理
# ==========================================

def load_and_process_data(file_path):
    df = pd.read_csv(file_path)
    
    # 基础清洗
    df['celebrity_age_during_season'] = pd.to_numeric(df['celebrity_age_during_season'], errors='coerce').fillna(0)
    
    # 解析 Result 得到淘汰周
    def get_elimination_week(res_str):
        if not isinstance(res_str, str):
            return 99 
        res_str = res_str.lower()
        if "eliminated week" in res_str:
            try:
                return int(res_str.split("week")[-1].strip())
            except:
                return 99
        elif "place" in res_str or "winner" in res_str:
            return 99 
        elif "withdrew" in res_str:
            return 99 
        return 99

    df['eliminated_week'] = df['results'].apply(get_elimination_week)
    
    # 提取每周分数并处理空值
    score_cols = [f'week{w}_judge{j}_score' for w in range(1, 12) for j in range(1, 5)]
    for col in score_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        else:
            df[col] = 0.0
            
    # 初始化预测结果列
    for w in range(1, 12):
        df[f'week{w}_predicted_rate'] = np.nan
            
    return df

# ==========================================
# 4. 核心逻辑类
# ==========================================

class CompetitionSimulator:
    def __init__(self, df, dict_data=None):
        self.df = df
        self.dict_data = dict_data if dict_data is not None else {} 
        self.seasons = sorted(df['season'].unique())
        
        # 划分训练集和测试集
        split_idx = int(len(self.seasons) * 0.8)
        self.train_seasons = self.seasons[:split_idx]
        self.test_seasons = self.seasons[split_idx:]
        
        # 特征标准化
        self.scaler = StandardScaler()
        self._fit_scaler()
        
        # 模型初始化
        self.input_dim = 7
        self.model = XSimpleDancingModel(input_dim=self.input_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.002)
        self.criterion = nn.MarginRankingLoss(margin=0.2)
        
        # 经验回放 Buffer
        self.replay_buffer = deque(maxlen=2000)
        self.batch_size = 16

    def _fit_scaler(self):
        """预先遍历数据以拟合 StandardScaler"""
        print("Fitting scaler on training data...")
        all_feats = []
        train_df = self.df[self.df['season'].isin(self.train_seasons)]
        
        # 抽样部分数据进行拟合
        for _, row in train_df.head(500).iterrows():
            triplet = (row['ballroom_partner'], row['celebrity_industry'], row['celebrity_homestate'])
            f_pih = get_triplet_advantage(triplet, self.dict_data)
            a = row['celebrity_age_during_season']
            s = row['season']
            js = [8.0, 8.0, 0.0, 0.5] # 模拟值
            all_feats.append([f_pih, a, s] + js)
            
        if all_feats:
            self.scaler.fit(all_feats)

    def extract_features(self, dancer_row, season, week):
        triplet = (dancer_row['ballroom_partner'], dancer_row['celebrity_industry'], dancer_row['celebrity_homestate'])
        f_pih = get_triplet_advantage(triplet, self.dict_data)
        
        a = dancer_row['celebrity_age_during_season']
        s = season
        
        history_scores = []
        for w in range(1, week):
            cols = [f'week{w}_judge{j}_score' for j in range(1, 5)]
            w_score = dancer_row[cols].sum()
            if w_score > 0:
                history_scores.append(w_score)
        
        js_features = calculate_js_features(history_scores)
        
        raw_features = np.array([[f_pih, a, s] + js_features], dtype=np.float32)
        scaled_features = self.scaler.transform(raw_features)
        
        return torch.tensor(scaled_features)

    def train_step(self, survivor_feats, eliminated_feats):
        self.replay_buffer.append((survivor_feats.detach(), eliminated_feats.detach()))
        
        updates = 3 if len(self.replay_buffer) > self.batch_size else 1
        losses = []
        
        self.model.train() # 开启训练模式
        
        for _ in range(updates):
            if len(self.replay_buffer) < self.batch_size:
                batch = list(self.replay_buffer)
            else:
                batch = random.sample(self.replay_buffer, self.batch_size)
            
            # 【修复关键点】：如果 batch 只有 1 个样本，复制一份以满足 BatchNorm 的要求
            if len(batch) == 1:
                batch.append(batch[0])
                
            batch_survivor = torch.cat([x[0] for x in batch])
            batch_eliminated = torch.cat([x[1] for x in batch])
            
            self.optimizer.zero_grad()
            score_survivor = self.model(batch_survivor)
            score_eliminated = self.model(batch_eliminated)
            
            target = torch.ones(score_survivor.size(0), 1)
            loss = self.criterion(score_survivor, score_eliminated, target)
            
            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())
            
        return np.mean(losses)

    def run_season(self, season, is_training=True, force_continue=False):
        correct_predictions = 0
        total_predictions = 0
        season_active = True
        current_week = 1
        
        while season_active and current_week <= 11:
            season_df = self.df[self.df['season'] == season]
            active_dancers = season_df[season_df['eliminated_week'] >= current_week].copy()
            
            score_cols = [f'week{current_week}_judge{j}_score' for j in range(1, 5)]
            has_scores = active_dancers[score_cols].sum(axis=1) > 0
            active_dancers = active_dancers[has_scores]

            if active_dancers.empty or len(active_dancers) <= 1:
                break 
            
            # 1. 评委分占比
            raw_scores = active_dancers[score_cols].sum(axis=1)
            total_judge_points = raw_scores.sum()
            if total_judge_points == 0:
                current_week += 1
                continue
            judge_shares = torch.tensor((raw_scores / total_judge_points).values, dtype=torch.float32)
            
            # 2. 模型预测观众占比
            feature_list = []
            indices = active_dancers.index
            
            self.model.eval() # 推理模式 (BatchNorm 使用统计均值，不依赖 BatchSize)
            with torch.no_grad():
                for _, row in active_dancers.iterrows():
                    feat = self.extract_features(row, season, current_week)
                    feature_list.append(feat)
                all_feats = torch.cat(feature_list)
                raw_r = self.model(all_feats).squeeze()
            
            audience_shares = torch.softmax(raw_r, dim=0)
            
            # 回填数据
            col_name = f'week{current_week}_predicted_rate'
            self.df.loc[indices, col_name] = audience_shares.numpy()
            
            # 3. 综合得分与淘汰
            total_scores = audience_shares + judge_shares
            min_score_idx = torch.argmin(total_scores).item()
            predicted_eliminated_idx = active_dancers.index[min_score_idx]
            
            # 4. 验证
            actual_eliminated = active_dancers[active_dancers['eliminated_week'] == current_week]
            num_eliminated = len(actual_eliminated)
            
            prediction_correct = False
            if num_eliminated == 0:
                prediction_correct = True
            elif num_eliminated >= 1:
                if predicted_eliminated_idx in actual_eliminated.index:
                    prediction_correct = True
                    correct_predictions += 1
                else:
                    prediction_correct = False
                    if is_training:
                        real_eliminated_row = actual_eliminated.iloc[0]
                        wrongly_accused_row = active_dancers.loc[predicted_eliminated_idx]
                        
                        feat_survivor = self.extract_features(wrongly_accused_row, season, current_week)
                        feat_eliminated = self.extract_features(real_eliminated_row, season, current_week)
                        self.train_step(feat_survivor, feat_eliminated)

            if num_eliminated > 0:
                total_predictions += 1

            # 5. 流程控制
            if prediction_correct:
                current_week += 1
            else:
                if force_continue:
                    current_week += 1
                else:
                    break 
        
        return correct_predictions, total_predictions

    def optimize(self, threshold=0.7):
        print(f"Starting Optimization on {len(self.train_seasons)} seasons...")
        for epoch in range(1, 101):
            total_correct = 0
            total_count = 0
            random.shuffle(self.train_seasons)
            
            for s in self.train_seasons:
                c, t = self.run_season(s, is_training=True, force_continue=False)
                total_correct += c
                total_count += t
            
            acc = total_correct / total_count if total_count > 0 else 0
            print(f"Epoch {epoch}: Train Accuracy = {acc:.4f} ({total_correct}/{total_count})")
            
            if acc >= threshold:
                print("Optimization finished.")
                break

    def test_and_fill(self):
        print(f"\nStarting Testing (Forced Continuation) on {len(self.test_seasons)} seasons...")
        total_correct = 0
        total_count = 0
        
        for s in self.test_seasons:
            c, t = self.run_season(s, is_training=False, force_continue=True)
            total_correct += c
            total_count += t
            
        acc = total_correct / total_count if total_count > 0 else 0
        print(f"Test Accuracy = {acc:.4f} ({total_correct}/{total_count})")

    def save_model_weights(self, path='xsimple_model.pth'):
        torch.save(self.model.state_dict(), path)
        print(f"Model weights saved to {path}")

    def load_model_weights(self, path='xsimple_model.pth', print_matrix=False):
        if os.path.exists(path):
            self.model.load_state_dict(torch.load(path))
            print(f"Model weights loaded from {path}")
            if print_matrix:
                print("\n--- Model Weights ---")
                for name, param in self.model.named_parameters():
                    print(f"{name}: {param.data}")
                    print('sum')
                    print(f"{name}: {torch.sum(param.data)}")
                print("---------------------\n")
        else:
            print(f"Warning: {path} not found.")

# ==========================================
# 5. 主程序
# ==========================================

if __name__ == "__main__":
    file_path = '/Users/xuyunpeng/Desktop/美赛c题2/season_3_27.csv' 
    output_path = 'season_3_27_with_predictions.csv'
    
    # 模拟外部数据字典
    # my_dict_data = {"some_key": "some_value"} 
    my_dict_data = calculate_competition_advantage(file_path)
    
    try:
        # 1. 加载数据
        df = load_and_process_data(file_path)
        
        # 2. 初始化模拟器
        sim = CompetitionSimulator(df, dict_data=my_dict_data)
        
        # 3. 优化
        sim.optimize(threshold=0.6)
        
        # 4. 保存并打印权重
        sim.save_model_weights()
        sim.load_model_weights(print_matrix=True)
        
        # 5. 测试并回填数据
        sim.test_and_fill()
        
        # 6. 保存结果
        df.to_csv(output_path, index=False)
        print(f"Results saved to {output_path}")
        
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()