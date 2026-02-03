import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import lingam
from sklearn.preprocessing import StandardScaler
import warnings

# 忽略警告
warnings.filterwarnings('ignore')

# ==========================================
# 1. 数据加载与特征工程 (因果分析用)
# ==========================================

def load_and_preprocess(file_path):
    print(f"正在加载数据: {file_path} ...")
    df = pd.read_csv(file_path)
    
    # --- 1. 计算 avg_score ---
    score_cols = [col for col in df.columns if 'score' in col.lower() and 'judge' in col.lower()]
    for col in score_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['avg_score'] = df[score_cols].mean(axis=1)
    
    # --- 2. 计算 avg_score_ratio ---
    season_avg_score_sum = df.groupby('season')['avg_score'].transform('sum')
    df['avg_score_ratio'] = df['avg_score'] / season_avg_score_sum.replace(0, np.nan)
    
    # --- 3. 计算 avg_vote ---
    vote_cols = [col for col in df.columns if 'vote_share' in col]
    for col in vote_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['avg_vote'] = df[vote_cols].mean(axis=1)
    
    # --- 4. 选定列 ---
    columns = [
        'avg_score_ratio', 'avg_vote', 'placement', 'season', 
        'ballroom_partner', 'celebrity_industry', 'celebrity_age_during_season'
    ]
    
    existing_cols = [c for c in columns if c in df.columns]
    data = df[existing_cols].copy()
    
    cols_to_numeric = ['placement', 'season', 'celebrity_age_during_season', 'avg_score_ratio', 'avg_vote']
    for col in cols_to_numeric:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')
            
    data = data.dropna()
    return data

# ==========================================
# 2. 预处理 (目标编码与标准化)
# ==========================================

def target_encode_variable(df, col, target_col='placement'):
    global_mean = df[target_col].mean()
    agg = df.groupby(col)[target_col].agg(['count', 'mean'])
    counts = agg['count']
    means = agg['mean']
    m = 5 
    smooth_means = (counts * means + m * global_mean) / (counts + m)
    return df[col].map(smooth_means)

def preprocess_features(data):
    df_encoded = data.copy()
    categorical_cols = ['ballroom_partner', 'celebrity_industry']
    
    for col in categorical_cols:
        if col in df_encoded.columns:
            counts = df_encoded[col].value_counts()
            threshold = 3
            rare = counts[counts < threshold].index
            df_encoded[col] = df_encoded[col].apply(lambda x: 'Others' if x in rare else x)
            df_encoded[col] = target_encode_variable(df_encoded, col, 'placement')
            
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df_encoded), columns=df_encoded.columns)
    return df_scaled

# ==========================================
# 3. Bootstrap LiNGAM (因果分析)
# ==========================================

def run_bootstrap_lingam(df, n_sampling=50):
    print(f"\n启动 Bootstrap DirectLiNGAM (采样次数: {n_sampling})...")
    model = lingam.DirectLiNGAM(random_state=42)
    result = model.bootstrap(df, n_sampling=n_sampling)
    return result.adjacency_matrices_

def calculate_total_causal_effects(matrices, labels, prob_threshold=0.2):
    print("\n" + "="*60)
    print("正在计算 avg_score_ratio 和 avg_vote 对 placement 的总因果效应")
    print("="*60)

    n_features = len(labels)
    probability_matrix = np.zeros((n_features, n_features))
    for matrix in matrices:
        adj = np.abs(matrix) > 0.0001 
        probability_matrix += adj.astype(float)
    probability_matrix /= len(matrices)
    
    mean_matrix = np.mean(matrices, axis=0)
    G = nx.DiGraph()
    for label in labels: G.add_node(label)
        
    for r in range(n_features):
        for c in range(n_features):
            if r == c: continue
            if probability_matrix[r, c] > prob_threshold:
                weight = mean_matrix[r, c]
                G.add_edge(labels[c], labels[r], weight=weight)

    def get_total_effect(source, target):
        if source not in G or target not in G: return 0.0
        total_effect = 0.0
        try:
            paths = list(nx.all_simple_paths(G, source, target, cutoff=5))
            for path in paths:
                path_effect = 1.0
                for i in range(len(path) - 1):
                    path_effect *= G[path[i]][path[i+1]]['weight']
                total_effect += path_effect
        except: pass
        return total_effect

    target = 'placement'
    sources = ['avg_score_ratio', 'avg_vote']
    results = {}
    for source in sources:
        effect = get_total_effect(source, target)
        results[source] = effect
        print(f"{source} 对 {target} 的总因果效应: {effect:.4f}")
    
    return G, results

def visualize_causal_graph(G, results):
    plt.figure(figsize=(12, 10))
    pos = nx.spring_layout(G, k=2.0, iterations=200, seed=42)
    node_size = 3000
    nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color='#F0F8FF', edgecolors='#4682B4')
    formatted_labels = {n: n.replace('_', '\n') if len(n)>10 else n for n in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=formatted_labels, font_size=9, font_weight='bold')
    
    edges = G.edges(data=True)
    if edges:
        weights = [abs(d['weight']) * 3 + 0.5 for u, v, d in edges]
        edge_colors = [d['weight'] for u, v, d in edges]
        nx.draw_networkx_edges(G, pos, width=weights, arrowsize=20, arrowstyle='-|>',
                               edge_color=edge_colors, edge_cmap=plt.cm.RdBu, edge_vmin=-1, edge_vmax=1,
                               connectionstyle="arc3,rad=0.1", node_size=node_size)
        edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in edges}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
    
    plt.title(f"Causal Effects on Placement\navg_score_ratio: {results.get('avg_score_ratio',0):.2f}, avg_vote: {results.get('avg_vote',0):.2f}", fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# ==========================================
# 5. 反事实模拟 (新增功能)
# ==========================================

def simulate_counterfactual(file_path):
    print("\n" + "="*60)
    print("启动反事实模拟 (Counterfactual Simulation)")
    print("规则变更: 排名相加 -> 倒数两名 -> 评委投票淘汰")
    print("="*60)
    
    # 读取原始数据以获取每周详情
    df_raw = pd.read_csv(file_path)
    
    # 目标模拟列表
    targets = [
        {'name': 'Billy Ray Cyrus', 'season': 4},
        {'name': 'Bristol Palin', 'season': 11},
        {'name': 'Bobby Bones', 'season': 27}
    ]
    
    # 辅助函数：获取某周的总分
    def get_week_score(row, week_num):
        cols = [c for c in df_raw.columns if f'week{week_num}_judge' in c and 'score' in c]
        scores = []
        for c in cols:
            val = pd.to_numeric(row[c], errors='coerce')
            if not pd.isna(val):
                scores.append(val)
        return sum(scores) if scores else np.nan

    # 辅助函数：获取某周的得票率
    def get_week_vote(row, week_num):
        col = f'week{week_num}_vote_share'
        if col in df_raw.columns:
            return pd.to_numeric(row[col], errors='coerce')
        return np.nan

    for target in targets:
        t_name = target['name']
        t_season = target['season']
        
        # 1. 筛选该赛季数据
        season_df = df_raw[df_raw['season'] == t_season].copy()
        if season_df.empty:
            print(f"未找到 Season {t_season} 的数据，跳过。")
            continue
            
        # 初始化选手状态
        # 结构: {name: {'active': True, 'scores': [], 'votes': [], 'cumulative_score': 0}}
        contestants = {}
        for _, row in season_df.iterrows():
            name = row['celebrity_name']
            contestants[name] = {
                'active': True,
                'history_scores': [], # 用于填补缺失值
                'last_vote': 0.0,      # 用于填补缺失值
                'cumulative_score': 0.0,
                'row_data': row # 保留原始行数据方便读取
            }
            
        print(f"\n>>> 模拟 Season {t_season} (关注对象: {t_name})")
        
        # 模拟周次 (假设最多11周)
        final_placement = -1
        
        for week in range(1, 12):
            active_names = [n for n, d in contestants.items() if d['active']]
            
            # 如果只剩3人或更少，进入决赛逻辑，模拟结束
            if len(active_names) <= 3:
                # 简单判定：决赛中按当前累计分排名
                sorted_finalists = sorted(active_names, key=lambda x: contestants[x]['cumulative_score'], reverse=True)
                for rank, name in enumerate(sorted_finalists):
                    if name == t_name:
                        final_placement = rank + 1
                break
            
            week_data = []
            
            # 收集本周数据
            for name in active_names:
                c_data = contestants[name]
                
                # 获取分数
                raw_score = get_week_score(c_data['row_data'], week)
                if pd.isna(raw_score) or raw_score == 0:
                    # 填补：使用历史平均分
                    if c_data['history_scores']:
                        score = np.mean(c_data['history_scores'])
                    else:
                        score = 20.0 # 默认基准分
                else:
                    score = raw_score
                    c_data['history_scores'].append(score)
                
                # 更新累计分
                c_data['cumulative_score'] += score
                
                # 获取得票率
                raw_vote = get_week_vote(c_data['row_data'], week)
                if pd.isna(raw_vote):
                    # 填补：使用上周得票率
                    vote = c_data['last_vote'] if c_data['last_vote'] > 0 else 0.05
                else:
                    vote = raw_vote
                    c_data['last_vote'] = vote
                    
                week_data.append({
                    'name': name,
                    'score': score,
                    'vote': vote,
                    'cumulative': c_data['cumulative_score']
                })
            
            # --- 排名计算 ---
            # 分数排名 (高分排名靠前，即 rank 小)
            week_data.sort(key=lambda x: x['score'], reverse=True)
            for i, d in enumerate(week_data):
                d['score_rank'] = i + 1
                
            # 投票排名 (高票排名靠前)
            week_data.sort(key=lambda x: x['vote'], reverse=True)
            for i, d in enumerate(week_data):
                d['vote_rank'] = i + 1
                
            # 总排名 (Rank Sum, 小的好)
            for d in week_data:
                d['rank_sum'] = d['score_rank'] + d['vote_rank']
                
            # 排序找出 Bottom 2 (Rank Sum 最大的两个)
            # 如果 Rank Sum 相同，分数低的在后面 (更危险)
            week_data.sort(key=lambda x: (x['rank_sum'], -x['score']), reverse=False)
            
            bottom_2 = week_data[-2:] # 最后两名
            
            # --- 淘汰逻辑 (评委二选一) ---
            # 规则：评委保留当周分数高的人。如果当周分数平，保留累计分高的人。
            p1 = bottom_2[0] # 倒数第二
            p2 = bottom_2[1] # 倒数第一
            
            eliminated_name = None
            
            # 比较当周分数
            if p1['score'] > p2['score']:
                eliminated_name = p2['name']
            elif p2['score'] > p1['score']:
                eliminated_name = p1['name']
            else:
                # 当周分数平，比较累计
                if p1['cumulative'] > p2['cumulative']:
                    eliminated_name = p2['name']
                else:
                    eliminated_name = p1['name']
            
            # 执行淘汰
            contestants[eliminated_name]['active'] = False
            current_placement = len(active_names)
            
            # 打印关键过程
            is_target_involved = (t_name in [p1['name'], p2['name']])
            if is_target_involved:
                print(f"  Week {week}: {t_name} 进入 Bottom 2! 对手: {p1['name'] if p2['name']==t_name else p2['name']}")
                if eliminated_name == t_name:
                    print(f"    -> 评委投票淘汰了 {t_name}。")
                    final_placement = current_placement
                else:
                    print(f"    -> 评委拯救了 {t_name}！")

            if eliminated_name == t_name:
                final_placement = current_placement
                break
        
        # 如果循环结束还在 active，说明进了决赛
        if final_placement == -1:
            # 简单假设决赛排名
            active_final = [n for n, d in contestants.items() if d['active']]
            if t_name in active_final:
                # 模拟决赛排名（按累计分）
                sorted_final = sorted(active_final, key=lambda x: contestants[x]['cumulative_score'], reverse=True)
                final_placement = sorted_final.index(t_name) + 1

        print(f"  >>> {t_name} (Season {t_season}) 反事实模拟最终排名: {final_placement}")
        
        # 获取真实排名对比
        real_placement = season_df[season_df['celebrity_name'] == t_name]['placement'].values[0]
        print(f"      (真实排名: {real_placement})")

# ==========================================
# 主程序
# ==========================================

if __name__ == "__main__":
    file_path = '/Users/xuyunpeng/Desktop/美赛c题2/season_3_34vote.csv'

    try:
        # 1-4. 原有的因果分析流程
        data = load_and_preprocess(file_path)
        df_encoded = preprocess_features(data)
        matrices = run_bootstrap_lingam(df_encoded, n_sampling=50)
        G, results = calculate_total_causal_effects(matrices, df_encoded.columns.tolist(), prob_threshold=0.2)
        visualize_causal_graph(G, results)
        
        # 5. 新增：反事实模拟
        simulate_counterfactual(file_path)
        
    except FileNotFoundError:
        print(f"错误: 找不到文件 '{file_path}'。")
    except Exception as e:
        import traceback
        traceback.print_exc()