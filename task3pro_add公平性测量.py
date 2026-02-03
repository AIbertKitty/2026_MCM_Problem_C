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
# 1. 数据加载与特征工程
# ==========================================

def load_and_preprocess(file_path):
    print(f"正在加载数据: {file_path} ...")
    df = pd.read_csv(file_path)
    
    # --- 1. 计算 avg_score (裁判分均值) ---
    score_cols = [col for col in df.columns if 'score' in col.lower() and 'judge' in col.lower()]
    for col in score_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['avg_score'] = df[score_cols].mean(axis=1)
    
    # --- 2. 计算 avg_vote (观众得票率均值) ---
    vote_cols = [col for col in df.columns if 'vote_share' in col]
    print(f"检测到 {len(vote_cols)} 个投票率列用于计算 avg_vote")
    
    for col in vote_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df['avg_vote'] = df[vote_cols].mean(axis=1)
    
    # --- 3. 选定列 ---
    columns = [
        'ballroom_partner',             # Unfair
        'celebrity_homestate',          # Unfair
        'celebrity_industry',           # Fair
        'celebrity_age_during_season',  # Fair
        'season',                       # Ignored
        'avg_score',                    # Target 1
        'avg_vote',                     # Target 2
        'placement'                     # Ignored/Result
    ]
    
    existing_cols = [c for c in columns if c in df.columns]
    data = df[existing_cols].copy()
    
    # 转换数值
    cols_to_numeric = ['placement', 'season', 'celebrity_age_during_season', 'avg_score', 'avg_vote']
    for col in cols_to_numeric:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')
            
    # 删除空值
    original_len = len(data)
    data = data.dropna()
    print(f"数据清洗完成: 从 {original_len} 行减少到 {len(data)} 行")
    
    # 打印 avg_vote 的基本统计，确保数据正常
    print(f"avg_vote 统计: Mean={data['avg_vote'].mean():.4f}, Std={data['avg_vote'].std():.4f}")
    
    return data

# ==========================================
# 2. 目标编码与标准化 (关键修改)
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
    
    # 1. 类别变量目标编码
    categorical_cols = [
        'celebrity_name', 'ballroom_partner', 'celebrity_industry', 
        'celebrity_homestate', 'celebrity_homecountry/region'
    ]
    
    print("正在应用目标编码...")
    for col in categorical_cols:
        if col in df_encoded.columns:
            counts = df_encoded[col].value_counts()
            threshold = 3
            rare = counts[counts < threshold].index
            df_encoded[col] = df_encoded[col].apply(lambda x: 'Others' if x in rare else x)
            df_encoded[col] = target_encode_variable(df_encoded, col, 'placement')
            
    # 2. [关键] 标准化 (Standardization)
    # LiNGAM 对量纲敏感，标准化有助于正确估计系数和因果方向
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df_encoded), columns=df_encoded.columns)
    
    print("数据已标准化 (Z-Score Normalization)")
    return df_scaled

# ==========================================
# 3. Bootstrap LiNGAM
# ==========================================

def run_bootstrap_lingam(df, n_sampling=50):
    print(f"\n启动 Bootstrap DirectLiNGAM (采样次数: {n_sampling})...")
    model = lingam.DirectLiNGAM(random_state=42)
    result = model.bootstrap(df, n_sampling=n_sampling)
    return result.adjacency_matrices_

# ==========================================
# 4. 公平性计算 (逻辑优化)
# ==========================================

def calculate_fairness_metrics(matrices, labels, prob_threshold=0.2):
    """
    prob_threshold: 边存在的概率阈值 (0.2 表示 50 次采样中至少出现 10 次)
    """
    print("\n" + "="*50)
    print("正在进行因果公平性分析 (标准化数据)")
    print("="*50)

    n_features = len(labels)
    
    # 1. 计算概率矩阵 (Probability Matrix)
    probability_matrix = np.zeros((n_features, n_features))
    for matrix in matrices:
        # 只要系数非零 (LiNGAM 输出通常是非零即为边)
        adj = np.abs(matrix) > 0.0001 
        probability_matrix += adj.astype(float)
    probability_matrix /= len(matrices)
    
    # 2. 计算平均系数矩阵 (Coefficient Matrix)
    # 仅对存在的边计算平均值，或者直接取总体平均
    mean_matrix = np.mean(matrices, axis=0)
    
    # 3. 构建 NetworkX 图
    # 策略：如果 概率 > 阈值，则认为边存在，权重取平均系数
    G = nx.DiGraph()
    for label in labels:
        G.add_node(label)
        
    edge_count = 0
    # LiNGAM: row = B * col (col -> row)
    for r in range(n_features):
        for c in range(n_features):
            if r == c: continue
            
            prob = probability_matrix[r, c]
            weight = mean_matrix[r, c]
            
            if prob > prob_threshold:
                # 添加边：col -> row
                G.add_edge(labels[c], labels[r], weight=weight)
                edge_count += 1
                
    print(f"基于概率阈值 {prob_threshold} 构建因果图，共 {edge_count} 条边。")

    # 4. 定义因素
    fair_factors = ['celebrity_industry', 'celebrity_age_during_season']
    unfair_factors = ['ballroom_partner', 'celebrity_homestate']
    targets = ['avg_score', 'avg_vote']
    
    # 5. 计算总效应函数
    def get_total_effect(source, target):
        if source not in G or target not in G:
            return 0.0
        
        total_effect = 0.0
        try:
            # 找出所有简单路径 (限制长度防止死循环，虽然 DAG 无环)
            paths = list(nx.all_simple_paths(G, source, target, cutoff=6))
            
            for path in paths:
                path_effect = 1.0
                # 串联求积
                for i in range(len(path) - 1):
                    u, v = path[i], path[i+1]
                    path_effect *= G[u][v]['weight']
                # 并联求和
                total_effect += path_effect
        except Exception as e:
            return 0.0
        return total_effect

    # 6. 分析循环
    results = {}
    
    for target in targets:
        if target not in labels:
            continue
            
        print(f"\n>>> 分析目标变量: [{target}]")
        
        # 调试：检查是否有入边
        in_edges = list(G.in_edges(target, data=True))
        if not in_edges:
            print(f"  [警告] 图中没有指向 {target} 的边！因果效应将全为 0。")
            print(f"  可能原因：该变量被判定为因果链的源头，或与其他变量独立。")
            continue
        else:
            print(f"  发现 {len(in_edges)} 条直接指向 {target} 的边。")
        
        fair_effect_sum = 0.0
        unfair_effect_sum = 0.0
        
        print(f"  [公平因素 ({', '.join(fair_factors)})]:")
        for factor in fair_factors:
            if factor in labels:
                eff = get_total_effect(factor, target)
                fair_effect_sum += abs(eff)
                print(f"    - {factor}: {eff:.4f} (Abs: {abs(eff):.4f})")
            
        print(f"  [不公平因素 ({', '.join(unfair_factors)})]:")
        for factor in unfair_factors:
            if factor in labels:
                eff = get_total_effect(factor, target)
                unfair_effect_sum += abs(eff)
                print(f"    - {factor}: {eff:.4f} (Abs: {abs(eff):.4f})")
            
        total_influence = fair_effect_sum + unfair_effect_sum
        if total_influence == 0:
            fairness_score = 0.0
            print("  [结果] 总影响力为 0，无法计算公平度占比。")
        else:
            fairness_score = fair_effect_sum / total_influence
            print("-" * 30)
            print(f"  >> {target} 的公平度 (Fairness Score): {fairness_score:.2%}")
            print(f"     (公平因素占比: {fair_effect_sum:.4f} / {total_influence:.4f})")
            
    return G

# ==========================================
# 5. 可视化
# ==========================================

def visualize_graph_from_object(G, prob_threshold):
    plt.figure(figsize=(15, 12))
    pos = nx.spring_layout(G, k=2.0, iterations=200, seed=42)
    
    # 节点
    node_size = 3000
    nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color='#F0F8FF', edgecolors='#4682B4')
    
    # 标签
    formatted_labels = {n: n.replace('_', '\n') if len(n)>10 else n for n in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=formatted_labels, font_size=10, font_weight='bold')
    
    # 边
    edges = G.edges(data=True)
    if edges:
        # 边的粗细对应权重的绝对值
        weights = [abs(d['weight']) * 3 for u, v, d in edges]
        # 边的颜色对应权重的正负 (红正蓝负)
        edge_colors = [d['weight'] for u, v, d in edges]
        
        nx.draw_networkx_edges(
            G, pos, width=weights, arrowsize=20, arrowstyle='-|>',
            edge_color=edge_colors, edge_cmap=plt.cm.RdBu, edge_vmin=-1, edge_vmax=1,
            connectionstyle="arc3,rad=0.1", node_size=node_size
        )
        
        edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in edges}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=9)
    
    plt.title(f"Causal Fairness Graph (Standardized)\nProb Threshold > {prob_threshold}", fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# ==========================================
# 主程序
# ==========================================

if __name__ == "__main__":
    file_path = '/Users/xuyunpeng/Desktop/美赛c题2/season_3_27_with_votes.csv'

    try:
        # 1. 加载
        data = load_and_preprocess(file_path)
        
        # 2. 预处理 (含标准化)
        df_encoded = preprocess_features(data)
        
        # 3. 运行算法
        matrices = run_bootstrap_lingam(df_encoded, n_sampling=50)
        
        # 4. 计算公平性 (使用较低的阈值 0.2 以捕获 avg_vote 的连接)
        # 注意：这里返回的 G 是已经基于阈值构建好的图
        G = calculate_fairness_metrics(matrices, df_encoded.columns.tolist(), prob_threshold=0.2)
        
        # 5. 可视化
        visualize_graph_from_object(G, prob_threshold=0.2)
        
    except FileNotFoundError:
        print(f"错误: 找不到文件 '{file_path}'。")
    except Exception as e:
        import traceback
        traceback.print_exc()