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
# 1. 数据加载与特征工程 (新增 avg_score_ratio)
# ==========================================

def load_and_preprocess(file_path):
    print(f"正在加载数据: {file_path} ...")
    df = pd.read_csv(file_path)
    
    # --- 1. 计算 avg_score (选手平均评委分) ---
    score_cols = [col for col in df.columns if 'score' in col.lower() and 'judge' in col.lower()]
    for col in score_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['avg_score'] = df[score_cols].mean(axis=1)
    
    # --- 2. 计算 avg_score_ratio (选手评委分占赛季总和的比例) ---
    # 按赛季分组计算该赛季所有选手的 avg_score 总和
    season_avg_score_sum = df.groupby('season')['avg_score'].transform('sum')
    # 计算占比，避免除以 0
    df['avg_score_ratio'] = df['avg_score'] / season_avg_score_sum.replace(0, np.nan)
    
    # --- 3. 计算 avg_vote (选手平均粉丝得票率) ---
    vote_cols = [col for col in df.columns if 'vote_share' in col]
    print(f"检测到 {len(vote_cols)} 个投票率列用于计算 avg_vote")
    
    for col in vote_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df['avg_vote'] = df[vote_cols].mean(axis=1)
    
    # --- 4. 选定感兴趣的列 ---
    columns = [
        'avg_score_ratio',  # 新增：评委分占比
        'avg_vote',         # 粉丝得票率均值
        'placement',        # 目标变量：赛季排名
        'season',           # 控制变量
        'ballroom_partner', # 可能的中介变量
        'celebrity_industry',# 可能的中介变量
        'celebrity_age_during_season' # 可能的中介变量
    ]
    
    existing_cols = [c for c in columns if c in df.columns]
    data = df[existing_cols].copy()
    
    # 确保数值列正确
    cols_to_numeric = ['placement', 'season', 'celebrity_age_during_season', 'avg_score_ratio', 'avg_vote']
    for col in cols_to_numeric:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')
            
    # 删除空值行
    original_len = len(data)
    data = data.dropna()
    print(f"数据清洗完成: 从 {original_len} 行减少到 {len(data)} 行")
    
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
    
    # 类别变量目标编码
    categorical_cols = ['ballroom_partner', 'celebrity_industry']
    
    print("正在应用目标编码...")
    for col in categorical_cols:
        if col in df_encoded.columns:
            counts = df_encoded[col].value_counts()
            threshold = 3
            rare = counts[counts < threshold].index
            df_encoded[col] = df_encoded[col].apply(lambda x: 'Others' if x in rare else x)
            df_encoded[col] = target_encode_variable(df_encoded, col, 'placement')
            
    # 标准化 (Standardization)
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
# 4. 计算总因果效应 (核心功能)
# ==========================================

def calculate_total_causal_effects(matrices, labels, prob_threshold=0.2):
    print("\n" + "="*60)
    print("正在计算 avg_score_ratio 和 avg_vote 对 placement 的总因果效应")
    print("="*60)

    n_features = len(labels)
    
    # 1. 计算概率矩阵 (边存在的概率)
    probability_matrix = np.zeros((n_features, n_features))
    for matrix in matrices:
        adj = np.abs(matrix) > 0.0001 
        probability_matrix += adj.astype(float)
    probability_matrix /= len(matrices)
    
    # 2. 计算平均系数矩阵 (边的权重)
    mean_matrix = np.mean(matrices, axis=0)
    
    # 3. 构建 NetworkX 图
    G = nx.DiGraph()
    for label in labels:
        G.add_node(label)
        
    for r in range(n_features):
        for c in range(n_features):
            if r == c: continue
            if probability_matrix[r, c] > prob_threshold:
                # LiNGAM: col -> row
                weight = mean_matrix[r, c]
                G.add_edge(labels[c], labels[r], weight=weight)

    # 4. 定义计算总因果效应的函数
    def get_total_effect(source, target):
        """
        计算从 source 到 target 的总因果效应，包括直接路径和所有间接路径
        """
        if source not in G or target not in G:
            return 0.0
        
        total_effect = 0.0
        try:
            # 找出所有简单路径 (限制长度防止死循环)
            paths = list(nx.all_simple_paths(G, source, target, cutoff=5))
            
            for path in paths:
                path_effect = 1.0
                # 串联求积
                for i in range(len(path) - 1):
                    u, v = path[i], path[i+1]
                    path_effect *= G[u][v]['weight']
                # 并联求和
                total_effect += path_effect
                
        except Exception as e:
            print(f"计算路径时出错: {e}")
            return 0.0
            
        return total_effect

    # 5. 计算并输出结果
    target = 'placement'
    sources = ['avg_score_ratio', 'avg_vote']
    
    results = {}
    for source in sources:
        effect = get_total_effect(source, target)
        results[source] = effect
        
        print(f"\n{source} 对 {target} 的总因果效应: { - effect:.4f}")
        print(f"  解释: 当 {source} 增加 1 个标准差时，{target} 平均变化 { - effect:.4f} 个标准差")
    
    return G, results

# ==========================================
# 5. 可视化
# ==========================================

def visualize_causal_graph(G, results):
    plt.figure(figsize=(12, 10))
    
    # 布局
    pos = nx.spring_layout(G, k=2.0, iterations=200, seed=42)
    
    # 节点样式
    node_size = 3000
    nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color='#F0F8FF', edgecolors='#4682B4')
    
    # 节点标签
    formatted_labels = {n: n.replace('_', '\n') if len(n)>10 else n for n in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=formatted_labels, font_size=9, font_weight='bold')
    
    # 边样式
    edges = G.edges(data=True)
    if edges:
        # 边的粗细对应权重的绝对值
        weights = [abs(d['weight']) * 3 + 0.5 for u, v, d in edges]
        # 边的颜色对应权重的正负 (红正蓝负)
        edge_colors = [d['weight'] for u, v, d in edges]
        
        nx.draw_networkx_edges(
            G, pos, width=weights, arrowsize=20, arrowstyle='-|>',
            edge_color=edge_colors, edge_cmap=plt.cm.RdBu, edge_vmin=-1, edge_vmax=1,
            connectionstyle="arc3,rad=0.1", node_size=node_size
        )
        
        edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in edges}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
    
    # 添加总因果效应标注
    plt.title(f"Causal Effects on Placement\navg_score_ratio: {results['avg_score_ratio']:.2f}, avg_vote: {results['avg_vote']:.2f}", 
              fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# ==========================================
# 主程序
# ==========================================

if __name__ == "__main__":
    file_path = '/Users/xuyunpeng/Desktop/美赛c题2/season_3_27_with_votes.csv'  # 替换为你的数据文件路径

    try:
        # 1. 加载数据
        data = load_and_preprocess(file_path)
        
        # 2. 预处理 (目标编码与标准化)
        df_encoded = preprocess_features(data)
        
        # 3. 运行 Bootstrap LiNGAM
        matrices = run_bootstrap_lingam(df_encoded, n_sampling=50)
        
        # 4. 计算总因果效应
        G, results = calculate_total_causal_effects(matrices, df_encoded.columns.tolist(), prob_threshold=0.2)
        
        # 5. 可视化
        visualize_causal_graph(G, results)
        
    except FileNotFoundError:
        print(f"错误: 找不到文件 '{file_path}'。请确保文件路径正确。")
    except Exception as e:
        import traceback
        traceback.print_exc()