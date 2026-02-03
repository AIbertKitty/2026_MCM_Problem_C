import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import lingam

# ==========================================
# 1. 数据加载与特征工程 (增加 avg_score 和 avg_vote)
# ==========================================

def load_and_preprocess(file_path):
    print(f"正在加载数据: {file_path} ...")
    df = pd.read_csv(file_path)
    
    # --- 原有功能：计算 avg_score (裁判分均值) ---
    score_cols = [col for col in df.columns if 'score' in col.lower() and 'judge' in col.lower()]
    for col in score_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['avg_score'] = df[score_cols].mean(axis=1)
    
    # --- 新增功能：计算 avg_vote (观众得票率均值) ---
    # 1. 识别 week1_vote_share 到 week11_vote_share 列
    # 这里使用字符串匹配，凡是包含 'vote_share' 的列都纳入计算
    vote_cols = [col for col in df.columns if 'vote_share' in col]
    
    print(f"检测到 {len(vote_cols)} 个投票率列: {vote_cols}")
    
    # 2. 转换为数值型 (无法转换的变成 NaN)
    for col in vote_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
    # 3. 计算每行的平均得票率 (Pandas 的 mean 默认会忽略 NaN，这符合处理空值的需求)
    # 如果某行所有周的投票率都是 NaN，结果将为 NaN
    df['avg_vote'] = df[vote_cols].mean(axis=1)
    
    # --- 选定感兴趣的列 (加入了 avg_score 和 avg_vote) ---
    columns = [
        'ballroom_partner', 
        'celebrity_industry', 
        'celebrity_homestate', 
        'celebrity_homecountry/region', 
        'celebrity_age_during_season', 
        'season', 
        'avg_score', # 裁判均分
        'avg_vote',  # 新增节点: 观众投票均分
        'placement'
    ]
    
    # 仅保留存在的列
    existing_cols = [c for c in columns if c in df.columns]
    data = df[existing_cols].copy()
    
    # 确保数值列正确
    cols_to_numeric = ['placement', 'season', 'celebrity_age_during_season', 'avg_score', 'avg_vote']
    for col in cols_to_numeric:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')
            
    # 删除含有空值的行
    # 注意：这里会删除那些 'avg_vote' 计算结果为 NaN 的行（即该选手没有任何一周的投票数据）
    # 以及其他关键特征缺失的行
    original_len = len(data)
    data = data.dropna()
    print(f"数据清洗完成: 从 {original_len} 行减少到 {len(data)} 行")
    print(f"最终包含变量: {data.columns.tolist()}")
    
    return data

# ==========================================
# 2. 目标编码 (Target Encoding)
# ==========================================

def target_encode_variable(df, col, target_col='placement'):
    # 使用 placement (排名) 作为目标进行编码
    global_mean = df[target_col].mean()
    agg = df.groupby(col)[target_col].agg(['count', 'mean'])
    counts = agg['count']
    means = agg['mean']
    m = 5 # 平滑系数
    smooth_means = (counts * means + m * global_mean) / (counts + m)
    return df[col].map(smooth_means)

def preprocess_features(data):
    df_encoded = data.copy()
    
    categorical_cols = [
        'celebrity_name', 'ballroom_partner', 'celebrity_industry', 
        'celebrity_homestate', 'celebrity_homecountry/region'
    ]
    
    print("正在应用目标编码...")
    for col in categorical_cols:
        if col in df_encoded.columns:
            # 过滤低频类别
            counts = df_encoded[col].value_counts()
            threshold = 2 if col == 'celebrity_name' else 3
            rare = counts[counts < threshold].index
            df_encoded[col] = df_encoded[col].apply(lambda x: 'Others' if x in rare else x)
            
            # 编码
            df_encoded[col] = target_encode_variable(df_encoded, col, 'placement')
            
    return df_encoded

# ==========================================
# 3. Bootstrap LiNGAM
# ==========================================

def run_bootstrap_lingam(df, n_sampling=50):
    print(f"\n启动 Bootstrap DirectLiNGAM (采样次数: {n_sampling})...")
    # 设置随机种子以复现结果
    model = lingam.DirectLiNGAM(random_state=42)
    result = model.bootstrap(df, n_sampling=n_sampling)
    return result.adjacency_matrices_

# ==========================================
# 4. 高可读性可视化
# ==========================================

def visualize_readable_graph(matrices, labels, prob_threshold=0.25):
    n_matrices = len(matrices)
    n_features = len(labels)
    
    # 计算概率矩阵
    probability_matrix = np.zeros((n_features, n_features))
    for matrix in matrices:
        # 这里的阈值 0.01 是为了过滤掉极小的数值噪声
        adj = np.abs(matrix) > 0.01 
        probability_matrix += adj.astype(float)
    probability_matrix /= n_matrices
    
    # 构建图
    G = nx.DiGraph()
    
    # 先添加所有节点
    for label in labels:
        G.add_node(label)
        
    # 添加边
    rows, cols = np.where(probability_matrix > prob_threshold)
    for r, c in zip(rows, cols):
        if r == c: continue
        prob = probability_matrix[r, c]
        # LiNGAM 的矩阵定义是: row = B * col，所以因果方向是 col -> row
        G.add_edge(labels[c], labels[r], weight=prob)

    # 移除孤立节点 (保留 season 以防万一)
    nodes_to_remove = []
    for node in G.nodes():
        if G.degree(node) == 0 and node != 'season':
            nodes_to_remove.append(node)
    G.remove_nodes_from(nodes_to_remove)

    print(f"\n=== 因果图统计 (边概率阈值 > {prob_threshold:.0%}) ===")
    print(f"节点列表: {list(G.nodes())}")
    for u, v, d in G.edges(data=True):
        print(f"  {u} -> {v} (概率: {d['weight']:.2f})")

    # --- 绘图配置 ---
    plt.figure(figsize=(18, 14))
    
    # 布局
    pos = nx.spring_layout(G, k=3.5, iterations=200, seed=10)
    
    # 节点样式
    node_size = 4500
    nx.draw_networkx_nodes(
        G, pos, 
        node_size=node_size, 
        node_color='#F0F8FF', # AliceBlue
        edgecolors='#4682B4', # SteelBlue
        linewidths=2.5,
        alpha=1.0
    )
    
    # 节点标签格式化 (换行)
    formatted_labels = {}
    for node in G.nodes():
        if len(node) > 10:
            formatted_labels[node] = node.replace('_', '\n')
        else:
            formatted_labels[node] = node
            
    nx.draw_networkx_labels(
        G, pos, 
        labels=formatted_labels,
        font_size=11, 
        font_family='sans-serif',
        font_weight='bold',
        font_color='#2F4F4F'
    )
    
    # --- 边与箭头优化 ---
    edges = G.edges(data=True)
    if edges:
        weights = [d['weight'] * 4 for u, v, d in edges]
        edge_colors = [d['weight'] for u, v, d in edges]
        
        nx.draw_networkx_edges(
            G, pos, 
            width=weights, 
            arrowsize=25, 
            arrowstyle='-|>', 
            edge_color=edge_colors, 
            edge_cmap=plt.cm.Blues, 
            connectionstyle="arc3,rad=0.15", 
            node_size=node_size, 
            min_source_margin=20,
            min_target_margin=20,
            alpha=0.8
        )
        
        # 边标签
        edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in edges}
        nx.draw_networkx_edge_labels(
            G, pos, 
            edge_labels=edge_labels, 
            font_size=10,
            font_weight='bold',
            font_color='#8B0000', 
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#CCCCCC", alpha=0.9),
            label_pos=0.5
        )
    
    plt.title(f"Causal Analysis Graph (With Avg Vote & Score)\nEdge Probability > {int(prob_threshold*100)}%", fontsize=18, pad=20)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# ==========================================
# 主程序
# ==========================================

if __name__ == "__main__":
    # 请确保文件名与你的本地文件一致
    file_path = '/Users/xuyunpeng/Desktop/美赛c题2/season_28_30_with_votes.csv'

    try:
        # 1. 加载 (含 avg_score 和 avg_vote 计算)
        data = load_and_preprocess(file_path)
        
        # 2. 预处理
        df_encoded = preprocess_features(data)
        
        # 3. 运行算法
        # 注意：LiNGAM 假设变量顺序可能影响结果，但 DirectLiNGAM 相对稳健
        matrices = run_bootstrap_lingam(df_encoded, n_sampling=50)
        
        # 4. 可视化
        # 阈值可调：0.25 表示在 50 次采样中至少出现 12 次才显示连线
        visualize_readable_graph(matrices, df_encoded.columns.tolist(), prob_threshold=0.3)
        
    except FileNotFoundError:
        print(f"错误: 找不到文件 '{file_path}'。请确保文件在当前目录下或修改 file_path 路径。")
    except Exception as e:
        print(f"发生错误: {e}")