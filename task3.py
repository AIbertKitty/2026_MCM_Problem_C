import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import lingam

# ==========================================
# 1. 数据加载与特征工程 (增加 avg_score)
# ==========================================

def load_and_preprocess(file_path):
    df = pd.read_csv(file_path)
    
    # --- 新增功能：计算 avg_score --- 计算 avg_r
    # 找到所有包含 judge 分数的列 (例如 week1_judge1_score)
    score_cols = [col for col in df.columns if 'score' in col.lower() and 'judge' in col.lower()]
    
    # 转换为数值型，无法转换的变成 NaN
    for col in score_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # 计算每行的平均分 (忽略 NaN)
    df['avg_score'] = df[score_cols].mean(axis=1)
    
    # 选定感兴趣的列 (加入了 avg_score)
    columns = [
        'ballroom_partner', 
        'celebrity_industry', 
        'celebrity_homestate', 
        'celebrity_homecountry/region', 
        'celebrity_age_during_season', 
        'season', 
        'avg_score', # 新增节点
        'placement'
    ]
    
    existing_cols = [c for c in columns if c in df.columns]
    data = df[existing_cols].copy()
    
    # 确保数值列正确
    cols_to_numeric = ['placement', 'season', 'celebrity_age_during_season', 'avg_score']
    for col in cols_to_numeric:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')
            
    # 删除含有空值的行
    data = data.dropna()
    print(f"数据预处理完成，包含变量: {data.columns.tolist()}")
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
    model = lingam.DirectLiNGAM()
    result = model.bootstrap(df, n_sampling=n_sampling)
    return result.adjacency_matrices_

# ==========================================
# 4. 高可读性可视化 (核心修改)
# ==========================================

def visualize_readable_graph(matrices, labels, prob_threshold=0.25):
    """
    优化点：
    1. 强制显示 season。
    2. 解决箭头重叠 (node_size 参数)。
    3. 标签清晰度 (bbox)。
    """
    n_matrices = len(matrices)
    n_features = len(labels)
    
    # 计算概率矩阵
    probability_matrix = np.zeros((n_features, n_features))
    for matrix in matrices:
        adj = np.abs(matrix) > 0.01 
        probability_matrix += adj.astype(float)
    probability_matrix /= n_matrices
    
    # 构建图
    G = nx.DiGraph()
    
    # 先添加所有节点，确保 season 即使没连线也在图中
    for label in labels:
        G.add_node(label)
        
    # 添加边
    rows, cols = np.where(probability_matrix > prob_threshold)
    for r, c in zip(rows, cols):
        if r == c: continue
        prob = probability_matrix[r, c]
        # LiNGAM: col -> row
        G.add_edge(labels[c], labels[r], weight=prob)

    # 移除除了 season 以外的孤立节点 (可选，为了整洁)
    # 如果 season 是孤立的，我们保留它，其他的孤立节点删掉
    nodes_to_remove = []
    for node in G.nodes():
        if G.degree(node) == 0 and node != 'season':
            nodes_to_remove.append(node)
    G.remove_nodes_from(nodes_to_remove)

    print(f"\n=== 因果图统计 (阈值 > {prob_threshold:.0%}) ===")
    print(f"节点列表: {list(G.nodes())}")
    for u, v, d in G.edges(data=True):
        print(f"  {u} -> {v} (概率: {d['weight']:.2f})")

    # --- 绘图配置 ---
    plt.figure(figsize=(18, 14)) # 更大的画布
    
    # 布局：增加 k 值使节点更分散，增加 iterations 确保稳定
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
    
    # 节点标签
    # 将长标签换行，例如 celebrity_industry -> celebrity\nindustry
    formatted_labels = {}
    for node in G.nodes():
        if len(node) > 10:
            # 在下划线处换行
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
    weights = [d['weight'] * 4 for u, v, d in edges]
    edge_colors = [d['weight'] for u, v, d in edges]
    
    # 关键：使用 node_size 参数让箭头在节点边缘停止
    # connectionstyle="arc3,rad=0.15" 让边弯曲，避免重叠
    nx.draw_networkx_edges(
        G, pos, 
        width=weights, 
        arrowsize=25, 
        arrowstyle='-|>', # 锐利的箭头
        edge_color=edge_colors, 
        edge_cmap=plt.cm.Blues, 
        connectionstyle="arc3,rad=0.15", 
        node_size=node_size, # 告诉 networkx 节点有多大，箭头就会停在圆圈外！
        min_source_margin=20,
        min_target_margin=20,
        alpha=0.8
    )
    
    # --- 边标签 (概率) 优化 ---
    edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in edges}
    
    # 自定义标签绘制，确保背景不透明
    nx.draw_networkx_edge_labels(
        G, pos, 
        edge_labels=edge_labels, 
        font_size=10,
        font_weight='bold',
        font_color='#8B0000', # 深红色字体
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#CCCCCC", alpha=0.9), # 白色圆角背景框
        label_pos=0.5 # 标签在边的中点
    )
    
    plt.title(f"Causal Analysis Graph (With Avg Score)\nEdge Probability > {int(prob_threshold*100)}%", fontsize=18, pad=20)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# ==========================================
# 主程序
# ==========================================

file_path = '/Users/xuyunpeng/Desktop/美赛c题2/2026_MCM_Problem_C_Data.csv'

try:
    # 1. 加载 (含 avg_score 计算)
    data = load_and_preprocess(file_path)
    
    # 2. 预处理
    df_encoded = preprocess_features(data)
    
    # 3. 运行算法
    matrices = run_bootstrap_lingam(df_encoded, n_sampling=50)
    
    # 4. 可视化 (优化版)
    # 阈值设为 0.25，既能过滤噪声，又能保留 season 可能存在的弱连接
    visualize_readable_graph(matrices, df_encoded.columns, prob_threshold=0.25)
    
except Exception as e:
    import traceback
    traceback.print_exc()