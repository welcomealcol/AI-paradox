import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.stats import chi2_contingency, mannwhitneyu
import platform

# ==================== 设置中文字体（解决图表中文乱码） ====================
# 根据操作系统选择合适的中文字体
system = platform.system()
if system == 'Windows':
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']  # 微软雅黑或黑体
elif system == 'Darwin':  # macOS
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Heiti TC']
else:  # Linux
    plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'Noto Sans CJK SC']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题

# ==================== 1. 读取数据（只取前7列，忽略Prompt列） ====================
def read_csv_ignore_last_column(filepath):
    """
    读取CSV文件，只取前7列（ID,Group,Major,Turn,CI,V,R），
    丢弃Prompt列（因为该列内部可能包含逗号导致列数增多，但我们不需要该列）。
    """
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        # 处理表头：前7列的实际名称
        header_parts = lines[0].strip().split(',')
        col_names = header_parts[:7]  # ID, Group, Major, Turn, CI, V, R
        # 处理数据行
        for line in lines[1:]:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            if len(parts) >= 7:
                first7 = parts[:7]
                data.append(first7)
            else:
                # 如果不足7列，跳过该行（异常）
                continue
    df = pd.DataFrame(data, columns=col_names)
    # 转换数值列
    df['Turn'] = df['Turn'].astype(int)
    return df

df = read_csv_ignore_last_column('duihua_merged_cleaned.csv')
print(f"总记录数: {len(df)}")
print(df.head())

# 分组
low_df = df[df['Group'] == 'Low'].copy()
high_df = df[df['Group'] == 'High'].copy()
print(f"Low组记录数: {len(low_df)}")
print(f"High组记录数: {len(high_df)}")
print()

# ==================== 2. 描述性统计：CI编码比例 ====================
def ci_proportions(df_group):
    ci_counts = df_group['CI'].value_counts()
    total = len(df_group)
    props = {ci: ci_counts.get(ci, 0)/total for ci in ['L1','L2','L3','L4']}
    return props

low_ci = ci_proportions(low_df)
high_ci = ci_proportions(high_df)

print("=== CI编码比例 ===")
print(f"Low组: L1={low_ci.get('L1',0):.1%}, L2={low_ci.get('L2',0):.1%}, L3={low_ci.get('L3',0):.1%}, L4={low_ci.get('L4',0):.1%}")
print(f"High组: L1={high_ci.get('L1',0):.1%}, L2={high_ci.get('L2',0):.1%}, L3={high_ci.get('L3',0):.1%}, L4={high_ci.get('L4',0):.1%}")
print()

# ==================== 3. 序列分析：状态转移矩阵 ====================
def build_transition_matrix(df_group):
    """
    构建状态转移矩阵（状态 = CI_V 组合）
    返回: 概率矩阵(DataFrame), 原始计数字典
    """
    transitions = {}
    # 按学生ID分组，确保顺序
    for sid, group in df_group.groupby('ID'):
        group = group.sort_values('Turn')
        states = [f"{row['CI']}_{row['V']}" for _, row in group.iterrows()]
        for i in range(len(states)-1):
            from_state = states[i]
            to_state = states[i+1]
            transitions[(from_state, to_state)] = transitions.get((from_state, to_state), 0) + 1
    
    all_states = sorted(set([s for (s,_) in transitions.keys()] + [s for (_,s) in transitions.keys()]))
    prob_mat = pd.DataFrame(0.0, index=all_states, columns=all_states)
    for (fr, to), cnt in transitions.items():
        prob_mat.loc[fr, to] = cnt
    # 归一化
    for state in all_states:
        row_sum = prob_mat.loc[state].sum()
        if row_sum > 0:
            prob_mat.loc[state] = prob_mat.loc[state] / row_sum
    return prob_mat, transitions

low_prob, low_counts = build_transition_matrix(low_df)
high_prob, high_counts = build_transition_matrix(high_df)

print("=== Low组状态转移矩阵（概率）===")
print(low_prob.round(2))
print("\n=== High组状态转移矩阵（概率）===")
print(high_prob.round(2))
print()

# 显示最频繁的转移路径
def top_paths(trans_counts, top_n=5):
    sorted_items = sorted(trans_counts.items(), key=lambda x: x[1], reverse=True)
    return sorted_items[:top_n]

print("Low组最频繁的5条转移路径:")
for (fr, to), cnt in top_paths(low_counts, 5):
    print(f"  {fr} -> {to} : {cnt}次")
print("\nHigh组最频繁的5条转移路径:")
for (fr, to), cnt in top_paths(high_counts, 5):
    print(f"  {fr} -> {to} : {cnt}次")
print()

# ==================== 4. 可视化行为路径图（优化布局和中文显示） ====================
def plot_path_diagram(trans_counts, title, filename, min_weight=2):
    """
    绘制行为转移路径图（全部直线箭头）。
    - 单向边：居中带箭头。
    - 双向边：两条平行直线，间距明显，各自带相反方向的箭头。
    - 箭头从节点圆边界开始/结束。
    """
    import matplotlib.pyplot as plt
    import networkx as nx
    import numpy as np
    from matplotlib.patches import FancyArrowPatch

    G = nx.MultiDiGraph()
    for (fr, to), cnt in trans_counts.items():
        if cnt >= min_weight:
            G.add_edge(fr, to, weight=cnt)

    if len(G.nodes) == 0:
        print(f"警告: {title} 中没有权重≥{min_weight}的边，无法绘图")
        return

    pos = nx.kamada_kawai_layout(G)
    fig, ax = plt.subplots(figsize=(20, 16), dpi=150)

    # 绘制节点
    node_size = 3000
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=node_size,
                           node_color='lightblue', edgecolors='black', linewidths=1.5)
    labels = {node: node.replace('_', '\n') for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, ax=ax, labels=labels, font_size=10,
                            font_family=plt.rcParams['font.sans-serif'][0])

    # 计算节点半径（数据坐标单位）：基于平均边长的 5% 左右
    all_points = np.array(list(pos.values()))
    if len(all_points) > 1:
        avg_edge_len = np.mean([np.hypot(pos[u][0]-pos[v][0], pos[u][1]-pos[v][1]) 
                                for u,v in G.edges()])
        radius = avg_edge_len * 0.08  # 半径约为平均边长的8%
    else:
        radius = 0.05
    radius = max(0.03, min(radius, 0.1))  # 限制范围

    # 收集所有边及其权重
    edge_weights = {}
    for (u, v, k, d) in G.edges(keys=True, data=True):
        edge_weights[(u, v)] = d['weight']

    # 找出所有边对
    processed = set()
    edges_to_draw = []  # (u, v, weight, is_bidirectional, offset_sign)
    for (u, v), w in edge_weights.items():
        if (u, v) in processed:
            continue
        if (v, u) in edge_weights:
            w_rev = edge_weights[(v, u)]
            edges_to_draw.append((u, v, w, True, +1))
            edges_to_draw.append((v, u, w_rev, True, -1))
            processed.add((u, v))
            processed.add((v, u))
        else:
            edges_to_draw.append((u, v, w, False, 0))
            processed.add((u, v))

    all_weights = [w for (_, _, w, _, _) in edges_to_draw]
    max_weight = max(all_weights) if all_weights else 1

    # 垂直单位向量
    def get_perp_unit(p1, p2):
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        length = np.hypot(dx, dy)
        if length == 0:
            return (0, 0)
        ux = dx / length
        uy = dy / length
        return (uy, -ux)  # 顺时针垂直

    # 计算偏移距离：基于平均边长的 15%，确保间距明显
    if len(all_points) > 1:
        avg_len = np.mean([np.hypot(pos[u][0]-pos[v][0], pos[u][1]-pos[v][1]) 
                           for u,v in G.edges()])
        offset_distance = avg_len * 0.12
    else:
        offset_distance = 0.08
    offset_distance = max(0.05, offset_distance)  # 至少0.05

    # 绘制每条边
    for (u, v, weight, is_bidir, offset_sign) in edges_to_draw:
        p1 = pos[u]
        p2 = pos[v]
        width = 1 + 5 * (weight / max_weight)

        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        length = np.hypot(dx, dy)
        if length < 1e-6:
            continue
        ux = dx / length
        uy = dy / length

        # 起点和终点（圆边界点）
        start = (p1[0] + ux * radius, p1[1] + uy * radius)
        end = (p2[0] - ux * radius, p2[1] - uy * radius)

        if is_bidir:
            perp = get_perp_unit(p1, p2)
            offset = (offset_distance * offset_sign * perp[0],
                      offset_distance * offset_sign * perp[1])
            start = (start[0] + offset[0], start[1] + offset[1])
            end = (end[0] + offset[0], end[1] + offset[1])

        arrow = FancyArrowPatch(start, end,
                                arrowstyle='->',
                                mutation_scale=20,
                                connectionstyle='arc3,rad=0.0',
                                linewidth=width, color='gray', alpha=0.8)
        ax.add_patch(arrow)

        mid = ((start[0] + end[0])/2, (start[1] + end[1])/2)
        ax.text(mid[0], mid[1], str(weight),
                fontsize=9,
                fontfamily=plt.rcParams['font.sans-serif'][0],
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.85, pad=2),
                ha='center', va='center')

    plt.title(title, fontsize=16, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"路径图已保存为 {filename}")

plot_path_diagram(low_counts, "Behavioral Transition Paths of the Low Usage Group (Weight ≥ 2)", "low_paths.png", min_weight=2)
plot_path_diagram(high_counts, "Behavioral Transition Paths of the High Usage Group (Weight ≥ 2)", "high_paths.png", min_weight=2)

# ==================== 5. 假设检验 ====================
# 5.1 卡方检验：深度加工(L3+L4) vs 浅层加工(L1+L2)
low_deep = ((low_df['CI'] == 'L3') | (low_df['CI'] == 'L4')).sum()
low_shallow = len(low_df) - low_deep
high_deep = ((high_df['CI'] == 'L3') | (high_df['CI'] == 'L4')).sum()
high_shallow = len(high_df) - high_deep
contingency = [[low_deep, low_shallow], [high_deep, high_shallow]]
chi2, p_chi, dof, expected = chi2_contingency(contingency)

print("=== 假设检验 ===")
print("深度加工(L3+L4) vs 浅层加工(L1+L2) 卡方检验:")
print(f"  Low组: 深度={low_deep}, 浅层={low_shallow}")
print(f"  High组: 深度={high_deep}, 浅层={high_shallow}")
print(f"  chi2={chi2:.3f}, p={p_chi:.5f}")
if p_chi < 0.001:
    print("  -> 差异极显著 (p<0.001)")
else:
    print(f"  -> p={p_chi:.3f}")
print()

# 5.2 Mann-Whitney U检验：验证行为得分 (V0=0, V1=1, V2=2, V3=3)
v_map = {'V0':0, 'V1':1, 'V2':2, 'V3':3}
low_v_scores = low_df['V'].map(v_map).dropna()
high_v_scores = high_df['V'].map(v_map).dropna()
u_stat, p_mw = mannwhitneyu(low_v_scores, high_v_scores, alternative='two-sided')

print("验证行为得分 Mann-Whitney U检验:")
print(f"  Low组平均得分: {low_v_scores.mean():.3f} (中位数: {low_v_scores.median()})")
print(f"  High组平均得分: {high_v_scores.mean():.3f} (中位数: {high_v_scores.median()})")
print(f"  U={u_stat:.1f}, p={p_mw:.5f}")
if p_mw < 0.001:
    print("  -> 差异极显著 (p<0.001)")
print()

# 5.3 调节行为出现比例
low_r_present = (low_df['R'] != '').sum()
high_r_present = (high_df['R'] != '').sum()
print("调节行为(R)出现比例:")
print(f"  Low组: {low_r_present}/{len(low_df)} = {low_r_present/len(low_df):.1%}")
print(f"  High组: {high_r_present}/{len(high_df)} = {high_r_present/len(high_df):.1%}")