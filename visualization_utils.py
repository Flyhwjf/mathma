"""
可视化工具模块
用于展示聚类、优化和迭代过程的结果
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Dict, Any


def plot_clustering_results(
    node_coords: np.ndarray,
    cluster_labels: np.ndarray,
    a_i: np.ndarray,
    b_i: np.ndarray,
    title: str = "时空聚类结果"
) -> plt.Figure:
    """
    绘制时空聚类结果图

    参数:
        node_coords: 节点坐标数组 (n x 2)
        cluster_labels: 聚类标签数组 (n,)
        a_i: 时间窗口开始时间数组 (n,)
        b_i: 时间窗口结束时间数组 (n,)
        title: 图表标题 (默认"时空聚类结果")

    返回:
        fig: matplotlib图形对象
    """
    # 设置样式
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")

    # 创建图形
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # 获取唯一的聚类标签
    unique_clusters = np.unique(cluster_labels)
    n_clusters = len(unique_clusters)

    # 1. 2D散点图（按聚类着色）
    for i, cluster_id in enumerate(unique_clusters):
        mask = cluster_labels == cluster_id
        cluster_coords = node_coords[mask]
        ax1.scatter(cluster_coords[:, 0], cluster_coords[:, 1],
                   label=f'聚类 {cluster_id}', s=50, alpha=0.7)

    ax1.set_xlabel('X 坐标')
    ax1.set_ylabel('Y 坐标')
    ax1.set_title('空间分布（按聚类着色）')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)

    # 2. 时间窗口可视化
    # 按聚类分组显示时间窗口
    colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))

    # 为每个节点创建位置（按聚类分组）
    y_positions = []
    cluster_groups = []
    current_y = 0

    for cluster_id in unique_clusters:
        mask = cluster_labels == cluster_id
        cluster_size = np.sum(mask)
        y_pos = np.arange(current_y, current_y + cluster_size)
        y_positions.extend(y_pos)
        cluster_groups.extend([cluster_id] * cluster_size)
        current_y += cluster_size + 2  # 添加间距

    y_positions = np.array(y_positions)
    cluster_groups = np.array(cluster_groups)

    # 绘制时间窗口误差条
    for i, cluster_id in enumerate(unique_clusters):
        mask = cluster_groups == cluster_id
        cluster_y = y_positions[mask]
        cluster_a = a_i[cluster_labels == cluster_id]
        cluster_b = b_i[cluster_labels == cluster_id]

        # 计算窗口中心点和宽度
        centers = (cluster_a + cluster_b) / 2
        widths = cluster_b - cluster_a

        ax2.errorbar(centers, cluster_y, xerr=[centers - cluster_a, cluster_b - centers],
                    fmt='o', color=colors[i], alpha=0.7, label=f'聚类 {cluster_id}',
                    capsize=3)

    ax2.set_xlabel('时间')
    ax2.set_ylabel('节点（按聚类分组）')
    ax2.set_title('时间窗口分布')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)

    # 调整布局
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()

    return fig


def plot_window_optimization(
    cluster_points: List[int],
    optimization_history: List[Dict[str, Any]],
    title: str = "滑动窗口优化过程"
) -> plt.Figure:
    """
    绘制滑动窗口优化过程图

    参数:
        cluster_points: 聚类内节点索引列表
        optimization_history: 优化历史记录列表
        title: 图表标题 (默认"滑动窗口优化过程")

    返回:
        fig: matplotlib图形对象
    """
    # 设置样式
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")

    # 创建图形
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # 1. 窗口位置可视化
    n_points = len(cluster_points)
    window_positions = []
    window_sizes = []
    costs = []

    for record in optimization_history:
        if 'window_start' in record and 'window_end' in record:
            window_positions.append((record['window_start'], record['window_end']))
            window_sizes.append(record['window_end'] - record['window_start'])
        if 'cost' in record:
            costs.append(record['cost'])

    # 绘制窗口滑动过程
    if window_positions:
        starts, ends = zip(*window_positions)
        iterations = range(len(window_positions))

        # 绘制窗口边界
        ax1.plot(iterations, starts, 'b-', label='窗口开始位置', alpha=0.7)
        ax1.plot(iterations, ends, 'r-', label='窗口结束位置', alpha=0.7)

        # 填充窗口区域
        for i in iterations:
            ax1.fill_between([i-0.4, i+0.4], starts[i], ends[i],
                            alpha=0.2, color='gray')

        ax1.set_xlabel('优化步骤')
        ax1.set_ylabel('节点索引')
        ax1.set_title('窗口滑动过程')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)

        # 设置y轴范围
        ax1.set_ylim(-1, n_points)

    # 2. 成本降低曲线
    if costs:
        ax2.plot(range(len(costs)), costs, 'g-', linewidth=2, marker='o', markersize=4)
        ax2.set_xlabel('优化步骤')
        ax2.set_ylabel('路径成本')
        ax2.set_title('成本降低过程')
        ax2.grid(True, alpha=0.3)

        # 标记最佳成本
        min_cost = min(costs)
        min_idx = costs.index(min_cost)
        ax2.plot(min_idx, min_cost, 'r*', markersize=12, label=f'最佳成本: {min_cost:.2f}')
        ax2.legend(loc='best')

    # 如果没有优化历史数据，显示提示
    if not optimization_history:
        ax1.text(0.5, 0.5, '无优化历史数据', ha='center', va='center',
                transform=ax1.transAxes, fontsize=12)
        ax1.set_title('窗口滑动过程')
        ax2.text(0.5, 0.5, '无成本数据', ha='center', va='center',
                transform=ax2.transAxes, fontsize=12)
        ax2.set_title('成本降低过程')

    # 调整布局
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()

    return fig


def plot_iteration_progress(
    iteration_history: List[Dict[str, Any]],
    title: str = "迭代改进跟踪图"
) -> plt.Figure:
    """
    绘制迭代改进跟踪图

    参数:
        iteration_history: 迭代历史记录列表
        title: 图表标题 (默认"迭代改进跟踪图")

    返回:
        fig: matplotlib图形对象
    """
    # 设置样式
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")

    # 创建图形
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 提取数据
    iterations = range(len(iteration_history))
    total_costs = []
    distances = []
    penalties = []
    cluster_changes = []

    for record in iteration_history:
        total_costs.append(record.get('total_cost', 0))
        distances.append(record.get('total_distance', 0))
        penalties.append(record.get('total_penalty', 0))
        cluster_changes.append(record.get('cluster_changes', 0))

    # 1. 综合成本收敛曲线
    ax1 = axes[0, 0]
    ax1.plot(iterations, total_costs, 'b-', linewidth=2, marker='o', markersize=4)
    ax1.set_xlabel('迭代次数')
    ax1.set_ylabel('综合成本')
    ax1.set_title('综合成本收敛曲线')
    ax1.grid(True, alpha=0.3)

    # 标记最佳成本
    if total_costs:
        min_cost = min(total_costs)
        min_idx = total_costs.index(min_cost)
        ax1.plot(min_idx, min_cost, 'r*', markersize=12, label=f'最佳成本: {min_cost:.2f}')
        ax1.legend(loc='best')

    # 2. 距离和惩罚项分解
    ax2 = axes[0, 1]
    if distances and penalties:
        width = 0.35
        x = np.arange(len(iterations))
        ax2.bar(x - width/2, distances, width, label='总距离', alpha=0.7)
        ax2.bar(x + width/2, penalties, width, label='总惩罚', alpha=0.7)
        ax2.set_xlabel('迭代次数')
        ax2.set_ylabel('成本分量')
        ax2.set_title('距离和惩罚项分解')
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3, axis='y')

    # 3. 聚类变化次数
    ax3 = axes[1, 0]
    if cluster_changes:
        ax3.plot(iterations, cluster_changes, 'g-', linewidth=2, marker='s', markersize=4)
        ax3.set_xlabel('迭代次数')
        ax3.set_ylabel('聚类变化次数')
        ax3.set_title('聚类稳定性')
        ax3.grid(True, alpha=0.3)

        # 计算平均变化
        avg_changes = np.mean(cluster_changes)
        ax3.axhline(y=avg_changes, color='r', linestyle='--', alpha=0.5,
                   label=f'平均变化: {avg_changes:.2f}')
        ax3.legend(loc='best')

    # 4. 成本降低百分比
    ax4 = axes[1, 1]
    if len(total_costs) > 1:
        initial_cost = total_costs[0]
        cost_reductions = [(initial_cost - cost) / initial_cost * 100 for cost in total_costs]

        ax4.plot(iterations, cost_reductions, 'm-', linewidth=2, marker='^', markersize=4)
        ax4.set_xlabel('迭代次数')
        ax4.set_ylabel('成本降低百分比 (%)')
        ax4.set_title('成本降低百分比')
        ax4.grid(True, alpha=0.3)

        # 标记最大降低
        max_reduction = max(cost_reductions)
        max_idx = cost_reductions.index(max_reduction)
        ax4.plot(max_idx, max_reduction, 'r*', markersize=12,
                label=f'最大降低: {max_reduction:.1f}%')
        ax4.legend(loc='best')

    # 如果没有历史数据，显示提示
    if not iteration_history:
        for ax in axes.flat:
            ax.text(0.5, 0.5, '无迭代历史数据', ha='center', va='center',
                   transform=ax.transAxes, fontsize=12)
            ax.set_title('')

    # 调整布局
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()

    return fig


def plot_hierarchical_path(
    inter_cluster_path: List[int],
    intra_cluster_paths: Dict[int, List[int]],
    node_coords: np.ndarray,
    title: str = "分层路径展示图"
) -> plt.Figure:
    """
    绘制分层路径展示图

    参数:
        inter_cluster_path: 聚类间路径序列
        intra_cluster_paths: 聚类内路径字典 {cluster_id: path}
        node_coords: 节点坐标数组 (n x 2)
        title: 图表标题 (默认"分层路径展示图")

    返回:
        fig: matplotlib图形对象
    """
    # 设置样式
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")

    # 创建图形
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # 1. 聚类间路径
    if inter_cluster_path:
        # 获取聚类中心（使用第一个节点作为代表）
        cluster_centers = {}
        for cluster_id, path in intra_cluster_paths.items():
            if path:
                # 使用路径中第一个节点的坐标作为聚类中心
                first_node = path[0]
                cluster_centers[cluster_id] = node_coords[first_node]

        # 绘制聚类中心
        cluster_ids = list(cluster_centers.keys())
        center_coords = np.array([cluster_centers[cid] for cid in cluster_ids])

        # 绘制聚类中心点
        ax1.scatter(center_coords[:, 0], center_coords[:, 1],
                   c=cluster_ids, cmap='tab20', s=200, alpha=0.7,
                   edgecolors='black', linewidth=1.5)

        # 标注聚类ID
        for i, cluster_id in enumerate(cluster_ids):
            ax1.text(center_coords[i, 0], center_coords[i, 1],
                    f'C{cluster_id}', fontsize=10, fontweight='bold',
                    ha='center', va='center', color='white')

        # 绘制聚类间路径
        if len(inter_cluster_path) > 1:
            inter_path_coords = []
            for cluster_id in inter_cluster_path:
                if cluster_id in cluster_centers:
                    inter_path_coords.append(cluster_centers[cluster_id])

            if len(inter_path_coords) > 1:
                inter_path_coords = np.array(inter_path_coords)
                ax1.plot(inter_path_coords[:, 0], inter_path_coords[:, 1],
                        'r--', linewidth=2, alpha=0.7, label='聚类间路径')

                # 添加箭头显示方向
                for i in range(len(inter_path_coords) - 1):
                    dx = inter_path_coords[i+1, 0] - inter_path_coords[i, 0]
                    dy = inter_path_coords[i+1, 1] - inter_path_coords[i, 1]
                    ax1.arrow(inter_path_coords[i, 0], inter_path_coords[i, 1],
                             dx*0.8, dy*0.8, head_width=0.5, head_length=0.5,
                             fc='red', ec='red', alpha=0.7)

        ax1.set_xlabel('X 坐标')
        ax1.set_ylabel('Y 坐标')
        ax1.set_title('聚类间路径')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)

    # 2. 聚类内路径
    ax2.set_title('聚类内详细路径')

    if intra_cluster_paths:
        # 使用不同的颜色和样式
        colors = plt.cm.tab20(np.linspace(0, 1, len(intra_cluster_paths)))
        markers = ['o', 's', '^', 'v', '<', '>', 'p', '*', 'h', 'D']

        for idx, (cluster_id, path) in enumerate(intra_cluster_paths.items()):
            if path:
                # 获取路径节点坐标
                path_coords = node_coords[path]

                # 绘制路径节点
                ax2.scatter(path_coords[:, 0], path_coords[:, 1],
                           c=[colors[idx]], marker=markers[idx % len(markers)],
                           s=100, alpha=0.7, edgecolors='black', linewidth=1,
                           label=f'聚类 {cluster_id}')

                # 绘制路径连线
                if len(path) > 1:
                    ax2.plot(path_coords[:, 0], path_coords[:, 1],
                            color=colors[idx], linestyle='-', linewidth=1.5, alpha=0.5)

                    # 添加箭头显示方向
                    for i in range(len(path_coords) - 1):
                        dx = path_coords[i+1, 0] - path_coords[i, 0]
                        dy = path_coords[i+1, 1] - path_coords[i, 1]
                        ax2.arrow(path_coords[i, 0], path_coords[i, 1],
                                 dx*0.8, dy*0.8, head_width=0.3, head_length=0.3,
                                 fc=colors[idx], ec=colors[idx], alpha=0.5)

                # 标注节点序号
                for i, node_idx in enumerate(path):
                    ax2.text(path_coords[i, 0], path_coords[i, 1],
                            str(node_idx), fontsize=8, ha='center', va='center')

        ax2.set_xlabel('X 坐标')
        ax2.set_ylabel('Y 坐标')
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)

    # 如果没有数据，显示提示
    if not inter_cluster_path and not intra_cluster_paths:
        ax1.text(0.5, 0.5, '无聚类间路径数据', ha='center', va='center',
                transform=ax1.transAxes, fontsize=12)
        ax2.text(0.5, 0.5, '无聚类内路径数据', ha='center', va='center',
                transform=ax2.transAxes, fontsize=12)

    # 调整布局
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()

    return fig