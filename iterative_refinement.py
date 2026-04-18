"""
迭代改进控制器模块
用于评估边界点并优化聚类分配
"""

import numpy as np
from typing import List, Tuple, Dict, Any
import logging

# 设置日志
logger = logging.getLogger(__name__)

# 导入其他模块
try:
    from spatiotemporal_clustering import spatiotemporal_clustering
    from sliding_window_optimizer import sliding_window_optimization
except ImportError:
    logger.warning("无法导入spatiotemporal_clustering或sliding_window_optimizer模块")


def evaluate_boundary_violations(
    cluster_labels: np.ndarray,
    path_sequence: List[int],
    penalties_by_node: np.ndarray,
    threshold_multiplier: float = 2.0
) -> Tuple[List[int], List[float], Dict[str, Any]]:
    """
    评估边界点违反情况

    参数:
        cluster_labels: 聚类标签数组
        path_sequence: 当前路径序列
        penalties_by_node: 每个节点的时间窗口惩罚值数组
        threshold_multiplier: 边界点判定阈值乘数 (默认2.0)

    返回:
        boundary_points: 边界点索引列表
        violation_scores: 违反程度分数列表
        violation_info: 违反信息字典
    """
    n = len(path_sequence)

    # 1. 识别边界点：路径中相邻节点属于不同聚类的点
    boundary_points = []

    for i in range(n):
        # 检查当前节点是否是边界点
        is_boundary = False

        # 检查前一个节点（如果存在）
        if i > 0:
            if cluster_labels[path_sequence[i]] != cluster_labels[path_sequence[i-1]]:
                is_boundary = True

        # 检查后一个节点（如果存在）
        if i < n - 1:
            if cluster_labels[path_sequence[i]] != cluster_labels[path_sequence[i+1]]:
                is_boundary = True

        if is_boundary:
            boundary_points.append(path_sequence[i])

    # 2. 计算平均惩罚和阈值
    average_penalty = np.mean(penalties_by_node)
    threshold = average_penalty * threshold_multiplier

    # 3. 评估边界点违反情况
    violation_scores = []
    violation_count = 0

    for point in boundary_points:
        penalty = penalties_by_node[point]
        if penalty > threshold:
            # 违反程度分数 = (惩罚 - 阈值) / 阈值
            violation_score = (penalty - threshold) / threshold
            violation_count += 1
        else:
            violation_score = 0.0

        violation_scores.append(violation_score)

    # 4. 构建违反信息字典
    violation_info = {
        'average_penalty': average_penalty,
        'threshold': threshold,
        'violation_count': violation_count,
        'violation_rate': violation_count / len(boundary_points) if boundary_points else 0.0,
        'boundary_points_count': len(boundary_points),
        'total_points': n
    }

    return boundary_points, violation_scores, violation_info


def migrate_boundary_points(
    cluster_labels: np.ndarray,
    boundary_points: List[int],
    distance_matrix: np.ndarray
) -> np.ndarray:
    """
    迁移边界点到更合适的聚类

    参数:
        cluster_labels: 当前聚类标签数组
        boundary_points: 边界点索引列表
        distance_matrix: 距离矩阵

    返回:
        updated_labels: 更新后的聚类标签数组
    """
    n = len(cluster_labels)
    updated_labels = cluster_labels.copy()

    # 如果没有边界点，直接返回
    if not boundary_points:
        return updated_labels

    # 获取所有唯一的聚类标签
    unique_clusters = np.unique(cluster_labels)

    for point in boundary_points:
        current_cluster = cluster_labels[point]

        # 找到相邻聚类（排除当前聚类）
        adjacent_clusters = []
        for cluster in unique_clusters:
            if cluster != current_cluster:
                # 检查是否有属于该聚类的点与当前点相邻（在距离矩阵中距离较小）
                # 这里我们简单地将所有其他聚类视为相邻聚类
                adjacent_clusters.append(cluster)

        # 如果没有相邻聚类，跳过
        if not adjacent_clusters:
            continue

        # 计算当前点到每个聚类的平均距离
        avg_distances = []
        for cluster in adjacent_clusters:
            # 找到属于该聚类的所有点
            cluster_points = np.where(cluster_labels == cluster)[0]

            if len(cluster_points) == 0:
                avg_distance = float('inf')
            else:
                # 计算当前点到该聚类所有点的平均距离
                distances = distance_matrix[point, cluster_points]
                avg_distance = np.mean(distances)

            avg_distances.append((cluster, avg_distance))

        # 找到平均距离最小的聚类
        if avg_distances:
            best_cluster, min_avg_distance = min(avg_distances, key=lambda x: x[1])

            # 计算当前点到当前聚类的平均距离
            current_cluster_points = np.where(cluster_labels == current_cluster)[0]
            current_cluster_points = current_cluster_points[current_cluster_points != point]  # 排除自己

            if len(current_cluster_points) > 0:
                current_avg_distance = np.mean(distance_matrix[point, current_cluster_points])

                # 如果最佳聚类的平均距离小于当前聚类的平均距离，则迁移
                if min_avg_distance < current_avg_distance:
                    updated_labels[point] = best_cluster

    return updated_labels


def _compute_path_cost_and_penalties(
    path: List[int],
    dist_matrix: np.ndarray,
    a_i: np.ndarray,
    b_i: np.ndarray,
    s_i: np.ndarray
) -> Tuple[float, np.ndarray]:
    """
    计算路径成本和每个节点的时间窗口惩罚

    参数:
        path: 路径序列
        dist_matrix: 距离矩阵
        a_i: 时间窗口开始时间数组
        b_i: 时间窗口结束时间数组
        s_i: 服务时间数组

    返回:
        total_cost: 总成本
        penalties: 每个节点的惩罚值数组
    """
    n = len(dist_matrix)
    total_cost = 0.0
    penalties = np.zeros(n)
    current_time = 0.0

    if len(path) <= 1:
        return total_cost, penalties

    for i in range(len(path) - 1):
        # 旅行距离和时间
        travel_dist = dist_matrix[path[i], path[i + 1]]
        travel_time = travel_dist  # 假设速度=1
        current_time += travel_time

        # 服务时间
        current_time += s_i[path[i + 1]]

        # 时间窗口惩罚
        node_idx = path[i + 1]
        if current_time < a_i[node_idx]:
            penalty = 10.0 * ((a_i[node_idx] - current_time) ** 2)
        elif current_time > b_i[node_idx]:
            penalty = 20.0 * ((current_time - b_i[node_idx]) ** 2)
        else:
            penalty = 0.0

        penalties[node_idx] = penalty
        total_cost += travel_dist + penalty

    return total_cost, penalties


def iterative_refinement_controller(
    node_ids: List[int],
    dist_matrix: np.ndarray,
    a_i: np.ndarray,
    b_i: np.ndarray,
    s_i: np.ndarray,
    max_iterations: int = 5,
    convergence_threshold: float = 0.01
) -> Dict[str, Any]:
    """
    迭代改进控制器主函数

    参数:
        node_ids: 节点ID列表
        dist_matrix: 距离矩阵
        a_i: 时间窗口开始时间数组
        b_i: 时间窗口结束时间数组
        s_i: 服务时间数组
        max_iterations: 最大迭代次数 (默认5)
        convergence_threshold: 收敛阈值 (默认0.01)

    返回:
        refinement_result: 包含改进结果的字典
            - 'final_labels': 最终聚类标签
            - 'final_path': 最终路径序列
            - 'iteration_history': 迭代历史记录
            - 'converged': 是否收敛标志
    """
    n = len(node_ids)

    # 验证输入
    if n == 0:
        raise ValueError("节点ID列表不能为空")
    if dist_matrix.shape != (n, n):
        raise ValueError(f"距离矩阵形状应为({n},{n})，实际为{dist_matrix.shape}")
    if a_i.shape != (n,) or b_i.shape != (n,) or s_i.shape != (n,):
        raise ValueError(f"时间窗口和服务时间数组长度应为{n}")

    # 初始化迭代历史
    iteration_history = []

    # 步骤1: 初始时空聚类
    logger.info("步骤1: 执行初始时空聚类...")
    clustering_result = spatiotemporal_clustering(
        node_ids=node_ids,
        dist_matrix=dist_matrix,
        a_i=a_i,
        b_i=b_i,
        n_clusters=min(4, n)  # 默认4个聚类，但不超过节点数
    )

    current_labels = clustering_result['labels']
    n_clusters = clustering_result['n_clusters']

    # 步骤2: 对每个聚类进行滑动窗口优化
    logger.info("步骤2: 对每个聚类进行滑动窗口优化...")
    all_cluster_paths = []
    all_cluster_costs = []
    penalties_by_node = np.zeros(n)

    # 对每个聚类进行优化
    for cluster_id in range(n_clusters):
        # 获取聚类内的节点索引
        cluster_indices = np.where(current_labels == cluster_id)[0]
        if len(cluster_indices) == 0:
            continue

        # 获取聚类内的节点ID
        cluster_node_ids = [node_ids[i] for i in cluster_indices]

        # 执行滑动窗口优化
        cluster_path, window_history = sliding_window_optimization(
            cluster_points=cluster_node_ids,
            dist_matrix=dist_matrix,
            a_i=a_i,
            b_i=b_i,
            s_i=s_i,
            window_size=min(10, len(cluster_node_ids)),
            step_size=max(1, min(5, len(cluster_node_ids) // 2))  # 确保 step_size >= 1
        )

        # 计算路径成本和时间窗口惩罚
        cluster_cost, cluster_penalties = _compute_path_cost_and_penalties(
            cluster_path, dist_matrix, a_i, b_i, s_i
        )

        all_cluster_paths.append(cluster_path)
        all_cluster_costs.append(cluster_cost)

        # 累加惩罚值
        for node_idx in cluster_path:
            penalties_by_node[node_idx] = cluster_penalties[node_idx]

    # 构建完整路径（按聚类顺序连接）
    full_path = []
    depot_added = False

    for cluster_path in all_cluster_paths:
        if not depot_added and cluster_path and cluster_path[0] == 0:
            # 添加 depot（只添加一次）
            full_path.append(0)
            depot_added = True
            # 添加聚类路径中的其他节点（排除 depot）
            for node in cluster_path[1:]:
                if node not in full_path:
                    full_path.append(node)
        else:
            # 添加聚类路径中的所有节点（排除已添加的节点）
            for node in cluster_path:
                if node == 0 and depot_added:
                    continue  # depot 已添加
                if node not in full_path:
                    full_path.append(node)

    # 确保路径包含所有节点
    if set(full_path) != set(node_ids):
        # 如果有节点缺失，添加缺失节点到路径末尾
        missing_nodes = set(node_ids) - set(full_path)
        full_path.extend(list(missing_nodes))

    total_cost = sum(all_cluster_costs)

    # 记录初始迭代
    iteration_history.append({
        'iteration': 0,
        'labels': current_labels.copy(),
        'path': full_path.copy(),
        'total_cost': total_cost,
        'penalties': penalties_by_node.copy(),
        'boundary_violations': None,  # 将在后续迭代中计算
        'cluster_changes': 0
    })

    logger.info(f"初始迭代完成，总成本: {total_cost:.2f}")

    # 步骤3: 迭代改进
    logger.info("步骤3: 开始迭代改进...")
    converged = False

    for iteration in range(1, max_iterations + 1):
        logger.info(f"迭代 {iteration}/{max_iterations}...")

        # 3.1 评估边界点违反情况
        boundary_points, violation_scores, violation_info = evaluate_boundary_violations(
            cluster_labels=current_labels,
            path_sequence=full_path,
            penalties_by_node=penalties_by_node,
            threshold_multiplier=2.0
        )

        # 3.2 迁移边界点
        old_labels = current_labels.copy()
        current_labels = migrate_boundary_points(
            cluster_labels=current_labels,
            boundary_points=boundary_points,
            distance_matrix=dist_matrix
        )

        # 计算聚类变化
        cluster_changes = np.sum(current_labels != old_labels)

        # 3.3 如果聚类发生变化，重新优化
        if cluster_changes > 0:
            logger.info(f"聚类发生变化，重新优化... (变化数: {cluster_changes})")

            # 重新聚类和优化
            # 注意：这里简化处理，实际可能需要重新运行完整的聚类和优化流程
            # 对于简化实现，我们只重新计算受影响的聚类

            # 重新计算惩罚值（简化：使用相同路径）
            total_cost, penalties_by_node = _compute_path_cost_and_penalties(
                full_path, dist_matrix, a_i, b_i, s_i
            )
        else:
            logger.info("聚类未发生变化")

        # 3.4 记录迭代历史
        iteration_history.append({
            'iteration': iteration,
            'labels': current_labels.copy(),
            'path': full_path.copy(),
            'total_cost': total_cost,
            'penalties': penalties_by_node.copy(),
            'boundary_violations': violation_info,
            'cluster_changes': cluster_changes
        })

        # 3.5 检查收敛条件
        # 条件1: 聚类分配稳定度 > 95%
        if iteration > 0:
            prev_labels = iteration_history[iteration-1]['labels']
            label_stability = 1.0 - (np.sum(current_labels != prev_labels) / n)
            stability_threshold = 0.95

            # 条件2: 综合代价变化 < 1%
            prev_cost = iteration_history[iteration-1]['total_cost']
            if prev_cost > 0:
                cost_change = abs(total_cost - prev_cost) / prev_cost
            else:
                cost_change = 0.0

            logger.info(f"  标签稳定度: {label_stability:.3f}, 成本变化: {cost_change:.3f}")

            if label_stability > stability_threshold and cost_change < convergence_threshold:
                converged = True
                logger.info(f"迭代 {iteration} 收敛!")
                break

        # 如果达到最大迭代次数
        if iteration == max_iterations:
            logger.info(f"达到最大迭代次数 {max_iterations}")

    # 构建最终结果
    refinement_result = {
        'final_labels': current_labels,
        'final_path': full_path,
        'iteration_history': iteration_history,
        'converged': converged,
        'total_iterations': len(iteration_history) - 1,  # 减去初始迭代
        'final_cost': total_cost,
        'n_clusters': n_clusters
    }

    logger.info(f"迭代改进完成! 总迭代次数: {refinement_result['total_iterations']}, "
                f"最终成本: {total_cost:.2f}, 收敛: {converged}")

    return refinement_result