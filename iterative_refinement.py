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
    dist_matrix: np.ndarray,
    threshold_multiplier: float = 2.0,
    crossing_threshold: float = 0.2,
    cost_imbalance_threshold: float = 1.5
) -> Tuple[List[int], List[float], Dict[str, Any]]:
    """
    评估边界点违反情况和触发重新聚类的条件

    参数:
        cluster_labels: 聚类标签数组
        path_sequence: 当前路径序列
        penalties_by_node: 每个节点的时间窗口惩罚值数组
        dist_matrix: 距离矩阵
        threshold_multiplier: 边界点判定阈值乘数 (默认2.0)
        crossing_threshold: 路径交叉阈值 (默认0.2，即20%)
        cost_imbalance_threshold: 成本不平衡阈值 (默认1.5)

    返回:
        boundary_points: 边界点索引列表
        violation_scores: 违反程度分数列表
        violation_info: 违反信息字典（包含所有触发条件）
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

    # 4. 检查路径交叉触发条件
    has_crossing, crossing_info = _detect_path_crossing(
        path_sequence, cluster_labels, dist_matrix, crossing_threshold
    )

    # 5. 检查聚类成本不平衡触发条件
    cluster_costs, cost_info = _compute_cluster_costs(
        path_sequence, cluster_labels, dist_matrix, penalties_by_node
    )

    cost_imbalance_triggered = False
    if cost_info['average_cost'] > 0:
        cost_imbalance_triggered = cost_info['cost_imbalance_ratio'] > cost_imbalance_threshold

    # 6. 构建完整的违反信息字典
    violation_info = {
        # 边界点惩罚信息
        'average_penalty': average_penalty,
        'threshold': threshold,
        'violation_count': violation_count,
        'violation_rate': violation_count / len(boundary_points) if boundary_points else 0.0,
        'boundary_points_count': len(boundary_points),
        'total_points': n,

        # 路径交叉信息
        'has_path_crossing': has_crossing,
        'path_crossing_info': crossing_info,

        # 聚类成本信息
        'cluster_costs': cluster_costs,
        'average_cluster_cost': cost_info['average_cost'],
        'max_cluster_cost': cost_info['max_cost'],
        'min_cluster_cost': cost_info['min_cost'],
        'cost_imbalance_ratio': cost_info['cost_imbalance_ratio'],
        'cost_imbalance_triggered': cost_imbalance_triggered,

        # 触发条件汇总
        'triggers': {
            'boundary_penalty_trigger': violation_count > 0,
            'path_crossing_trigger': has_crossing,
            'cost_imbalance_trigger': cost_imbalance_triggered,
            'any_trigger': violation_count > 0 or has_crossing or cost_imbalance_triggered
        }
    }

    return boundary_points, violation_scores, violation_info


def migrate_boundary_points(
    cluster_labels: np.ndarray,
    boundary_points: List[int],
    distance_matrix: np.ndarray,
    path_sequence: List[int] = None,
    adjacency_threshold: float = 0.3
) -> np.ndarray:
    """
    迁移边界点到更合适的聚类

    参数:
        cluster_labels: 当前聚类标签数组
        boundary_points: 边界点索引列表
        distance_matrix: 距离矩阵
        path_sequence: 路径序列（用于检测相邻聚类）
        adjacency_threshold: 相邻性阈值（默认0.3，即30%）

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

        # 找到相邻聚类（改进版）
        adjacent_clusters = _find_adjacent_clusters(
            point, current_cluster, cluster_labels, distance_matrix,
            path_sequence, adjacency_threshold
        )

        # 如果没有相邻聚类，跳过
        if not adjacent_clusters:
            continue

        # 计算当前点到每个相邻聚类的平均距离
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


def _find_adjacent_clusters(
    point: int,
    current_cluster: int,
    cluster_labels: np.ndarray,
    distance_matrix: np.ndarray,
    path_sequence: List[int] = None,
    adjacency_threshold: float = 0.3
) -> List[int]:
    """
    找到与给定点相邻的聚类

    参数:
        point: 当前点索引
        current_cluster: 当前聚类ID
        cluster_labels: 聚类标签数组
        distance_matrix: 距离矩阵
        path_sequence: 路径序列
        adjacency_threshold: 相邻性阈值

    返回:
        adjacent_clusters: 相邻聚类ID列表
    """
    unique_clusters = np.unique(cluster_labels)
    adjacent_clusters = []

    # 方法1: 基于路径的相邻性检测
    if path_sequence is not None:
        # 找到点在路径中的位置
        if point in path_sequence:
            point_idx = path_sequence.index(point)

            # 检查路径中相邻的聚类
            for offset in [-2, -1, 1, 2]:  # 检查前后2个位置
                check_idx = point_idx + offset
                if 0 <= check_idx < len(path_sequence):
                    neighbor_point = path_sequence[check_idx]
                    neighbor_cluster = cluster_labels[neighbor_point]
                    if neighbor_cluster != current_cluster and neighbor_cluster not in adjacent_clusters:
                        adjacent_clusters.append(neighbor_cluster)

    # 方法2: 基于距离的相邻性检测
    if not adjacent_clusters:
        # 计算当前点到所有其他点的平均距离
        all_distances = distance_matrix[point]
        avg_distance = np.mean(all_distances[all_distances > 0]) if np.any(all_distances > 0) else 0

        if avg_distance > 0:
            for cluster in unique_clusters:
                if cluster == current_cluster:
                    continue

                # 找到属于该聚类的所有点
                cluster_points = np.where(cluster_labels == cluster)[0]
                if len(cluster_points) == 0:
                    continue

                # 计算当前点到该聚类点的平均距离
                cluster_distances = distance_matrix[point, cluster_points]
                avg_cluster_distance = np.mean(cluster_distances)

                # 如果平均距离小于阈值，则认为是相邻聚类
                if avg_cluster_distance < avg_distance * adjacency_threshold:
                    adjacent_clusters.append(cluster)

    # 如果以上方法都没有找到相邻聚类，返回所有其他聚类（保持向后兼容）
    if not adjacent_clusters:
        adjacent_clusters = [c for c in unique_clusters if c != current_cluster]

    return adjacent_clusters


def _detect_path_crossing(
    path: List[int],
    cluster_labels: np.ndarray,
    dist_matrix: np.ndarray,
    crossing_threshold: float = 0.2
) -> Tuple[bool, Dict[str, Any]]:
    """
    检测路径交叉情况

    参数:
        path: 路径序列
        cluster_labels: 聚类标签数组
        dist_matrix: 距离矩阵
        crossing_threshold: 交叉阈值 (默认0.2，即20%)

    返回:
        has_crossing: 是否存在显著交叉
        crossing_info: 交叉信息字典
    """
    n = len(path)
    crossing_info = {
        'has_crossing': False,
        'crossing_pairs': [],
        'crossing_ratios': [],
        'max_crossing_ratio': 0.0,
        'total_crossing_segments': 0
    }

    if n < 4:  # 至少需要4个点才能有交叉
        return False, crossing_info

    # 识别聚类边界
    cluster_boundaries = []
    for i in range(n-1):
        if cluster_labels[path[i]] != cluster_labels[path[i+1]]:
            cluster_boundaries.append(i)  # 边界在i和i+1之间

    if len(cluster_boundaries) < 2:  # 至少需要2个边界才能有交叉
        return False, crossing_info

    # 检查相邻聚类之间的路径段
    crossing_pairs = []
    crossing_ratios = []

    for i in range(len(cluster_boundaries)-1):
        # 获取两个相邻边界
        boundary1 = cluster_boundaries[i]
        boundary2 = cluster_boundaries[i+1]

        # 获取聚类ID
        cluster1 = cluster_labels[path[boundary1]]
        cluster2 = cluster_labels[path[boundary2]]

        if cluster1 == cluster2:
            continue  # 同一聚类，跳过

        # 计算路径段长度（从boundary1到boundary2+1）
        path_segment_length = 0.0
        for j in range(boundary1, boundary2):
            path_segment_length += dist_matrix[path[j], path[j+1]]

        # 计算聚类中心之间的直接距离
        # 找到聚类1和聚类2的中心点（使用路径中的第一个点作为代表）
        cluster1_points = [path[j] for j in range(n) if cluster_labels[path[j]] == cluster1]
        cluster2_points = [path[j] for j in range(n) if cluster_labels[path[j]] == cluster2]

        if not cluster1_points or not cluster2_points:
            continue

        # 使用聚类中第一个点的距离作为代表
        direct_distance = dist_matrix[cluster1_points[0], cluster2_points[0]]

        if direct_distance > 0:
            crossing_ratio = path_segment_length / direct_distance
            if crossing_ratio > (1.0 + crossing_threshold):  # 路径长度 > 直接距离 * (1 + 阈值)
                crossing_pairs.append((cluster1, cluster2))
                crossing_ratios.append(crossing_ratio)

    if crossing_pairs:
        crossing_info.update({
            'has_crossing': True,
            'crossing_pairs': crossing_pairs,
            'crossing_ratios': crossing_ratios,
            'max_crossing_ratio': max(crossing_ratios) if crossing_ratios else 0.0,
            'total_crossing_segments': len(crossing_pairs)
        })
        return True, crossing_info

    return False, crossing_info


def _compute_cluster_costs(
    path: List[int],
    cluster_labels: np.ndarray,
    dist_matrix: np.ndarray,
    penalties_by_node: np.ndarray
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    计算每个聚类的综合成本

    参数:
        path: 路径序列
        cluster_labels: 聚类标签数组
        dist_matrix: 距离矩阵
        penalties_by_node: 每个节点的惩罚值数组

    返回:
        cluster_costs: 每个聚类的成本数组
        cost_info: 成本信息字典
    """
    unique_clusters = np.unique(cluster_labels)
    n_clusters = len(unique_clusters)
    cluster_costs = np.zeros(n_clusters)

    # 计算每个聚类的成本
    for i, cluster_id in enumerate(unique_clusters):
        # 找到属于该聚类的所有节点
        cluster_nodes = [node for node in path if cluster_labels[node] == cluster_id]

        if not cluster_nodes:
            continue

        # 计算聚类内的旅行距离
        cluster_distance = 0.0
        for j in range(len(cluster_nodes)-1):
            node1 = cluster_nodes[j]
            # 找到node1在完整路径中的位置
            pos1 = path.index(node1)
            # 找到下一个同聚类节点在路径中的位置
            next_cluster_node = None
            for k in range(pos1+1, len(path)):
                if cluster_labels[path[k]] == cluster_id:
                    next_cluster_node = path[k]
                    break

            if next_cluster_node:
                cluster_distance += dist_matrix[node1, next_cluster_node]

        # 计算聚类内的总惩罚
        cluster_penalty = sum(penalties_by_node[node] for node in cluster_nodes)

        # 综合成本 = 距离 + 惩罚
        cluster_costs[i] = cluster_distance + cluster_penalty

    # 计算成本统计信息
    avg_cost = np.mean(cluster_costs) if n_clusters > 0 else 0.0
    max_cost = np.max(cluster_costs) if n_clusters > 0 else 0.0
    min_cost = np.min(cluster_costs) if n_clusters > 0 else 0.0

    cost_info = {
        'cluster_costs': cluster_costs,
        'average_cost': avg_cost,
        'max_cost': max_cost,
        'min_cost': min_cost,
        'cost_imbalance_ratio': (max_cost / avg_cost) if avg_cost > 0 else 0.0,
        'n_clusters': n_clusters
    }

    return cluster_costs, cost_info


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

        # 3.1 评估边界点违反情况和触发条件
        boundary_points, violation_scores, violation_info = evaluate_boundary_violations(
            cluster_labels=current_labels,
            path_sequence=full_path,
            penalties_by_node=penalties_by_node,
            dist_matrix=dist_matrix,
            threshold_multiplier=2.0,
            crossing_threshold=0.2,
            cost_imbalance_threshold=1.5
        )

        # 3.2 迁移边界点
        old_labels = current_labels.copy()
        current_labels = migrate_boundary_points(
            cluster_labels=current_labels,
            boundary_points=boundary_points,
            distance_matrix=dist_matrix,
            path_sequence=full_path,
            adjacency_threshold=0.3
        )

        # 计算聚类变化
        cluster_changes = np.sum(current_labels != old_labels)

        # 3.3 检查是否需要重新优化（基于触发条件或聚类变化）
        need_reoptimization = (
            cluster_changes > 0 or  # 聚类发生变化
            violation_info['triggers']['any_trigger']  # 任何触发条件满足
        )

        if need_reoptimization:
            trigger_reasons = []
            if cluster_changes > 0:
                trigger_reasons.append(f"聚类变化({cluster_changes})")
            if violation_info['triggers']['boundary_penalty_trigger']:
                trigger_reasons.append("边界点惩罚触发")
            if violation_info['triggers']['path_crossing_trigger']:
                trigger_reasons.append("路径交叉触发")
            if violation_info['triggers']['cost_imbalance_trigger']:
                trigger_reasons.append("成本不平衡触发")

            logger.info(f"触发重新优化: {', '.join(trigger_reasons)}")

            # 重新聚类和优化受影响的部分
            if cluster_changes > 0 or violation_info['triggers']['any_trigger']:
                # 如果触发条件强烈，重新运行完整的聚类和优化
                if (violation_info['triggers']['path_crossing_trigger'] or
                    violation_info['triggers']['cost_imbalance_trigger'] or
                    cluster_changes > n * 0.1):  # 超过10%的节点发生变化

                    logger.info("执行完整重新聚类和优化...")

                    # 重新聚类
                    clustering_result = spatiotemporal_clustering(
                        node_ids=node_ids,
                        dist_matrix=dist_matrix,
                        a_i=a_i,
                        b_i=b_i,
                        n_clusters=min(4, n)
                    )
                    current_labels = clustering_result['labels']
                    n_clusters = clustering_result['n_clusters']

                    # 重新优化所有聚类
                    all_cluster_paths = []
                    all_cluster_costs = []
                    penalties_by_node = np.zeros(n)

                    for cluster_id in range(n_clusters):
                        cluster_indices = np.where(current_labels == cluster_id)[0]
                        if len(cluster_indices) == 0:
                            continue

                        cluster_node_ids = [node_ids[i] for i in cluster_indices]
                        cluster_path, _ = sliding_window_optimization(
                            cluster_points=cluster_node_ids,
                            dist_matrix=dist_matrix,
                            a_i=a_i,
                            b_i=b_i,
                            s_i=s_i,
                            window_size=min(10, len(cluster_node_ids)),
                            step_size=max(1, min(5, len(cluster_node_ids) // 2))
                        )

                        cluster_cost, cluster_penalties = _compute_path_cost_and_penalties(
                            cluster_path, dist_matrix, a_i, b_i, s_i
                        )

                        all_cluster_paths.append(cluster_path)
                        all_cluster_costs.append(cluster_cost)

                        for node_idx in cluster_path:
                            penalties_by_node[node_idx] = cluster_penalties[node_idx]

                    # 重新构建完整路径
                    full_path = []
                    depot_added = False

                    for cluster_path in all_cluster_paths:
                        if not depot_added and cluster_path and cluster_path[0] == 0:
                            full_path.append(0)
                            depot_added = True
                            for node in cluster_path[1:]:
                                if node not in full_path:
                                    full_path.append(node)
                        else:
                            for node in cluster_path:
                                if node == 0 and depot_added:
                                    continue
                                if node not in full_path:
                                    full_path.append(node)

                    if set(full_path) != set(node_ids):
                        missing_nodes = set(node_ids) - set(full_path)
                        full_path.extend(list(missing_nodes))

                    total_cost = sum(all_cluster_costs)
                else:
                    # 只重新计算惩罚值（简化处理）
                    logger.info("重新计算惩罚值...")
                    total_cost, penalties_by_node = _compute_path_cost_and_penalties(
                        full_path, dist_matrix, a_i, b_i, s_i
                    )
        else:
            logger.info("无需重新优化")

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