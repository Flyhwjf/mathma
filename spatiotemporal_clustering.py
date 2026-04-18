"""
时空聚类算法模块
用于将大规模节点按时空特征进行聚类，为滑动窗口优化做准备
"""

import numpy as np
import logging
from typing import List, Tuple, Dict, Any, TypedDict

# 设置日志
logger = logging.getLogger(__name__)


class ClusteringResult(TypedDict):
    """聚类结果类型定义"""
    labels: np.ndarray
    centroids: np.ndarray
    clusters: List[List[int]]
    spatiotemporal_dist: np.ndarray
    cluster_stats: List[Dict[str, Any]]
    history: List[Dict[str, Any]]
    n_clusters: int
    n_nodes: int


def compute_spatiotemporal_distance_matrix(
    dist_matrix: np.ndarray,
    a_i: np.ndarray,
    b_i: np.ndarray,
    spatial_weight: float = 0.6,
    temporal_weight: float = 0.4
) -> np.ndarray:
    """
    计算时空综合距离矩阵

    参数:
        dist_matrix: 空间距离矩阵 (n x n)
        a_i: 时间窗口开始时间数组 (n,)
        b_i: 时间窗口结束时间数组 (n,)
        spatial_weight: 空间距离权重 (默认0.6)
        temporal_weight: 时间距离权重 (默认0.4)

    返回:
        spatiotemporal_dist: 时空综合距离矩阵 (n x n)
    """
    n = dist_matrix.shape[0]

    # 验证输入
    if dist_matrix.shape != (n, n):
        raise ValueError(f"距离矩阵应为方阵，实际形状为{dist_matrix.shape}")
    if a_i.shape != (n,) or b_i.shape != (n,):
        raise ValueError(f"时间窗口数组长度应为{n}，实际为a_i:{a_i.shape}, b_i:{b_i.shape}")
    if abs(spatial_weight + temporal_weight - 1.0) > 1e-10:
        raise ValueError(f"权重之和应为1.0，实际为{spatial_weight + temporal_weight}")

    # 1. 计算归一化空间距离
    spatial_max = dist_matrix.max()
    if spatial_max > 0:
        normalized_spatial = dist_matrix / spatial_max
    else:
        normalized_spatial = dist_matrix.copy()

    # 2. 计算时间距离
    # 时间距离 = |a_i - a_j| + |b_i - b_j|
    # 使用NumPy广播进行向量化计算
    a_i_expanded = a_i[:, np.newaxis]  # Shape (n, 1)
    b_i_expanded = b_i[:, np.newaxis]  # Shape (n, 1)
    temporal_dist = np.abs(a_i_expanded - a_i) + np.abs(b_i_expanded - b_i)

    # 归一化时间距离
    temporal_max = temporal_dist.max()
    if temporal_max > 0:
        normalized_temporal = temporal_dist / temporal_max
    else:
        normalized_temporal = temporal_dist.copy()

    # 3. 计算综合距离
    spatiotemporal_dist = (
        spatial_weight * normalized_spatial +
        temporal_weight * normalized_temporal
    )

    # 确保对角线为0
    np.fill_diagonal(spatiotemporal_dist, 0.0)

    return spatiotemporal_dist


def improved_kmeans_clustering(
    distance_matrix: np.ndarray,
    n_clusters: int = 4,
    max_iterations: int = 50,
    tolerance: float = 0.05
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    """
    改进的K-means聚类算法（基于距离矩阵）

    参数:
        distance_matrix: 距离矩阵 (n x n)
        n_clusters: 聚类数量 (默认4)
        max_iterations: 最大迭代次数 (默认50)
        tolerance: 收敛容忍度 (默认0.05)

    返回:
        cluster_labels: 聚类标签数组 (n,)
        centroids: 聚类中心索引数组 (n_clusters,)
        history: 迭代历史记录列表
    """
    n = distance_matrix.shape[0]

    # 验证输入
    if distance_matrix.shape != (n, n):
        raise ValueError(f"距离矩阵应为方阵，实际形状为{distance_matrix.shape}")
    if n_clusters <= 0 or n_clusters > n:
        raise ValueError(f"聚类数量应在[1, {n}]范围内，实际为{n_clusters}")
    if max_iterations <= 0:
        raise ValueError(f"最大迭代次数应大于0，实际为{max_iterations}")

    # 初始化历史记录
    history = []

    # 1. 初始化聚类中心（最大最小距离法）
    centroids = np.zeros(n_clusters, dtype=int)

    # 选择第一个中心：随机选择或选择距离总和最大的点
    # 这里选择距离总和最大的点作为第一个中心
    distance_sums = distance_matrix.sum(axis=1)
    centroids[0] = np.argmax(distance_sums)

    # 选择后续中心：选择距离已有中心最远的点
    for k in range(1, n_clusters):
        # 计算每个点到已有中心的最小距离
        min_distances = np.zeros(n)
        for i in range(n):
            min_dist_to_centers = np.min(distance_matrix[i, centroids[:k]])
            min_distances[i] = min_dist_to_centers

        # 选择最小距离最大的点作为新中心
        centroids[k] = np.argmax(min_distances)

    # 2. 迭代聚类
    cluster_labels = np.zeros(n, dtype=int)
    prev_labels = None

    for iteration in range(max_iterations):
        # 分配点到最近的聚类中心
        for i in range(n):
            distances_to_centers = distance_matrix[i, centroids]
            cluster_labels[i] = np.argmin(distances_to_centers)

        # 检查收敛
        if prev_labels is not None and np.all(cluster_labels == prev_labels):
            history.append({
                'iteration': iteration,
                'centroids': centroids.copy(),
                'labels': cluster_labels.copy(),
                'converged': True
            })
            break

        # 更新聚类中心
        new_centroids = centroids.copy()
        for k in range(n_clusters):
            cluster_points = np.where(cluster_labels == k)[0]

            if len(cluster_points) == 0:
                # 空聚类处理：选择距离其他中心最远的点作为新中心
                distances_to_other_centers = np.zeros(n)
                # 获取所有不是当前中心的中心点
                other_centroids_mask = np.ones(n_clusters, dtype=bool)
                other_centroids_mask[k] = False
                other_centroids = centroids[other_centroids_mask]

                if len(other_centroids) > 0:
                    # 创建非中心点的掩码
                    non_centroid_mask = np.ones(n, dtype=bool)
                    non_centroid_mask[centroids] = False
                    # 向量化计算每个非中心点到其他中心的最小距离
                    if np.any(non_centroid_mask):
                        # 获取所有非中心点的距离子矩阵
                        non_centroid_distances = distance_matrix[non_centroid_mask][:, other_centroids]
                        min_distances = np.min(non_centroid_distances, axis=1)
                        # 将结果放回正确位置
                        non_centroid_indices = np.where(non_centroid_mask)[0]
                        distances_to_other_centers[non_centroid_indices] = min_distances

                # 选择距离其他中心最远的点
                if np.any(distances_to_other_centers > 0):
                    new_center = np.argmax(distances_to_other_centers)
                    new_centroids[k] = new_center
                continue

            # 计算每个点到同簇其他点的平均距离
            avg_distances = np.zeros(len(cluster_points))
            for idx, point in enumerate(cluster_points):
                # 计算点到同簇其他点的平均距离
                same_cluster_points = cluster_points[cluster_points != point]
                if len(same_cluster_points) > 0:
                    avg_distances[idx] = np.mean(distance_matrix[point, same_cluster_points])
                else:
                    avg_distances[idx] = 0.0

            # 选择平均距离最小的点作为新中心
            if len(avg_distances) > 0:
                min_avg_idx = np.argmin(avg_distances)
                new_centroids[k] = cluster_points[min_avg_idx]

        # 记录迭代历史
        history.append({
            'iteration': iteration,
            'centroids': centroids.copy(),
            'labels': cluster_labels.copy(),
            'converged': False
        })

        # 检查中心点是否变化
        if np.array_equal(centroids, new_centroids):
            # 中心点未变化，检查标签变化是否小于容忍度
            if prev_labels is not None:
                label_change_rate = np.mean(cluster_labels != prev_labels)
                if label_change_rate < tolerance:
                    history[-1]['converged'] = True
                    break

        centroids = new_centroids.copy()
        prev_labels = cluster_labels.copy()

    # 如果达到最大迭代次数，标记最后一次迭代为收敛
    if len(history) > 0:
        history[-1]['converged'] = True

    return cluster_labels, centroids, history


def spatiotemporal_clustering(
    node_ids: List[int],
    dist_matrix: np.ndarray,
    a_i: np.ndarray,
    b_i: np.ndarray,
    n_clusters: int = 4
) -> ClusteringResult:
    """
    时空聚类主函数

    参数:
        node_ids: 节点ID列表
        dist_matrix: 空间距离矩阵
        a_i: 时间窗口开始时间数组
        b_i: 时间窗口结束时间数组
        n_clusters: 聚类数量 (默认4)

    返回:
        clustering_result: 包含聚类结果的字典
            - 'labels': 聚类标签数组
            - 'centroids': 聚类中心索引数组
            - 'clusters': 按聚类分组的节点ID列表
            - 'spatiotemporal_dist': 时空距离矩阵
    """
    n = len(node_ids)

    # 验证输入
    if n == 0:
        raise ValueError("节点ID列表不能为空")
    if dist_matrix.shape != (n, n):
        raise ValueError(f"距离矩阵形状应为({n},{n})，实际为{dist_matrix.shape}")
    if a_i.shape != (n,) or b_i.shape != (n,):
        raise ValueError(f"时间窗口数组长度应为{n}，实际为a_i:{a_i.shape}, b_i:{b_i.shape}")
    if n_clusters <= 0 or n_clusters > n:
        raise ValueError(f"聚类数量应在[1, {n}]范围内，实际为{n_clusters}")

    # 1. 计算时空距离矩阵
    logger.info(f"计算时空距离矩阵 (n={n})...")
    spatiotemporal_dist = compute_spatiotemporal_distance_matrix(
        dist_matrix, a_i, b_i, spatial_weight=0.6, temporal_weight=0.4
    )

    # 2. 执行改进的K-means聚类
    logger.info(f"执行改进的K-means聚类 (k={n_clusters})...")
    cluster_labels, centroids, history = improved_kmeans_clustering(
        spatiotemporal_dist, n_clusters=n_clusters, max_iterations=50
    )

    # 3. 组织聚类结果
    clusters = []
    for k in range(n_clusters):
        cluster_indices = np.where(cluster_labels == k)[0]
        cluster_node_ids = [node_ids[i] for i in cluster_indices]
        clusters.append(cluster_node_ids)

    # 4. 计算聚类统计信息
    cluster_stats = []
    for k in range(n_clusters):
        cluster_indices = np.where(cluster_labels == k)[0]
        if len(cluster_indices) == 0:
            cluster_stats.append({
                'cluster_id': k,
                'size': 0,
                'center_node_id': None,
                'avg_intra_cluster_distance': 0.0
            })
            continue

        # 计算簇内平均距离 - 使用向量化计算
        # 获取簇内点的距离子矩阵
        cluster_dist_submatrix = spatiotemporal_dist[np.ix_(cluster_indices, cluster_indices)]
        # 获取上三角部分（不包括对角线）
        upper_tri_indices = np.triu_indices(len(cluster_indices), k=1)
        intra_distances = cluster_dist_submatrix[upper_tri_indices]

        avg_intra_distance = np.mean(intra_distances) if len(intra_distances) > 0 else 0.0

        # 获取中心点对应的节点ID
        center_idx = centroids[k]
        center_node_id = node_ids[center_idx] if center_idx < n else None

        cluster_stats.append({
            'cluster_id': k,
            'size': len(cluster_indices),
            'center_node_id': center_node_id,
            'avg_intra_cluster_distance': avg_intra_distance
        })

    # 5. 构建结果字典
    result = {
        'labels': cluster_labels,
        'centroids': centroids,
        'clusters': clusters,
        'spatiotemporal_dist': spatiotemporal_dist,
        'cluster_stats': cluster_stats,
        'history': history,
        'n_clusters': n_clusters,
        'n_nodes': n
    }

    logger.info(f"时空聚类完成！共{n}个节点分为{n_clusters}个聚类。")
    for stat in cluster_stats:
        logger.info(f"  聚类{stat['cluster_id']}: {stat['size']}个节点，"
                    f"中心节点{stat['center_node_id']}，"
                    f"平均簇内距离{stat['avg_intra_cluster_distance']:.4f}")

    return result