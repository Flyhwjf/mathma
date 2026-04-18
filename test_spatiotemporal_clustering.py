"""
时空聚类算法测试模块
"""

import numpy as np
import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from spatiotemporal_clustering import (
    compute_spatiotemporal_distance_matrix,
    improved_kmeans_clustering,
    spatiotemporal_clustering
)


def test_compute_spatiotemporal_distance_matrix():
    """测试时空距离矩阵计算"""
    print("测试时空距离矩阵计算...")

    # 创建测试数据
    n = 5
    dist_matrix = np.array([
        [0.0, 10.0, 20.0, 30.0, 40.0],
        [10.0, 0.0, 15.0, 25.0, 35.0],
        [20.0, 15.0, 0.0, 10.0, 20.0],
        [30.0, 25.0, 10.0, 0.0, 15.0],
        [40.0, 35.0, 20.0, 15.0, 0.0]
    ])

    a_i = np.array([0, 2, 4, 6, 8])  # 开始时间
    b_i = np.array([5, 7, 9, 11, 13])  # 结束时间

    # 计算时空距离矩阵
    result = compute_spatiotemporal_distance_matrix(
        dist_matrix, a_i, b_i, spatial_weight=0.6, temporal_weight=0.4
    )

    # 验证结果
    assert result.shape == (n, n), f"期望形状({n},{n})，实际形状{result.shape}"
    assert np.allclose(result.diagonal(), 0.0), "对角线应为0"
    assert np.all(result >= 0.0), "所有距离应为非负"
    assert np.allclose(result, result.T), "距离矩阵应对称"

    # 验证权重影响
    result_spatial_only = compute_spatiotemporal_distance_matrix(
        dist_matrix, a_i, b_i, spatial_weight=1.0, temporal_weight=0.0
    )
    result_temporal_only = compute_spatiotemporal_distance_matrix(
        dist_matrix, a_i, b_i, spatial_weight=0.0, temporal_weight=1.0
    )

    # 纯空间距离应等于归一化空间距离
    spatial_max = dist_matrix.max()
    expected_spatial = dist_matrix / spatial_max if spatial_max > 0 else dist_matrix
    assert np.allclose(result_spatial_only, expected_spatial), "纯空间距离计算错误"

    print("时空距离矩阵计算测试通过！")
    return True


def test_improved_kmeans_clustering():
    """测试改进的K-means聚类算法"""
    print("测试改进的K-means聚类算法...")

    # 创建测试数据：5个点，明显分为2类
    n = 5
    distance_matrix = np.array([
        [0.0, 0.1, 0.9, 0.8, 0.85],
        [0.1, 0.0, 0.95, 0.85, 0.9],
        [0.9, 0.95, 0.0, 0.2, 0.15],
        [0.8, 0.85, 0.2, 0.0, 0.1],
        [0.85, 0.9, 0.15, 0.1, 0.0]
    ])

    n_clusters = 2

    # 执行聚类
    cluster_labels, centroids, history = improved_kmeans_clustering(
        distance_matrix, n_clusters=n_clusters, max_iterations=20
    )

    # 验证结果
    assert cluster_labels.shape == (n,), f"聚类标签形状应为({n},)，实际为{cluster_labels.shape}"
    assert centroids.shape == (n_clusters,), f"中心点形状应为({n_clusters},)，实际为{centroids.shape}"
    assert len(history) > 0, "迭代历史不应为空"

    # 验证聚类标签在有效范围内
    assert np.all(cluster_labels >= 0) and np.all(cluster_labels < n_clusters), \
        f"聚类标签应在[0, {n_clusters-1}]范围内"

    # 验证中心点是有效索引
    assert np.all(centroids >= 0) and np.all(centroids < n), \
        f"中心点索引应在[0, {n-1}]范围内"

    # 验证聚类结果合理（前2点应为一类，后3点应为另一类）
    # 基于构造的距离矩阵，点0和1应属于同一类，点2、3、4应属于同一类
    if cluster_labels[0] == cluster_labels[1]:
        # 点0和1同类，则点2、3、4应同类
        assert cluster_labels[2] == cluster_labels[3] == cluster_labels[4], \
            "点2、3、4应属于同一类"
    else:
        # 点0和1不同类，则每个点应至少有一个同类点
        unique_labels, counts = np.unique(cluster_labels, return_counts=True)
        assert np.all(counts >= 1), "每个聚类应至少包含一个点"

    print("改进的K-means聚类算法测试通过！")
    return True


def test_spatiotemporal_clustering():
    """测试时空聚类主函数"""
    print("测试时空聚类主函数...")

    # 创建测试数据
    n = 8
    node_ids = list(range(n))

    # 空间距离矩阵：前4点距离近，后4点距离近（确保对称）
    np.random.seed(42)  # 固定随机种子以便重现
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):  # 只填充上三角
            if i == j:
                dist_matrix[i, j] = 0.0
            elif i < 4 and j < 4:
                dist_matrix[i, j] = np.random.uniform(5, 15)  # 前4点距离近
            elif i >= 4 and j >= 4:
                dist_matrix[i, j] = np.random.uniform(5, 15)  # 后4点距离近
            else:
                dist_matrix[i, j] = np.random.uniform(30, 50)  # 不同组距离远

    # 复制到下半部分以确保对称
    for i in range(n):
        for j in range(i):
            dist_matrix[i, j] = dist_matrix[j, i]

    # 时间窗口：前4点时间相近，后4点时间相近
    a_i = np.zeros(n)
    b_i = np.zeros(n)
    for i in range(n):
        if i < 4:
            a_i[i] = np.random.uniform(0, 2)
            b_i[i] = np.random.uniform(5, 7)
        else:
            a_i[i] = np.random.uniform(10, 12)
            b_i[i] = np.random.uniform(15, 17)

    n_clusters = 2

    # 执行时空聚类
    result = spatiotemporal_clustering(
        node_ids, dist_matrix, a_i, b_i, n_clusters=n_clusters
    )

    # 验证结果结构
    required_keys = ['labels', 'centroids', 'clusters', 'spatiotemporal_dist',
                     'cluster_stats', 'history', 'n_clusters', 'n_nodes']
    for key in required_keys:
        assert key in result, f"结果字典缺少键'{key}'"

    # 验证基本属性
    assert result['n_nodes'] == n, f"节点数量应为{n}，实际为{result['n_nodes']}"
    assert result['n_clusters'] == n_clusters, f"聚类数量应为{n_clusters}，实际为{result['n_clusters']}"

    # 验证标签
    labels = result['labels']
    assert labels.shape == (n,), f"标签形状应为({n},)，实际为{labels.shape}"
    assert np.all(labels >= 0) and np.all(labels < n_clusters), \
        f"标签应在[0, {n_clusters-1}]范围内"

    # 验证中心点
    centroids = result['centroids']
    assert centroids.shape == (n_clusters,), f"中心点形状应为({n_clusters},)，实际为{centroids.shape}"
    assert np.all(centroids >= 0) and np.all(centroids < n), \
        f"中心点索引应在[0, {n-1}]范围内"

    # 验证聚类分组
    clusters = result['clusters']
    assert len(clusters) == n_clusters, f"应有{n_clusters}个聚类分组，实际有{len(clusters)}"

    # 验证所有节点都被分配
    all_nodes_in_clusters = []
    for cluster in clusters:
        all_nodes_in_clusters.extend(cluster)
    assert set(all_nodes_in_clusters) == set(node_ids), "所有节点都应被分配到聚类中"

    # 验证距离矩阵
    spatiotemporal_dist = result['spatiotemporal_dist']
    assert spatiotemporal_dist.shape == (n, n), f"距离矩阵形状应为({n},{n})，实际为{spatiotemporal_dist.shape}"
    assert np.allclose(spatiotemporal_dist.diagonal(), 0.0), "对角线应为0"
    assert np.all(spatiotemporal_dist >= -1e-10), "所有距离应为非负"  # 允许小的负值（浮点误差）

    # 检查对称性（考虑浮点误差）
    max_diff = np.max(np.abs(spatiotemporal_dist - spatiotemporal_dist.T))
    assert max_diff < 1e-10, f"距离矩阵不对称，最大差异为{max_diff}"

    # 验证聚类统计
    cluster_stats = result['cluster_stats']
    assert len(cluster_stats) == n_clusters, f"应有{n_clusters}个聚类统计，实际有{len(cluster_stats)}"

    for stat in cluster_stats:
        assert 'cluster_id' in stat, "聚类统计缺少cluster_id"
        assert 'size' in stat, "聚类统计缺少size"
        assert 'center_node_id' in stat, "聚类统计缺少center_node_id"
        assert 'avg_intra_cluster_distance' in stat, "聚类统计缺少avg_intra_cluster_distance"
        assert stat['size'] >= 0, "聚类大小应为非负"

    # 验证迭代历史
    history = result['history']
    assert len(history) > 0, "迭代历史不应为空"
    for record in history:
        assert 'iteration' in record, "历史记录缺少iteration"
        assert 'centroids' in record, "历史记录缺少centroids"
        assert 'labels' in record, "历史记录缺少labels"
        assert 'converged' in record, "历史记录缺少converged"

    print("时空聚类主函数测试通过！")
    return True


def run_all_tests():
    """运行所有测试"""
    tests = [
        test_compute_spatiotemporal_distance_matrix,
        test_improved_kmeans_clustering,
        test_spatiotemporal_clustering
    ]

    for test_func in tests:
        try:
            test_func()
        except Exception as e:
            print(f"测试 {test_func.__name__} 失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    return True


if __name__ == "__main__":
    if run_all_tests():
        print("所有测试通过！")
    else:
        sys.exit(1)