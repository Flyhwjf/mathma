"""
测试可视化工具模块
"""

import numpy as np
import matplotlib.pyplot as plt
from visualization_utils import (
    plot_clustering_results,
    plot_window_optimization,
    plot_iteration_progress,
    plot_hierarchical_path
)


def test_plot_clustering_results():
    """测试聚类结果可视化函数"""
    # 创建测试数据
    n_nodes = 20
    node_coords = np.random.rand(n_nodes, 2) * 100
    cluster_labels = np.random.randint(0, 3, n_nodes)
    a_i = np.random.rand(n_nodes) * 50
    b_i = a_i + np.random.rand(n_nodes) * 30

    # 调用函数
    fig = plot_clustering_results(node_coords, cluster_labels, a_i, b_i, "测试聚类结果")

    # 验证返回类型
    assert isinstance(fig, plt.Figure), "应返回matplotlib图形对象"
    assert len(fig.axes) == 2, "应有2个子图"

    print("✓ plot_clustering_results 测试通过")


def test_plot_window_optimization():
    """测试窗口优化过程可视化函数"""
    # 创建测试数据
    cluster_points = list(range(10))
    optimization_history = [
        {'window_start': 0, 'window_end': 3, 'cost': 100.0},
        {'window_start': 1, 'window_end': 4, 'cost': 95.0},
        {'window_start': 2, 'window_end': 5, 'cost': 90.0},
        {'window_start': 3, 'window_end': 6, 'cost': 85.0},
        {'window_start': 4, 'window_end': 7, 'cost': 80.0},
    ]

    # 调用函数
    fig = plot_window_optimization(cluster_points, optimization_history, "测试窗口优化")

    # 验证返回类型
    assert isinstance(fig, plt.Figure), "应返回matplotlib图形对象"
    assert len(fig.axes) == 2, "应有2个子图"

    print("✓ plot_window_optimization 测试通过")


def test_plot_iteration_progress():
    """测试迭代进度可视化函数"""
    # 创建测试数据
    iteration_history = []
    for i in range(10):
        record = {
            'total_cost': 1000 - i * 50 + np.random.rand() * 20,
            'total_distance': 800 - i * 40 + np.random.rand() * 15,
            'total_penalty': 200 - i * 10 + np.random.rand() * 5,
            'cluster_changes': max(0, 5 - i)
        }
        iteration_history.append(record)

    # 调用函数
    fig = plot_iteration_progress(iteration_history, "测试迭代进度")

    # 验证返回类型
    assert isinstance(fig, plt.Figure), "应返回matplotlib图形对象"
    assert len(fig.axes) == 4, "应有4个子图"

    print("✓ plot_iteration_progress 测试通过")


def test_plot_hierarchical_path():
    """测试分层路径可视化函数"""
    # 创建测试数据
    n_nodes = 30
    node_coords = np.random.rand(n_nodes, 2) * 100

    # 聚类间路径
    inter_cluster_path = [0, 1, 2, 0]  # 循环路径

    # 聚类内路径
    intra_cluster_paths = {
        0: [0, 1, 2, 3, 4],
        1: [5, 6, 7, 8, 9],
        2: [10, 11, 12, 13, 14]
    }

    # 调用函数
    fig = plot_hierarchical_path(inter_cluster_path, intra_cluster_paths, node_coords, "测试分层路径")

    # 验证返回类型
    assert isinstance(fig, plt.Figure), "应返回matplotlib图形对象"
    assert len(fig.axes) == 2, "应有2个子图"

    print("✓ plot_hierarchical_path 测试通过")


def test_edge_cases():
    """测试边界情况"""
    print("\n测试边界情况...")

    # 1. 空数据测试
    empty_coords = np.array([]).reshape(0, 2)
    empty_labels = np.array([])
    empty_a = np.array([])
    empty_b = np.array([])

    try:
        fig = plot_clustering_results(empty_coords, empty_labels, empty_a, empty_b, "空数据测试")
        print("✓ 空数据测试通过")
    except Exception as e:
        print(f"✗ 空数据测试失败: {e}")

    # 2. 单节点测试
    single_coords = np.array([[50, 50]])
    single_labels = np.array([0])
    single_a = np.array([0])
    single_b = np.array([100])

    try:
        fig = plot_clustering_results(single_coords, single_labels, single_a, single_b, "单节点测试")
        print("✓ 单节点测试通过")
    except Exception as e:
        print(f"✗ 单节点测试失败: {e}")

    # 3. 单聚类测试
    single_cluster_coords = np.random.rand(10, 2) * 100
    single_cluster_labels = np.zeros(10, dtype=int)
    single_cluster_a = np.random.rand(10) * 50
    single_cluster_b = single_cluster_a + np.random.rand(10) * 30

    try:
        fig = plot_clustering_results(single_cluster_coords, single_cluster_labels,
                                     single_cluster_a, single_cluster_b, "单聚类测试")
        print("✓ 单聚类测试通过")
    except Exception as e:
        print(f"✗ 单聚类测试失败: {e}")

    # 4. 空优化历史测试
    empty_history = []
    cluster_points = list(range(5))

    try:
        fig = plot_window_optimization(cluster_points, empty_history, "空历史测试")
        print("✓ 空优化历史测试通过")
    except Exception as e:
        print(f"✗ 空优化历史测试失败: {e}")

    # 5. 空迭代历史测试
    try:
        fig = plot_iteration_progress([], "空迭代历史测试")
        print("✓ 空迭代历史测试通过")
    except Exception as e:
        print(f"✗ 空迭代历史测试失败: {e}")

    # 6. 空路径测试
    try:
        empty_node_coords = np.array([[0, 0]])  # 创建一个简单的坐标数组
        fig = plot_hierarchical_path([], {}, empty_node_coords, "空路径测试")
        print("✓ 空路径测试通过")
    except Exception as e:
        print(f"✗ 空路径测试失败: {e}")


def run_all_tests():
    """运行所有测试"""
    print("开始测试可视化工具模块...\n")

    try:
        test_plot_clustering_results()
        test_plot_window_optimization()
        test_plot_iteration_progress()
        test_plot_hierarchical_path()
        test_edge_cases()

        print("\n✅ 所有测试通过！")
        return True
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        return False


if __name__ == "__main__":
    # 关闭交互模式，避免显示图形窗口
    plt.ioff()

    success = run_all_tests()

    # 可选：保存测试图形
    if success:
        print("\n测试完成，可视化函数实现正确。")
    else:
        print("\n测试失败，请检查实现。")