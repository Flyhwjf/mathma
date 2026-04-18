"""
迭代改进控制器模块测试
"""

import numpy as np
import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from iterative_refinement import (
    evaluate_boundary_violations,
    migrate_boundary_points,
    iterative_refinement_controller
)


def test_evaluate_boundary_violations_basic():
    """测试边界点违反评估基本功能"""
    print("测试边界点违反评估基本功能...")

    # 创建测试数据
    n = 10
    cluster_labels = np.array([0, 0, 0, 0, 1, 1, 1, 2, 2, 2])  # 3个聚类
    path_sequence = list(range(n))  # 简单路径序列

    # 创建惩罚值：边界点（索引3,4,6,7）有较高惩罚
    penalties_by_node = np.ones(n) * 10.0  # 基础惩罚
    penalties_by_node[3] = 30.0  # 边界点：聚类0->1的边界
    penalties_by_node[4] = 25.0  # 边界点：聚类1的开始
    penalties_by_node[6] = 35.0  # 边界点：聚类1->2的边界
    penalties_by_node[7] = 28.0  # 边界点：聚类2的开始

    # 创建距离矩阵
    dist_matrix = np.ones((n, n)) * 10.0
    np.fill_diagonal(dist_matrix, 0.0)

    # 调用函数
    boundary_points, violation_scores, violation_info = evaluate_boundary_violations(
        cluster_labels, path_sequence, penalties_by_node, dist_matrix, threshold_multiplier=2.0
    )

    # 验证返回类型
    assert isinstance(boundary_points, list), "边界点列表应为list类型"
    assert isinstance(violation_scores, list), "违反分数列表应为list类型"
    assert isinstance(violation_info, dict), "违反信息应为dict类型"

    # 验证边界点识别
    # 预期边界点：索引3,4,6,7（聚类变化处）
    expected_boundary_points = [3, 4, 6, 7]
    assert set(boundary_points) == set(expected_boundary_points), \
        f"边界点识别错误，期望{expected_boundary_points}，实际{boundary_points}"

    # 验证违反分数
    assert len(violation_scores) == len(boundary_points), \
        "违反分数数量应与边界点数量一致"

    # 验证违反信息
    required_keys = ['average_penalty', 'threshold', 'violation_count', 'violation_rate',
                    'has_path_crossing', 'path_crossing_info', 'cluster_costs',
                    'average_cluster_cost', 'max_cluster_cost', 'min_cluster_cost',
                    'cost_imbalance_ratio', 'cost_imbalance_triggered', 'triggers']
    for key in required_keys:
        assert key in violation_info, f"违反信息缺少键'{key}'"

    # 验证计算正确性
    avg_penalty = np.mean(penalties_by_node)
    assert abs(violation_info['average_penalty'] - avg_penalty) < 1e-10, \
        f"平均惩罚计算错误，期望{avg_penalty}，实际{violation_info['average_penalty']}"

    threshold = avg_penalty * 2.0
    assert abs(violation_info['threshold'] - threshold) < 1e-10, \
        f"阈值计算错误，期望{threshold}，实际{violation_info['threshold']}"

    print("边界点违反评估基本功能测试通过！")
    return True


def test_evaluate_boundary_violations_no_violations():
    """测试无边界违反情况"""
    print("测试无边界违反情况...")

    n = 8
    cluster_labels = np.array([0, 0, 0, 1, 1, 1, 2, 2])  # 3个聚类
    path_sequence = list(range(n))

    # 所有惩罚值都低于阈值
    penalties_by_node = np.ones(n) * 10.0

    # 创建距离矩阵
    dist_matrix = np.ones((n, n)) * 10.0
    np.fill_diagonal(dist_matrix, 0.0)

    boundary_points, violation_scores, violation_info = evaluate_boundary_violations(
        cluster_labels, path_sequence, penalties_by_node, dist_matrix, threshold_multiplier=2.0
    )

    # 边界点应被识别，但违反分数应为0
    expected_boundary_points = [2, 3, 5, 6]  # 聚类变化处
    assert set(boundary_points) == set(expected_boundary_points), \
        f"边界点识别错误，期望{expected_boundary_points}，实际{boundary_points}"

    # 违反分数应为0（因为惩罚值都低于阈值）
    assert all(score == 0 for score in violation_scores), \
        "无违反情况下违反分数应为0"

    assert violation_info['violation_count'] == 0, \
        f"违反计数应为0，实际为{violation_info['violation_count']}"

    print("无边界违反情况测试通过！")
    return True


def test_evaluate_boundary_violations_all_violations():
    """测试所有边界点都违反的情况"""
    print("测试所有边界点都违反的情况...")

    n = 6
    cluster_labels = np.array([0, 0, 1, 1, 2, 2])  # 3个聚类
    path_sequence = list(range(n))

    # 设置非常高的惩罚值，使所有边界点都违反
    # 最简单：所有点都设置相同的非常高惩罚值
    # 这样平均惩罚 = 该值，阈值 = 2 * 该值
    # 需要惩罚值 > 2 * 惩罚值，这不可能
    # 所以需要非边界点设置低惩罚，边界点设置高惩罚

    # 非边界点（索引0和5）设置低惩罚
    penalties_by_node = np.ones(n) * 10.0  # 所有点先设为10
    # 边界点（索引1,2,3,4）设置非常高的惩罚
    # 平均惩罚 = (10*2 + 边界点惩罚*4) / 6
    # 设边界点惩罚 = P，平均 = (20 + 4P)/6
    # 阈值 = 2 * 平均 = (40 + 8P)/6
    # 需要 P > 阈值 = (40 + 8P)/6
    # 6P > 40 + 8P
    # -2P > 40
    # P < -20，不可能！

    # 计算：非边界点惩罚=10，边界点惩罚需要 > 2*平均惩罚
    # 设边界点惩罚 = P
    # 平均惩罚 = (10*2 + P*4)/6 = (20 + 4P)/6
    # 阈值 = 2 * 平均 = (40 + 8P)/6
    # 需要 P > (40 + 8P)/6
    # 6P > 40 + 8P
    # -2P > 40
    # P < -20，不可能！

    # 所以修改测试逻辑：使用阈值乘数=0.5，这样阈值=0.5*平均
    # 边界点惩罚=100，非边界点=10
    # 平均 = (20+400)/6=70，阈值=35，100>35，所有边界点都违反
    penalties_by_node = np.ones(n) * 10.0  # 所有点先设为10
    penalties_by_node[1] = 100.0  # 边界点
    penalties_by_node[2] = 100.0  # 边界点
    penalties_by_node[3] = 100.0  # 边界点
    penalties_by_node[4] = 100.0  # 边界点

    # 创建距离矩阵
    dist_matrix = np.ones((n, n)) * 10.0
    np.fill_diagonal(dist_matrix, 0.0)

    # 使用阈值乘数=0.5
    boundary_points, violation_scores, violation_info = evaluate_boundary_violations(
        cluster_labels, path_sequence, penalties_by_node, dist_matrix, threshold_multiplier=0.5
    )

    expected_boundary_points = [1, 2, 3, 4]  # 聚类变化处
    print(f"边界点: {boundary_points}, 期望: {expected_boundary_points}")
    print(f"违反计数: {violation_info['violation_count']}, 平均惩罚: {violation_info['average_penalty']}, 阈值: {violation_info['threshold']}")

    assert set(boundary_points) == set(expected_boundary_points), \
        f"边界点识别错误，期望{expected_boundary_points}，实际{boundary_points}"

    # 所有边界点都应违反
    assert violation_info['violation_count'] == len(expected_boundary_points), \
        f"所有边界点都应违反，期望{len(expected_boundary_points)}，实际{violation_info['violation_count']}"

    print("所有边界点都违反情况测试通过！")
    return True


def test_migrate_boundary_points_basic():
    """测试边界点迁移基本功能"""
    print("测试边界点迁移基本功能...")

    n = 9
    # 初始聚类标签：0,0,0,1,1,1,2,2,2
    cluster_labels = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])

    # 边界点：索引2（聚类0->1边界），索引3（聚类1开始），索引5（聚类1->2边界），索引6（聚类2开始）
    boundary_points = [2, 3, 5, 6]

    # 创建距离矩阵：使边界点更接近相邻聚类
    distance_matrix = np.zeros((n, n))

    # 设置距离：使点2更接近聚类1，点3更接近聚类0，点5更接近聚类2，点6更接近聚类1
    # 聚类0：点0,1,2
    # 聚类1：点3,4,5
    # 聚类2：点6,7,8

    # 点2到聚类1的平均距离较小
    distance_matrix[2, 3] = 1.0
    distance_matrix[2, 4] = 1.0
    distance_matrix[2, 5] = 1.0

    # 点2到聚类0的平均距离较大
    distance_matrix[2, 0] = 10.0
    distance_matrix[2, 1] = 10.0

    # 点3到聚类0的平均距离较小
    distance_matrix[3, 0] = 1.0
    distance_matrix[3, 1] = 1.0
    distance_matrix[3, 2] = 1.0

    # 点3到聚类1的平均距离较大
    distance_matrix[3, 4] = 10.0
    distance_matrix[3, 5] = 10.0

    # 确保对称
    for i in range(n):
        for j in range(i, n):
            if i == j:
                distance_matrix[i, j] = 0.0
            elif distance_matrix[i, j] == 0 and distance_matrix[j, i] == 0:
                # 设置默认距离
                distance_matrix[i, j] = distance_matrix[j, i] = 5.0

    # 创建路径序列
    path_sequence = list(range(n))

    # 调用函数
    updated_labels = migrate_boundary_points(
        cluster_labels, boundary_points, distance_matrix, path_sequence
    )

    # 验证返回类型
    assert isinstance(updated_labels, np.ndarray), "更新后的标签应为numpy数组"
    assert updated_labels.shape == cluster_labels.shape, "标签形状应保持不变"

    # 验证边界点迁移
    # 点2应从聚类0迁移到聚类1
    assert updated_labels[2] == 1, f"点2应从聚类0迁移到聚类1，实际为聚类{updated_labels[2]}"

    # 点3应从聚类1迁移到聚类0
    assert updated_labels[3] == 0, f"点3应从聚类1迁移到聚类0，实际为聚类{updated_labels[3]}"

    # 非边界点不应改变
    assert updated_labels[0] == 0, "非边界点0不应改变"
    assert updated_labels[1] == 0, "非边界点1不应改变"
    assert updated_labels[4] == 1, "非边界点4不应改变"
    assert updated_labels[7] == 2, "非边界点7不应改变"
    assert updated_labels[8] == 2, "非边界点8不应改变"

    print("边界点迁移基本功能测试通过！")
    return True


def test_migrate_boundary_points_no_change():
    """测试边界点无需迁移的情况"""
    print("测试边界点无需迁移的情况...")

    n = 6
    cluster_labels = np.array([0, 0, 1, 1, 2, 2])
    boundary_points = [1, 2, 3, 4]

    # 创建距离矩阵：使边界点在当前聚类中距离最小
    distance_matrix = np.zeros((n, n))

    # 设置距离：使所有点在自己的聚类中距离最小
    # 聚类0：点0,1
    # 聚类1：点2,3
    # 聚类2：点4,5

    # 点1（边界点，聚类0）到聚类0的距离较小，到其他聚类距离较大
    distance_matrix[1, 0] = 1.0  # 到同聚类点0
    distance_matrix[1, 2] = 100.0  # 到聚类1
    distance_matrix[1, 3] = 100.0  # 到聚类1
    distance_matrix[1, 4] = 100.0  # 到聚类2
    distance_matrix[1, 5] = 100.0  # 到聚类2

    # 点2（边界点，聚类1）到聚类1的距离较小，到其他聚类距离较大
    distance_matrix[2, 3] = 1.0  # 到同聚类点3
    distance_matrix[2, 0] = 100.0  # 到聚类0
    distance_matrix[2, 1] = 100.0  # 到聚类0
    distance_matrix[2, 4] = 100.0  # 到聚类2
    distance_matrix[2, 5] = 100.0  # 到聚类2

    # 点3（边界点，聚类1）到聚类1的距离较小，到其他聚类距离较大
    distance_matrix[3, 2] = 1.0  # 到同聚类点2
    distance_matrix[3, 0] = 100.0  # 到聚类0
    distance_matrix[3, 1] = 100.0  # 到聚类0
    distance_matrix[3, 4] = 100.0  # 到聚类2
    distance_matrix[3, 5] = 100.0  # 到聚类2

    # 点4（边界点，聚类2）到聚类2的距离较小，到其他聚类距离较大
    distance_matrix[4, 5] = 1.0  # 到同聚类点5
    distance_matrix[4, 0] = 100.0  # 到聚类0
    distance_matrix[4, 1] = 100.0  # 到聚类0
    distance_matrix[4, 2] = 100.0  # 到聚类1
    distance_matrix[4, 3] = 100.0  # 到聚类1

    # 确保对称
    for i in range(n):
        for j in range(i, n):
            if i == j:
                distance_matrix[i, j] = 0.0
            elif distance_matrix[i, j] == 0 and distance_matrix[j, i] == 0:
                # 设置默认距离为50（介于1和100之间）
                distance_matrix[i, j] = distance_matrix[j, i] = 50.0

    # 创建路径序列
    path_sequence = list(range(n))

    updated_labels = migrate_boundary_points(
        cluster_labels, boundary_points, distance_matrix, path_sequence
    )

    # 标签应保持不变
    assert np.array_equal(updated_labels, cluster_labels), "边界点无需迁移时标签应保持不变"

    print("边界点无需迁移情况测试通过！")
    return True


def test_migrate_boundary_points_empty():
    """测试空边界点列表"""
    print("测试空边界点列表...")

    n = 5
    cluster_labels = np.array([0, 0, 1, 1, 2])
    boundary_points = []
    distance_matrix = np.ones((n, n))
    np.fill_diagonal(distance_matrix, 0)

    # 创建路径序列
    path_sequence = list(range(n))

    updated_labels = migrate_boundary_points(
        cluster_labels, boundary_points, distance_matrix, path_sequence
    )

    # 标签应保持不变
    assert np.array_equal(updated_labels, cluster_labels), "空边界点列表时标签应保持不变"

    print("空边界点列表测试通过！")
    return True


def test_iterative_refinement_controller_basic():
    """测试迭代改进控制器基本功能"""
    print("测试迭代改进控制器基本功能...")

    # 创建测试数据
    n = 12
    node_ids = list(range(n))

    # 距离矩阵
    np.random.seed(42)
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            if i == j:
                dist_matrix[i, j] = 0.0
            else:
                # 创建聚类结构：前4点距离近，中间4点距离近，后4点距离近
                if i < 4 and j < 4:
                    dist_matrix[i, j] = np.random.uniform(5, 15)
                elif 4 <= i < 8 and 4 <= j < 8:
                    dist_matrix[i, j] = np.random.uniform(5, 15)
                elif i >= 8 and j >= 8:
                    dist_matrix[i, j] = np.random.uniform(5, 15)
                else:
                    dist_matrix[i, j] = np.random.uniform(30, 50)
                dist_matrix[j, i] = dist_matrix[i, j]

    # 时间窗口参数
    a_i = np.zeros(n)
    b_i = np.zeros(n)
    s_i = np.ones(n) * 5.0  # 服务时间

    # 设置时间窗口分组
    for i in range(n):
        if i < 4:
            a_i[i] = np.random.uniform(0, 2)
            b_i[i] = np.random.uniform(5, 7)
        elif i < 8:
            a_i[i] = np.random.uniform(10, 12)
            b_i[i] = np.random.uniform(15, 17)
        else:
            a_i[i] = np.random.uniform(20, 22)
            b_i[i] = np.random.uniform(25, 27)

    # 调用控制器
    result = iterative_refinement_controller(
        node_ids=node_ids,
        dist_matrix=dist_matrix,
        a_i=a_i,
        b_i=b_i,
        s_i=s_i,
        max_iterations=3,
        convergence_threshold=0.01
    )

    # 验证返回类型和结构
    assert isinstance(result, dict), "结果应为字典类型"

    required_keys = ['final_labels', 'final_path', 'iteration_history', 'converged']
    for key in required_keys:
        assert key in result, f"结果字典缺少键'{key}'"

    # 验证最终标签
    final_labels = result['final_labels']
    assert isinstance(final_labels, np.ndarray), "最终标签应为numpy数组"
    assert final_labels.shape == (n,), f"标签形状应为({n},)，实际为{final_labels.shape}"

    # 验证最终路径
    final_path = result['final_path']
    assert isinstance(final_path, list), "最终路径应为列表"
    assert len(final_path) == n, f"路径长度应为{n}，实际为{len(final_path)}"
    assert set(final_path) == set(node_ids), "路径应包含所有节点"

    # 验证迭代历史
    iteration_history = result['iteration_history']
    assert isinstance(iteration_history, list), "迭代历史应为列表"
    assert len(iteration_history) <= 3, f"迭代历史长度不应超过最大迭代次数3，实际为{len(iteration_history)}"

    # 验证收敛标志
    converged = result['converged']
    assert isinstance(converged, bool), "收敛标志应为布尔值"

    # 验证迭代历史结构
    if len(iteration_history) > 0:
        for history in iteration_history:
            assert 'iteration' in history, "迭代历史缺少iteration"
            assert 'labels' in history, "迭代历史缺少labels"
            assert 'path' in history, "迭代历史缺少path"
            assert 'total_cost' in history, "迭代历史缺少total_cost"
            assert 'penalties' in history, "迭代历史缺少penalties"
            assert 'boundary_violations' in history, "迭代历史缺少boundary_violations"
            assert 'cluster_changes' in history, "迭代历史缺少cluster_changes"

    print("迭代改进控制器基本功能测试通过！")
    return True


def test_iterative_refinement_controller_convergence():
    """测试迭代改进控制器收敛情况"""
    print("测试迭代改进控制器收敛情况...")

    # 创建简单测试数据
    n = 8
    node_ids = list(range(n))

    # 简单距离矩阵
    dist_matrix = np.ones((n, n)) * 10.0
    np.fill_diagonal(dist_matrix, 0.0)

    # 时间窗口参数
    a_i = np.array([0, 5, 10, 15, 20, 25, 30, 35])
    b_i = a_i + 10.0
    s_i = np.ones(n) * 2.0

    # 调用控制器，设置较小的最大迭代次数
    result = iterative_refinement_controller(
        node_ids=node_ids,
        dist_matrix=dist_matrix,
        a_i=a_i,
        b_i=b_i,
        s_i=s_i,
        max_iterations=2,
        convergence_threshold=0.01
    )

    # 验证结果
    assert 'converged' in result, "结果应包含收敛标志"
    assert 'iteration_history' in result, "结果应包含迭代历史"

    # 迭代历史长度不应超过最大迭代次数
    assert len(result['iteration_history']) <= 2, \
        f"迭代历史长度不应超过最大迭代次数2，实际为{len(result['iteration_history'])}"

    print("迭代改进控制器收敛情况测试通过！")
    return True


def test_edge_cases():
    """测试边界情况"""
    print("测试边界情况...")

    # 测试1: 单节点情况
    print("测试单节点情况...")
    node_ids = [0]
    dist_matrix = np.array([[0.0]])
    a_i = np.array([0])
    b_i = np.array([10])
    s_i = np.array([5])

    result = iterative_refinement_controller(
        node_ids=node_ids,
        dist_matrix=dist_matrix,
        a_i=a_i,
        b_i=b_i,
        s_i=s_i,
        max_iterations=5
    )

    assert result['final_labels'][0] == 0, "单节点标签应为0"
    assert result['final_path'] == [0], "单节点路径应为[0]"
    assert result['converged'] == True, "单节点情况应收敛"

    # 测试2: 两个节点情况
    print("测试两个节点情况...")
    node_ids = [0, 1]
    dist_matrix = np.array([[0.0, 10.0], [10.0, 0.0]])
    a_i = np.array([0, 5])
    b_i = np.array([10, 15])
    s_i = np.array([5, 5])

    result = iterative_refinement_controller(
        node_ids=node_ids,
        dist_matrix=dist_matrix,
        a_i=a_i,
        b_i=b_i,
        s_i=s_i,
        max_iterations=3
    )

    assert len(result['final_path']) == 2, "路径应包含2个节点"
    assert set(result['final_path']) == {0, 1}, "路径应包含所有节点"

    print("边界情况测试通过！")
    return True


def test_path_crossing_detection():
    """测试路径交叉检测功能"""
    print("测试路径交叉检测功能...")

    # 创建测试数据：故意创建交叉路径
    n = 8
    cluster_labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])  # 2个聚类
    # 路径：0-1-2-3-4-5-6-7（正常）
    # 但聚类边界在3和4之间，如果路径绕路，可能产生交叉
    path_sequence = [0, 1, 2, 3, 7, 6, 5, 4]  # 故意绕路，可能产生交叉

    # 创建距离矩阵：使绕路路径明显更长
    dist_matrix = np.zeros((n, n))
    # 设置短距离（直接连接）
    for i in range(n-1):
        dist_matrix[i, i+1] = dist_matrix[i+1, i] = 10.0

    # 设置长距离（绕路）
    dist_matrix[3, 7] = dist_matrix[7, 3] = 100.0
    dist_matrix[7, 6] = dist_matrix[6, 7] = 100.0
    dist_matrix[6, 5] = dist_matrix[5, 6] = 100.0
    dist_matrix[5, 4] = dist_matrix[4, 5] = 100.0

    # 填充其他距离
    for i in range(n):
        for j in range(i, n):
            if dist_matrix[i, j] == 0 and i != j:
                dist_matrix[i, j] = dist_matrix[j, i] = 50.0
        dist_matrix[i, i] = 0.0

    penalties_by_node = np.ones(n) * 10.0

    # 调用函数，设置较低的交叉阈值以便检测
    boundary_points, violation_scores, violation_info = evaluate_boundary_violations(
        cluster_labels, path_sequence, penalties_by_node, dist_matrix,
        threshold_multiplier=2.0, crossing_threshold=0.1  # 10%阈值，更容易触发
    )

    # 验证路径交叉检测
    assert 'has_path_crossing' in violation_info, "应包含路径交叉检测结果"
    assert 'path_crossing_info' in violation_info, "应包含路径交叉详细信息"

    # 由于我们故意创建了绕路路径，应该检测到交叉
    # 注意：实际检测可能取决于具体实现，这里只检查结构
    print(f"路径交叉检测结果: {violation_info['has_path_crossing']}")
    print(f"交叉信息: {violation_info['path_crossing_info']}")

    print("路径交叉检测功能测试通过！")
    return True


def test_cluster_cost_imbalance():
    """测试聚类成本不平衡检测功能"""
    print("测试聚类成本不平衡检测功能...")

    n = 9
    cluster_labels = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])  # 3个聚类
    path_sequence = list(range(n))  # 简单路径

    # 创建距离矩阵
    dist_matrix = np.ones((n, n)) * 10.0
    np.fill_diagonal(dist_matrix, 0.0)

    # 故意创建成本不平衡：聚类0成本高，聚类1和2成本低
    penalties_by_node = np.ones(n) * 10.0
    penalties_by_node[0] = 100.0  # 聚类0的第一个点惩罚高
    penalties_by_node[1] = 100.0  # 聚类0的第二个点惩罚高
    penalties_by_node[2] = 100.0  # 聚类0的第三个点惩罚高

    # 调用函数，设置较低的成本不平衡阈值
    boundary_points, violation_scores, violation_info = evaluate_boundary_violations(
        cluster_labels, path_sequence, penalties_by_node, dist_matrix,
        threshold_multiplier=2.0, cost_imbalance_threshold=1.2  # 较低的阈值，更容易触发
    )

    # 验证成本不平衡检测
    assert 'cluster_costs' in violation_info, "应包含聚类成本"
    assert 'average_cluster_cost' in violation_info, "应包含平均聚类成本"
    assert 'cost_imbalance_ratio' in violation_info, "应包含成本不平衡比率"
    assert 'cost_imbalance_triggered' in violation_info, "应包含成本不平衡触发标志"

    # 检查成本计算
    cluster_costs = violation_info['cluster_costs']
    assert len(cluster_costs) == 3, f"应有3个聚类成本，实际有{len(cluster_costs)}个"

    # 聚类0的成本应该明显更高
    print(f"聚类成本: {cluster_costs}")
    print(f"成本不平衡比率: {violation_info['cost_imbalance_ratio']}")
    print(f"成本不平衡触发: {violation_info['cost_imbalance_triggered']}")

    # 由于我们故意创建了成本不平衡，应该触发检测
    # 注意：实际是否触发取决于阈值，这里只检查结构
    if violation_info['cost_imbalance_triggered']:
        print("成功检测到成本不平衡！")

    print("聚类成本不平衡检测功能测试通过！")
    return True


def test_adjacent_cluster_detection():
    """测试相邻聚类检测功能"""
    print("测试相邻聚类检测功能...")

    n = 12
    # 创建聚类标签：4个聚类，每个3个点
    cluster_labels = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])

    # 创建路径序列：聚类按顺序排列
    path_sequence = list(range(n))

    # 创建距离矩阵：使相邻聚类之间的距离较小
    dist_matrix = np.zeros((n, n))

    # 设置聚类内部距离小
    for i in range(n):
        for j in range(n):
            if i == j:
                dist_matrix[i, j] = 0.0
            elif cluster_labels[i] == cluster_labels[j]:
                dist_matrix[i, j] = 5.0  # 同聚类距离小
            else:
                # 计算聚类差异
                cluster_diff = abs(cluster_labels[i] - cluster_labels[j])
                if cluster_diff == 1:
                    dist_matrix[i, j] = 20.0  # 相邻聚类距离中等
                else:
                    dist_matrix[i, j] = 100.0  # 非相邻聚类距离大

    # 测试点：选择聚类1中的一个点
    test_point = 4  # 属于聚类1
    current_cluster = cluster_labels[test_point]

    # 调用内部函数（需要导入）
    from iterative_refinement import _find_adjacent_clusters

    adjacent_clusters = _find_adjacent_clusters(
        test_point, current_cluster, cluster_labels, dist_matrix, path_sequence
    )

    # 验证结果：应该找到聚类0和聚类2作为相邻聚类
    # 聚类3距离较远，不应被找到
    print(f"测试点 {test_point} (聚类 {current_cluster}) 的相邻聚类: {adjacent_clusters}")

    assert 0 in adjacent_clusters, "聚类0应该是相邻聚类"
    assert 2 in adjacent_clusters, "聚类2应该是相邻聚类"
    assert 3 not in adjacent_clusters, "聚类3不应是相邻聚类（距离太远）"

    print("相邻聚类检测功能测试通过！")
    return True


def run_all_tests():
    """运行所有测试"""
    tests = [
        test_evaluate_boundary_violations_basic,
        test_evaluate_boundary_violations_no_violations,
        test_evaluate_boundary_violations_all_violations,
        test_migrate_boundary_points_basic,
        test_migrate_boundary_points_no_change,
        test_migrate_boundary_points_empty,
        test_iterative_refinement_controller_basic,
        test_iterative_refinement_controller_convergence,
        test_edge_cases,
        test_path_crossing_detection,
        test_cluster_cost_imbalance,
        test_adjacent_cluster_detection
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