"""
滑动窗口优化器模块
用于在聚类内部使用滑动窗口进行局部QUBO优化
"""

import numpy as np
import random
import math
from typing import List, Tuple, Dict, Any
from copy import deepcopy


def solve_window_qubo(
    window_points: List[int],
    dist_matrix: np.ndarray,
    a_i: np.ndarray,
    b_i: np.ndarray,
    s_i: np.ndarray,
    lambda_weight: float = 1.0
) -> Tuple[List[int], float, Dict[str, Any]]:
    """
    解决单个窗口的QUBO问题

    参数:
        window_points: 窗口内节点索引列表
        dist_matrix: 完整距离矩阵
        a_i: 时间窗口开始时间数组
        b_i: 时间窗口结束时间数组
        s_i: 服务时间数组
        lambda_weight: 时间窗口惩罚权重 (默认1.0)

    返回:
        window_path: 窗口内优化后的路径序列
        total_cost: 总成本（距离+惩罚）
        qubo_info: QUBO求解相关信息字典
    """
    # 注意：对于滑动窗口优化，窗口可能不包含仓库0
    # 我们优化窗口内的节点顺序，不添加额外节点
    # 保持窗口节点集合不变

    n = len(window_points)

    # 提取子距离矩阵
    sub_dist = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            sub_dist[i, j] = dist_matrix[window_points[i], window_points[j]]

    # 提取子时间窗口参数
    sub_a = np.array([a_i[window_points[i]] for i in range(n)])
    sub_b = np.array([b_i[window_points[i]] for i in range(n)])
    sub_s = np.array([s_i[window_points[i]] for i in range(n)])

    # 构建QUBO矩阵
    qubo_size = n * n
    Q = np.zeros((qubo_size, qubo_size))

    # 距离项（线性路径，非循环）
    for i in range(n):
        for j in range(n):
            if i != j:
                for k in range(n - 1):  # 只连接位置k到k+1，不连接最后一个位置回到第一个
                    idx1 = i * n + k
                    idx2 = j * n + (k + 1)
                    Q[idx1, idx2] += sub_dist[i, j]

    # 时间窗口惩罚项
    # 计算期望到达时间（基于距离估计）
    for i in range(n):
        for k in range(n):
            idx = i * n + k
            # 估计到达时间（简单假设匀速）
            est_arrival = k * 10  # 简单估计

            # 时间窗口惩罚（二次惩罚）
            if est_arrival < sub_a[i]:
                penalty = 10.0 * ((sub_a[i] - est_arrival) ** 2)
            elif est_arrival > sub_b[i]:
                penalty = 20.0 * ((est_arrival - sub_b[i]) ** 2)
            else:
                penalty = 0

            Q[idx, idx] += lambda_weight * penalty

    # 约束项：每个位置只能有一个节点
    for k in range(n):
        for i in range(n):
            idx1 = i * n + k
            for j in range(n):
                if i != j:
                    idx2 = j * n + k
                    Q[idx1, idx2] += 1000  # 大惩罚

    # 约束项：每个节点只能出现在一个位置
    for i in range(n):
        for k in range(n):
            idx1 = i * n + k
            for l in range(n):
                if k != l:
                    idx2 = i * n + l
                    Q[idx1, idx2] += 1000  # 大惩罚

    # 模拟退火求解
    def simulated_annealing(Q, n_iterations=10000, initial_temp=100.0, cooling_rate=0.99):
        """模拟退火求解QUBO"""
        qubo_size = Q.shape[0]
        n = int(math.sqrt(qubo_size))  # 应为窗口大小

        # 随机初始解
        current_solution = np.zeros(qubo_size)
        # 生成随机排列
        perm = list(range(n))
        random.shuffle(perm)
        for i, pos in enumerate(perm):
            current_solution[i * n + pos] = 1

        current_energy = current_solution @ Q @ current_solution

        best_solution = current_solution.copy()
        best_energy = current_energy

        temperature = initial_temp

        for iteration in range(n_iterations):
            # 生成邻居解：交换两个位置
            neighbor = current_solution.copy()

            # 找到当前解对应的排列
            pos_to_node = {}
            node_to_pos = {}
            for idx in range(qubo_size):
                if current_solution[idx] > 0.5:
                    i = idx // n
                    k = idx % n
                    pos_to_node[k] = i
                    node_to_pos[i] = k

            # 随机交换两个位置
            if len(pos_to_node) >= 2:
                pos1, pos2 = random.sample(list(pos_to_node.keys()), 2)
                node1 = pos_to_node[pos1]
                node2 = pos_to_node[pos2]

                # 更新邻居解
                neighbor[node1 * n + pos1] = 0
                neighbor[node1 * n + pos2] = 1
                neighbor[node2 * n + pos2] = 0
                neighbor[node2 * n + pos1] = 1

            neighbor_energy = neighbor @ Q @ neighbor

            # 接受准则
            if neighbor_energy < current_energy:
                current_solution = neighbor
                current_energy = neighbor_energy
            else:
                delta = neighbor_energy - current_energy
                acceptance_prob = math.exp(-delta / temperature)
                if random.random() < acceptance_prob:
                    current_solution = neighbor
                    current_energy = neighbor_energy

            # 更新最优解
            if current_energy < best_energy:
                best_solution = current_solution.copy()
                best_energy = current_energy

            # 降温
            temperature *= cooling_rate

        return best_solution, best_energy

    # 求解QUBO
    solution, energy = simulated_annealing(Q)

    # 解码路径
    window_path = []
    n_nodes = len(window_points)

    # 从解中提取排列
    pos_to_node = {}
    for idx in range(len(solution)):
        if solution[idx] > 0.5:
            i = idx // n_nodes  # 节点索引
            k = idx % n_nodes   # 位置索引
            pos_to_node[k] = i

    # 按位置顺序构建路径
    for pos in range(n_nodes):
        if pos in pos_to_node:
            node_idx = pos_to_node[pos]
            window_path.append(window_points[node_idx])

    # 如果路径包含仓库0，确保以仓库0开始
    if 0 in window_path and window_path[0] != 0:
        # 找到仓库0的位置并旋转路径
        depot_idx = window_path.index(0)
        window_path = window_path[depot_idx:] + window_path[:depot_idx]

    # 计算总成本（距离 + 时间窗口惩罚）
    total_distance = 0
    total_penalty = 0

    # 计算距离
    for i in range(len(window_path) - 1):
        total_distance += dist_matrix[window_path[i], window_path[i + 1]]

    # 计算时间窗口惩罚
    current_time = 0
    for i, node in enumerate(window_path):
        if i > 0:  # 从第二个节点开始计算旅行时间
            travel_time = dist_matrix[window_path[i-1], node]
            current_time += travel_time

        # 服务时间
        current_time += s_i[node]

        # 时间窗口检查（二次惩罚）
        if current_time < a_i[node]:
            total_penalty += 10.0 * ((a_i[node] - current_time) ** 2)
        elif current_time > b_i[node]:
            total_penalty += 20.0 * ((current_time - b_i[node]) ** 2)

    total_cost = total_distance + lambda_weight * total_penalty

    # 构建QUBO信息字典
    qubo_info = {
        'qubo_matrix': Q,
        'solution': solution,
        'energy': energy,
        'total_distance': total_distance,
        'total_penalty': total_penalty,
        'lambda_weight': lambda_weight
    }

    return window_path, total_cost, qubo_info


def sliding_window_optimization(
    cluster_points: List[int],
    dist_matrix: np.ndarray,
    a_i: np.ndarray,
    b_i: np.ndarray,
    s_i: np.ndarray,
    window_size: int = 10,
    step_size: int = 5
) -> Tuple[List[int], List[Dict[str, Any]]]:
    """
    滑动窗口优化主函数

    参数:
        cluster_points: 聚类内节点索引列表
        dist_matrix: 完整距离矩阵
        a_i: 时间窗口开始时间数组
        b_i: 时间窗口结束时间数组
        s_i: 服务时间数组
        window_size: 窗口大小 (默认10)
        step_size: 滑动步长 (默认5)

    返回:
        optimized_path: 优化后的路径序列
        window_history: 每个窗口的优化历史记录列表
    """
    # 确保聚类包含仓库0
    if 0 not in cluster_points:
        cluster_points = [0] + cluster_points

    n = len(cluster_points)

    # 适应小聚类：调整窗口大小和步长
    if n <= window_size:
        window_size = max(3, n - 1)
        step_size = max(1, window_size // 2)

    # 初始化路径：最近邻启发式
    def nearest_neighbor_path(points, dist_matrix):
        """最近邻算法生成初始路径"""
        if not points:
            return []

        # 从仓库0开始
        unvisited = set(points)
        unvisited.remove(0)
        path = [0]
        current = 0

        while unvisited:
            # 找到最近的未访问节点
            nearest = min(unvisited, key=lambda x: dist_matrix[current, x])
            path.append(nearest)
            unvisited.remove(nearest)
            current = nearest

        return path

    # 生成初始路径
    current_path = nearest_neighbor_path(cluster_points, dist_matrix)

    # 计算初始总成本
    def compute_path_cost(path, dist_matrix, a_i, b_i, s_i, lambda_weight=1.0):
        """计算路径总成本（距离+时间窗口惩罚）"""
        if len(path) <= 1:
            return 0, 0, 0

        total_distance = 0
        total_penalty = 0
        current_time = 0

        for i in range(len(path) - 1):
            # 旅行距离
            travel_dist = dist_matrix[path[i], path[i + 1]]
            total_distance += travel_dist

            # 旅行时间（假设速度=1）
            travel_time = travel_dist
            current_time += travel_time

            # 服务时间
            current_time += s_i[path[i + 1]]

            # 时间窗口检查（二次惩罚）
            if current_time < a_i[path[i + 1]]:
                total_penalty += 10.0 * ((a_i[path[i + 1]] - current_time) ** 2)
            elif current_time > b_i[path[i + 1]]:
                total_penalty += 20.0 * ((current_time - b_i[path[i + 1]]) ** 2)

        total_cost = total_distance + lambda_weight * total_penalty
        return total_cost, total_distance, total_penalty

    initial_cost, initial_dist, initial_penalty = compute_path_cost(
        current_path, dist_matrix, a_i, b_i, s_i
    )

    print(f"初始路径: {current_path}, 初始成本: {initial_cost:.2f}")

    # 滑动窗口优化
    window_history = []
    iteration = 0
    max_iterations = 10  # 最大迭代次数

    while iteration < max_iterations:
        improved = False

        # 滑动窗口
        for start_idx in range(0, n - window_size + 1, step_size):
            end_idx = start_idx + window_size

            # 获取窗口内的节点索引（从当前路径中）
            window_indices = list(range(start_idx, min(end_idx, n)))
            window_points = [current_path[i] for i in window_indices]

            # 优化窗口
            optimized_window_path, window_cost, qubo_info = solve_window_qubo(
                window_points=window_points,
                dist_matrix=dist_matrix,
                a_i=a_i,
                b_i=b_i,
                s_i=s_i,
                lambda_weight=1.0
            )

            # 记录窗口优化历史
            window_history.append({
                'iteration': iteration,
                'window_start': start_idx,
                'window_end': end_idx,
                'window_points': window_points,
                'optimized_path': optimized_window_path,
                'total_cost': window_cost,
                'qubo_info': qubo_info
            })

            # 如果优化后的窗口路径不同，则更新当前路径
            if optimized_window_path != window_points:
                # 创建新路径：替换窗口部分
                new_path = current_path.copy()
                for i, idx in enumerate(window_indices):
                    if idx < len(new_path):
                        new_path[idx] = optimized_window_path[i]

                # 计算新路径成本
                new_cost, new_dist, new_penalty = compute_path_cost(
                    new_path, dist_matrix, a_i, b_i, s_i
                )

                # 如果成本降低，则接受新路径
                if new_cost < initial_cost:
                    current_path = new_path
                    initial_cost = new_cost
                    improved = True
                    print(f"迭代 {iteration}, 窗口 [{start_idx}:{end_idx}]: 成本从 {initial_cost:.2f} 降低到 {new_cost:.2f}")

        # 如果没有改进，提前终止
        if not improved:
            print(f"迭代 {iteration}: 无改进，提前终止")
            break

        iteration += 1

    # 最终成本计算
    final_cost, final_dist, final_penalty = compute_path_cost(
        current_path, dist_matrix, a_i, b_i, s_i
    )

    print(f"最终路径: {current_path}, 最终成本: {final_cost:.2f}, 距离: {final_dist:.2f}, 惩罚: {final_penalty:.2f}")

    return current_path, window_history