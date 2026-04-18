"""
滑动窗口优化器模块测试
"""

import numpy as np
import unittest
from sliding_window_optimizer import solve_window_qubo, sliding_window_optimization


class TestSlidingWindowOptimizer(unittest.TestCase):
    """滑动窗口优化器测试类"""

    def setUp(self):
        """测试前准备数据"""
        # 创建测试距离矩阵 (6个点: 0-5, 0是仓库)
        self.dist_matrix = np.array([
            [0, 10, 15, 20, 25, 30],
            [10, 0, 5, 10, 15, 20],
            [15, 5, 0, 5, 10, 15],
            [20, 10, 5, 0, 5, 10],
            [25, 15, 10, 5, 0, 5],
            [30, 20, 15, 10, 5, 0]
        ], dtype=np.float64)

        # 时间窗口参数 (a_i: 开始时间, b_i: 结束时间, s_i: 服务时间)
        self.a_i = np.array([0, 0, 5, 10, 15, 20], dtype=np.float64)
        self.b_i = np.array([100, 50, 60, 70, 80, 90], dtype=np.float64)
        self.s_i = np.array([0, 5, 5, 5, 5, 5], dtype=np.float64)

        # 测试窗口点 (包含仓库0)
        self.window_points = [0, 1, 2, 3]

    def test_solve_window_qubo_basic(self):
        """测试窗口QUBO求解基本功能"""
        # 调用函数
        window_path, total_cost, qubo_info = solve_window_qubo(
            window_points=self.window_points,
            dist_matrix=self.dist_matrix,
            a_i=self.a_i,
            b_i=self.b_i,
            s_i=self.s_i,
            lambda_weight=1.0
        )

        # 验证返回类型
        self.assertIsInstance(window_path, list)
        self.assertIsInstance(total_cost, float)
        self.assertIsInstance(qubo_info, dict)

        # 验证路径包含所有窗口点
        self.assertEqual(len(window_path), len(self.window_points))
        self.assertEqual(set(window_path), set(self.window_points))

        # 验证路径以仓库0开始
        self.assertEqual(window_path[0], 0)

        # 验证成本非负
        self.assertGreaterEqual(total_cost, 0)

        # 验证QUBO信息包含必要字段
        self.assertIn('qubo_matrix', qubo_info)
        self.assertIn('solution', qubo_info)
        self.assertIn('energy', qubo_info)

        print(f"窗口QUBO测试通过: 路径={window_path}, 成本={total_cost:.2f}")

    def test_solve_window_qubo_small_window(self):
        """测试小窗口QUBO求解"""
        small_window = [0, 1, 2]

        window_path, total_cost, qubo_info = solve_window_qubo(
            window_points=small_window,
            dist_matrix=self.dist_matrix,
            a_i=self.a_i,
            b_i=self.b_i,
            s_i=self.s_i,
            lambda_weight=1.0
        )

        self.assertEqual(len(window_path), len(small_window))
        self.assertEqual(set(window_path), set(small_window))
        self.assertEqual(window_path[0], 0)
        self.assertGreaterEqual(total_cost, 0)

        print(f"小窗口QUBO测试通过: 路径={window_path}, 成本={total_cost:.2f}")

    def test_solve_window_qubo_different_lambda(self):
        """测试不同惩罚权重的QUBO求解"""
        window_path1, total_cost1, qubo_info1 = solve_window_qubo(
            window_points=self.window_points,
            dist_matrix=self.dist_matrix,
            a_i=self.a_i,
            b_i=self.b_i,
            s_i=self.s_i,
            lambda_weight=0.5  # 较小权重
        )

        window_path2, total_cost2, qubo_info2 = solve_window_qubo(
            window_points=self.window_points,
            dist_matrix=self.dist_matrix,
            a_i=self.a_i,
            b_i=self.b_i,
            s_i=self.s_i,
            lambda_weight=2.0  # 较大权重
        )

        # 不同权重可能产生不同结果，但都应有效
        self.assertEqual(len(window_path1), len(self.window_points))
        self.assertEqual(len(window_path2), len(self.window_points))
        self.assertEqual(window_path1[0], 0)
        self.assertEqual(window_path2[0], 0)

        print(f"不同权重测试通过: λ=0.5时成本={total_cost1:.2f}, λ=2.0时成本={total_cost2:.2f}")

    def test_sliding_window_optimization(self):
        """测试滑动窗口优化主函数"""
        # 测试聚类点 (包含仓库0)
        cluster_points = [0, 1, 2, 3, 4, 5]

        # 调用滑动窗口优化
        optimized_path, window_history = sliding_window_optimization(
            cluster_points=cluster_points,
            dist_matrix=self.dist_matrix,
            a_i=self.a_i,
            b_i=self.b_i,
            s_i=self.s_i,
            window_size=4,
            step_size=2
        )

        # 验证返回类型
        self.assertIsInstance(optimized_path, list)
        self.assertIsInstance(window_history, list)

        # 验证路径包含所有聚类点
        self.assertEqual(len(optimized_path), len(cluster_points))
        self.assertEqual(set(optimized_path), set(cluster_points))

        # 验证路径以仓库0开始
        self.assertEqual(optimized_path[0], 0)

        # 验证窗口历史记录
        self.assertGreater(len(window_history), 0)
        for history in window_history:
            self.assertIsInstance(history, dict)
            self.assertIn('window_points', history)
            self.assertIn('optimized_path', history)
            self.assertIn('total_cost', history)

        print(f"滑动窗口优化测试通过: 优化路径={optimized_path}, 窗口优化次数={len(window_history)}")

    def test_sliding_window_optimization_small_cluster(self):
        """测试小聚类滑动窗口优化"""
        small_cluster = [0, 1, 2, 3]

        optimized_path, window_history = sliding_window_optimization(
            cluster_points=small_cluster,
            dist_matrix=self.dist_matrix,
            a_i=self.a_i,
            b_i=self.b_i,
            s_i=self.s_i,
            window_size=3,
            step_size=1
        )

        self.assertEqual(len(optimized_path), len(small_cluster))
        self.assertEqual(set(optimized_path), set(small_cluster))
        self.assertEqual(optimized_path[0], 0)

        print(f"小聚类滑动窗口优化测试通过: 路径={optimized_path}")


if __name__ == '__main__':
    unittest.main()