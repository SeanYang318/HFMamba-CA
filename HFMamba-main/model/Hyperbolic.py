import torch
from geoopt.manifolds import PoincareBall
from torch import nn


class HyperbolicSumModule(torch.nn.Module):

    def __init__(self, c: float = 1.0, learnable_c: bool = True):
        """
        Args:
            c (float): 庞加莱球模型的曲率参数。必须大于0。
            learnable_c (bool): 如果为 True，则曲率 c 将作为一个可学习参数。
        """
        super().__init__()
        if c <= 0:
            raise ValueError("曲率 c 必须大于 0。")

        self.manifold = PoincareBall(c=c, learnable=learnable_c)  # 当 learnable_c 为 True 时，c 会被注册为模型的可学习参数。

        self.to_hyperbolic_activation = torch.nn.Tanh()

        self.w_1 = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.w_2 = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))


    def _map_to_poincare_ball(self, x: torch.Tensor) -> torch.Tensor:
        """
        将欧几里得空间数据粗略地映射到庞加莱球模型中。
        这里使用 tanh 确保点在单位球内部，然后使用 project 将其投影到流形上，
        以处理浮点精度问题导致的微小超出。
        """
        x_mapped = self.to_hyperbolic_activation(x)  # 使用 tanh 将数据压缩到 (-1, 1) 范围
        x_poincare = self.manifold.projx(x_mapped)
        return x_poincare

    def _map_from_poincare_ball(self, x_poincare: torch.Tensor) -> torch.Tensor:
        """
        将庞加莱球双曲空间中的数据映射回欧几里得空间。
        这里使用 logmap0 将双曲点映射到原点的切空间，这可以被看作是欧几里得向量。
        """
        x_euclidean = self.manifold.logmap0(x_poincare)  # logmap0 (对数映射到原点) 将双曲点映射到原点的切空间中的向量。
        return x_euclidean

    def forward(self, x1, x2, x3):

        # 1. 将欧几里得数据映射到庞加莱球双曲空间
        h_x1 = self._map_to_poincare_ball(x1)
        h_x2 = self._map_to_poincare_ball(x2)
        h_x3 = self._map_to_poincare_ball(x3)

        # w_1 = torch.sigmoid(self.w_1)
        # w_2 = torch.sigmoid(self.w_2)

        w_1 = self.w_1
        w_2 = self.w_2


        # 对于两个点的“相加”，Möbius Addition 更加直接。
        h_sum_h = self.manifold.mobius_add(w_1 * h_x1, (1 - w_1) * h_x2)
        h_sum = self.manifold.mobius_add(w_2 * h_sum_h, (1 - w_2) * h_x3)

        # 3. 将双曲求和结果映射回欧几里得空间
        output_euclidean = self._map_from_poincare_ball(h_sum)
        # output_euclidean = h_sum

        return output_euclidean

    @property
    def curvature(self) -> torch.Tensor:
        """返回当前的曲率 c。"""
        return self.manifold.c