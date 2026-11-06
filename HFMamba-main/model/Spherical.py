import torch
import torch.nn as nn
import torch.nn.functional as F


def slerp(a: torch.Tensor, b: torch.Tensor, t: torch.Tensor, eps: float = 1e-7):
    """
    球面 slerp（沿大圆测地线插值）
    a, b: (..., d) 且已满足 ||a||=||b||=1
    t: 标量或可广播到 (..., 1)，建议落在 [0,1]
    """
    dot = (a * b).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)  # (...,1)
    theta = torch.acos(dot)                                   # (...,1)

    sin_theta = torch.sin(theta)
    near_zero = sin_theta.abs() < eps

    # theta 很小时：用线性插值再归一化
    lin = F.normalize((1 - t) * a + t * b, dim=-1)

    t_theta = t * theta
    coef_a = torch.sin((1 - t) * theta) / (sin_theta + eps)
    coef_b = torch.sin(t_theta) / (sin_theta + eps)
    out = coef_a * a + coef_b * b

    out = torch.where(near_zero.expand_as(out), lin, out)
    return F.normalize(out, dim=-1)


def logmap_north_pole_unit_sphere(x: torch.Tensor, eps: float = 1e-7):
    """
    x: (..., d)，且 ||x||=1
    将单位球上的点映射到北极点 p0=[0,..,0,1] 的切空间（输出 v ⟂ p0）
    返回 v: (..., d)
    """
    d = x.size(-1)
    p0 = torch.zeros(d, device=x.device, dtype=x.dtype)
    p0[-1] = 1.0
    p0 = p0.view(*(1,) * (x.dim() - 1), d)  # 广播

    cos_theta = (p0 * x).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)
    theta = torch.acos(cos_theta)  # (...,1)

    sin_theta = torch.sin(theta)
    near_zero = sin_theta.abs() < eps

    v = (theta / (sin_theta + eps)) * (x - cos_theta * p0)
    v = torch.where(near_zero.expand_as(v), torch.zeros_like(v), v)
    return v


class SphericalSumModule(nn.Module):
    """
    输出：切空间向量（选项1）
    调用方式：fusion_x = hy(spa_x, spe_x, x)
    """
    def __init__(self, learnable_w1: bool = True, learnable_w2: bool = True):
        super().__init__()
        if learnable_w1:
            self.w_1 = nn.Parameter(torch.tensor(0.0))
        else:
            self.register_buffer("w_1", torch.tensor(0.0))

        if learnable_w2:
            self.w_2 = nn.Parameter(torch.tensor(0.0))
        else:
            self.register_buffer("w_2", torch.tensor(0.0))

        self.to_sphere_activation = nn.Tanh()

    def _map_to_unit_sphere(self, x: torch.Tensor) -> torch.Tensor:
        x = self.to_sphere_activation(x)
        return F.normalize(x, dim=-1, eps=1e-9)

    def forward(self, spa_x: torch.Tensor, spe_x: torch.Tensor, x: torch.Tensor):
        # 1) 映射到单位球
        h1 = self._map_to_unit_sphere(spa_x)
        h2 = self._map_to_unit_sphere(spe_x)
        h3 = self._map_to_unit_sphere(x)

        # 2) 两次球面测地线融合
        t1 = torch.sigmoid(self.w_1)  # 标量，广播用
        t2 = torch.sigmoid(self.w_2)

        h12 = slerp(h1, h2, t1)
        h123 = slerp(h12, h3, t2)

        # 3) 球面点 -> 切空间向量（输出选项1）
        out = logmap_north_pole_unit_sphere(h123)
        return out