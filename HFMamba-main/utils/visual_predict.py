import numpy as np
import spectral as spy
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from spectral import spy_colors
import torch

from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans




def visualize_predict(gt,predict_label,save_predict_path,save_gt_path,only_vis_label=False):
    row, col = gt.shape[0], gt.shape[1]
    predict = np.reshape(predict_label,(row,col)) + 1
    if only_vis_label:
        vis_predict = np.where(gt==0,gt,predict)
    else:
        vis_predict = predict
    spy.save_rgb(save_predict_path, vis_predict, colors=spy_colors)
    spy.save_rgb(save_gt_path, gt, colors=spy_colors)



# def visualize_full_tsne(features: torch.Tensor, labels: torch.Tensor, save_path: str, dataset_name: str):
#     """
#     完整样本的t-SNE可视化
#
#     Args:
#         features (torch.Tensor): 输入特征，形状为 [1, C, H, W]。
#         labels (torch.Tensor): 对应标签，形状为 [1, H, W]。
#         save_path (str): 保存图片的路径。
#         dataset_name (str): 数据集名称，用于标题。
#     """
#     # 确保features和labels是NumPy数组
#     if isinstance(features, torch.Tensor):
#         features = features.cpu().numpy()
#     if isinstance(labels, torch.Tensor):
#         labels = labels.cpu().numpy()
#
#     # 假设输入的features是 [1, C, H, W]
#     # 需要将其重塑为 [H*W, C] 以便t-SNE处理
#     # 移除批次维度
#     features = features.squeeze(0) # 形状变为 [C, H, W]
#     labels = labels.squeeze(0)     # 形状变为 [H, W]
#
#     # 将特征从 [C, H, W] 重塑为 [H*W, C]
#     # 或者从 [H, W, C] 重塑为 [H*W, C] 如果你想交换维度
#     # 这里假设C是特征维度，H*W是样本数
#     num_channels = features.shape[0]
#     height, width = features.shape[1], features.shape[2]
#     reshaped_features = features.transpose(1, 2, 0).reshape(-1, num_channels) # 从 [C, H, W] -> [H, W, C] -> [H*W, C]
#
#     # 将标签从 [H, W] 重塑为 [H*W]
#     reshaped_labels = labels.reshape(-1)
#
#     # 定义颜色映射
#     colormap = plt.colormaps['tab20']
#
#     # 执行t-SNE
#     # 降低perplexity以适应可能的大样本量，或者根据实际样本数量调整
#     # t-SNE的计算复杂度很高，对于 H*W 数量的样本，可能需要较长时间
#     tsne = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=42, verbose=1)
#     embeddings = tsne.fit_transform(reshaped_features)
#
#     # 可视化
#     plt.figure(figsize=(15, 12))
#     scatter = plt.scatter(embeddings[:, 0], embeddings[:, 1],
#                           c=reshaped_labels, cmap=colormap,
#                           alpha=0.7, s=5, edgecolors='none')
#
#     # 添加颜色条
#     # 确保ticks只包含实际存在的类别ID，并且是整数
#     unique_labels = np.unique(reshaped_labels)
#     cbar = plt.colorbar(scatter, ticks=unique_labels)
#     cbar.set_label('Class ID')
#
#     plt.title(f'Full Dataset t-SNE Visualization ({dataset_name})')
#     plt.xlabel('t-SNE Dimension 1')
#     plt.ylabel('t-SNE Dimension 2')
#     plt.savefig(save_path, dpi=300, bbox_inches='tight')
#     plt.close()


# def visualize_full_tsne(
#     features: torch.Tensor,
#     labels: torch.Tensor,
#     save_path: str,
#     dataset_name: str,
#     max_points_per_class: int = 20,
#     ignore_index: int = -1,
#     random_state: int = 42
# ):
#     """
#     仅使用“标注点”做t-SNE：过滤 ignore_index，并且每个类别最多抽 max_points_per_class 个点。
#     features: [1, C, H, W]
#     labels:   [1, H, W] 或 [H, W]
#     """
#     # --- 1. Tensor -> NumPy ---
#     if isinstance(features, torch.Tensor):
#         features = features.detach().cpu().numpy()
#     if isinstance(labels, torch.Tensor):
#         labels = labels.detach().cpu().numpy()
#
#     # --- 2. squeeze batch 维 ---
#     if features.ndim == 4:
#         features = features.squeeze(0)  # [C, H, W]
#     if labels.ndim == 3:
#         labels = labels.squeeze(0)     # [H, W]
#
#     # --- 3. reshape 成 [H*W, C] 和 [H*W] ---
#     C, H, W = features.shape
#     reshaped_features = features.transpose(1, 2, 0).reshape(-1, C)  # [H*W, C]
#     reshaped_labels = labels.reshape(-1)                              # [H*W]
#
#     # --- 4. 过滤标注点（忽略背景/无效）---
#     valid_mask = reshaped_labels != ignore_index
#     valid_features = reshaped_features[valid_mask]
#     valid_labels = reshaped_labels[valid_mask]
#
#     if valid_features.shape[0] < 2:
#         print("[t-SNE] Not enough labeled points after filtering.")
#         return
#
#     # --- 5. 每类最多抽 max_points_per_class 个点 ---
#     rng = np.random.default_rng(random_state)
#
#     sampled_features_list = []
#     sampled_labels_list = []
#
#     unique_cls = np.unique(valid_labels)
#     for cls_id in unique_cls:
#         cls_mask = (valid_labels == cls_id)
#         cls_features = valid_features[cls_mask]
#         n = cls_features.shape[0]
#         if n == 0:
#             continue
#
#         if n > max_points_per_class:
#             idx = rng.choice(n, size=max_points_per_class, replace=False)
#             cls_features = cls_features[idx]
#             cls_labels = valid_labels[cls_mask][idx]
#         else:
#             cls_labels = valid_labels[cls_mask]
#
#         sampled_features_list.append(cls_features)
#         sampled_labels_list.append(cls_labels)
#
#     sampled_features = np.concatenate(sampled_features_list, axis=0)  # [N, C]
#     sampled_labels = np.concatenate(sampled_labels_list, axis=0)      # [N]
#
#     if sampled_features.shape[0] < 2:
#         print("[t-SNE] Not enough sampled points per class.")
#         return
#
#     # --- 6. t-SNE：perplexity 必须 < N，并与样本规模匹配 ---
#     n_samples = sampled_features.shape[0]
#     perplexity = min(30, max(5, (n_samples - 1) // 3))
#     # 再做一次保险（极端情况下 N 很小）
#     perplexity = min(perplexity, n_samples - 1)
#
#     tsne = TSNE(
#         n_components=2,
#         perplexity=perplexity,
#         max_iter=1000,
#         random_state=random_state,
#         verbose=1
#     )
#     embeddings = tsne.fit_transform(sampled_features)  # [N, 2]
#
#     # --- 7. 绘图：颜色按“真值类别” ---
#     plt.figure(figsize=(15, 12))
#
#     unique_labels = np.unique(sampled_labels).astype(int)
#     n_colors = len(unique_labels)
#
#     # tab20 只有20色；类别多时就用连续colormap并离散化
#     if n_colors <= 20:
#         colormap = plt.colormaps['tab20']
#         # matplotlib 的 scatter cmap 对离散标签时通常可以工作，但颜色数可能不够；
#         # 这里保持你原思路，用tab20即可。
#     else:
#         colormap = plt.cm.get_cmap('nipy_spectral', n_colors)
#
#     scatter = plt.scatter(
#         embeddings[:, 0], embeddings[:, 1],
#         c=sampled_labels.astype(int),
#         cmap=colormap,
#         alpha=0.8,
#         s=15,
#         edgecolors='none'
#     )
#
#     cbar = plt.colorbar(scatter, ticks=unique_labels)
#     cbar.set_label('Class ID')
#
#     plt.title(f'Labeled-points t-SNE (max {max_points_per_class}/class) - {dataset_name}')
#     plt.xlabel('t-SNE Dimension 1')
#     plt.ylabel('t-SNE Dimension 2')
#     plt.savefig(save_path, dpi=300, bbox_inches='tight')
#     plt.close()



def visualize_full_tsne(
    features: torch.Tensor,
    labels: torch.Tensor,
    save_path: str,
    dataset_name: str,
    ignore_index: int = -1,
    random_state: int = 42,
    # ===== metrics params (scheme F uses prototypes; k_same is unused) =====
    k_same: int = 5,        # 不再使用（保留接口兼容你旧代码）
    k_search: int = 50,    # 用作 separation 的 prototype 查询邻居数上限（近似加速）
    normalize_features: bool = True,
    # ===== scheme F params =====
    prototype_max_per_class: int = 5,   # 每个类最多生成多少个 prototype
    separation_proto_k: int = None,      # 覆盖 k_search 用于 separation 的查询邻居数；None则用min(M_total,k_search)
    # ===== distance clip =====
    dist_clip_min: float = 0.0,
    dist_clip_max: float = 10.0,
    # ===== tsne params =====
    tsne_max_iter: int = 1000
):
    if isinstance(features, torch.Tensor):
        features = features.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()

    # squeeze batch
    if features.ndim == 4:
        features = features.squeeze(0)  # [C, H, W]
    if labels.ndim == 3:
        labels = labels.squeeze(0)     # [H, W]

    C, H, W = features.shape
    num_channels = C
    reshaped_features = features.transpose(1, 2, 0).reshape(-1, num_channels)  # [H*W, C]
    reshaped_labels = labels.reshape(-1)                                     # [H*W]

    # ===== 过滤 ignore_index（只保留标注点）=====
    valid_mask = reshaped_labels != ignore_index
    sampled_features = reshaped_features[valid_mask]                  # [N, D]
    sampled_labels = reshaped_labels[valid_mask].astype(int)         # [N]
    N = sampled_features.shape[0]

    if N < 2:
        print("[t-SNE] Not enough labeled points after filtering.")
        return None

    # 可选：特征归一化
    if normalize_features:
        norm = np.linalg.norm(sampled_features, axis=1, keepdims=True) + 1e-12
        sampled_features = sampled_features / norm

    # ============================================================
    # 方案 F：每类多原型（KMeans），Cohesion/Separation 用 prototype 距离
    # ============================================================
    unique_classes = np.unique(sampled_labels)

    prototypes_list = []
    proto_class_list = []

    for cls_id in unique_classes:
        Xc = sampled_features[sampled_labels == cls_id]  # [nc, D]
        nc = Xc.shape[0]
        m = min(prototype_max_per_class, nc)

        if m <= 1:
            # 只有 1 个样本时，prototype 就用该点本身
            centroids = Xc[:1].copy()
        else:
            # KMeans 得到 m 个中心
            km = KMeans(
                n_clusters=m,
                init="k-means++",
                n_init=10,
                random_state=random_state
            )
            km.fit(Xc)
            centroids = km.cluster_centers_  # [m, D]

        prototypes_list.append(centroids)
        proto_class_list.append(np.full((centroids.shape[0],), cls_id, dtype=int))

    prototypes = np.concatenate(prototypes_list, axis=0)       # [M, D]
    proto_classes = np.concatenate(proto_class_list, axis=0)   # [M]
    M_total = prototypes.shape[0]

    if M_total < 2:
        print("[t-SNE] Not enough prototypes to compute separation.")
        return None

    # ---- Cohesion：到同类最近 prototype 的距离（可精确）----
    cohesion_point = np.zeros(N, dtype=np.float64)

    for cls_id in unique_classes:
        idx_i = np.where(sampled_labels == cls_id)[0]  # samples in this class
        P_c = prototypes[proto_classes == cls_id]       # prototypes for this class

        nn_c = NearestNeighbors(n_neighbors=1, metric='euclidean', algorithm='auto')
        nn_c.fit(P_c)
        d, _ = nn_c.kneighbors(sampled_features[idx_i], return_distance=True)
        cohesion_point[idx_i] = d[:, 0]

    # ---- Separation：到最近“异类 prototype”的距离（用 top-k prototype 邻居近似加速）----
    # separation_proto_k 决定每个样本查多少个最邻近 prototypes，然后取最近的异类那个
    if separation_proto_k is None:
        separation_proto_k = min(M_total, k_search)
    separation_proto_k = max(1, min(separation_proto_k, M_total))

    nn_proto = NearestNeighbors(n_neighbors=separation_proto_k, metric='euclidean', algorithm='auto')
    nn_proto.fit(prototypes)

    # distances_proto: [N, Kp]，indices_proto: [N, Kp]
    distances_proto, indices_proto = nn_proto.kneighbors(sampled_features, return_distance=True)

    separation_point = np.full(N, np.nan, dtype=np.float64)
    for i in range(N):
        li = sampled_labels[i]
        neigh_proto_idxs = indices_proto[i]          # prototype indices, sorted by distance
        neigh_proto_classes = proto_classes[neigh_proto_idxs]

        # 找到第一个“异类prototype”（因为已按距离排序，所以就是最近异类）
        mask_diff = (neigh_proto_classes != li)
        if np.any(mask_diff):
            j = np.argmax(mask_diff)  # first True
            separation_point[i] = distances_proto[i, j]
        else:
            separation_point[i] = np.nan

    # ---- 距离裁剪到 [0, 10] ----
    cohesion_point = np.clip(cohesion_point, dist_clip_min, dist_clip_max)
    separation_point = np.clip(separation_point, dist_clip_min, dist_clip_max)

    # ============================================================
    # Macro：每类一票的平均
    # ============================================================
    cohesion_per_class = {}
    separation_per_class = {}

    for cls_id in unique_classes:
        cls_mask = (sampled_labels == cls_id)
        cohesion_per_class[cls_id] = float(np.nanmean(cohesion_point[cls_mask]))
        separation_per_class[cls_id] = float(np.nanmean(separation_point[cls_mask]))

    cohesion_macro = float(np.mean(list(cohesion_per_class.values()))) if len(cohesion_per_class) > 0 else np.nan
    separation_macro = float(np.mean(list(separation_per_class.values()))) if len(separation_per_class) > 0 else np.nan

    print(
        f"[Metrics(F-proto)] {dataset_name} | "
        f"Cohesion(macro,↓)= {cohesion_macro:.2f} | "
        f"Separation(macro,↑)= {separation_macro:.2f}"
    )

    # ===== 5) t-SNE 绘图：仍用全部标注点 =====
    perplexity = min(30, max(5, (N - 1) // 3))
    perplexity = min(perplexity, N - 1)

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        max_iter=tsne_max_iter,
        random_state=random_state,
        verbose=1
    )
    embeddings = tsne.fit_transform(sampled_features)  # [N, 2]

    # ===== 6) 可视化 =====
    plt.figure(figsize=(15, 12))
    unique_labels = np.unique(sampled_labels)
    n_colors = len(unique_labels)

    if n_colors <= 20:
        colormap = plt.colormaps['tab20']
    else:
        colormap = plt.cm.get_cmap('nipy_spectral', n_colors)

    scatter = plt.scatter(
        embeddings[:, 0], embeddings[:, 1],
        c=sampled_labels.astype(int),
        cmap=colormap,
        alpha=0.85,
        s=25,
        edgecolors='none'
    )

    cbar = plt.colorbar(scatter, ticks=np.sort(unique_labels))
    cbar.set_label('Class ID')

    plt.title(
        f'Labeled-points t-SNE ({dataset_name})\n'
        f'All labeled points, N={N} | Cohesion={cohesion_macro:.2f}, Separation={separation_macro:.2f}'
    )
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    return {
        "N": N,
        "Cohesion_macro": cohesion_macro,
        "Separation_macro": separation_macro,
        "cohesion_per_class": cohesion_per_class,
        "separation_per_class": separation_per_class,
        "prototype_max_per_class": prototype_max_per_class,
        "M_total_prototypes": int(M_total)
    }