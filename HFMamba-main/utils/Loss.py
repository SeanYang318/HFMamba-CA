import warnings
import torch.nn.functional as F
import torch
from scipy.special import logsumexp
from pyexpat import features



def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None,
           warning=True):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if ((output_h > 1 and output_w > 1 and input_h > 1
                     and input_w > 1) and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)):
                    warnings.warn(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')
    return F.interpolate(input, size, scale_factor, mode, align_corners)





def calculate_contrastive_loss(category_seg_features, num_classes, temperature = 1):

    total = []
    actual_category_keys = sorted(category_seg_features.keys())
    for i in actual_category_keys:
        features = category_seg_features[i]
        features_normalized = torch.nn.functional.normalize(features, p=2, dim=1)
        sim_matrix = torch.matmul(features_normalized, features_normalized.T)
        total_sum = torch.sum(torch.exp(sim_matrix)) / sim_matrix.shape[0] ** 2

        total_cl = 0
        for j in actual_category_keys:
            if j != i:
                features_cl = category_seg_features[j]
                features_normalized_cl = torch.nn.functional.normalize(features_cl, p=2, dim=1)
                sim_matrix_cl = torch.matmul(features_normalized, features_normalized_cl.T)
                total_sum_class = torch.sum(torch.exp(sim_matrix_cl)) / sim_matrix_cl.shape[0] ** 2
                total_cl += total_sum_class
            else:
                total_cl += 0
        total.append(-torch.log(total_sum/total_cl))

    loss = sum(total) / num_classes

    return loss





def supervised_contrastive_loss(seg_logits,label):

    batch_size, num_classes, H, W = seg_logits.shape
    category_indices_list = {i: [] for i in range(num_classes)}
    non_negative_mask = (label[0] != -1)
    non_negative_indices = torch.nonzero(non_negative_mask, as_tuple=True)  # 返回 (rows_tensor, cols_tensor)
    for r, c in zip(non_negative_indices[0], non_negative_indices[1]):
        class_id = label[0, r, c].item()
        if 0 <= class_id <= num_classes - 1 :
            category_indices_list[class_id].append((r.item(), c.item()))

    category_seg_features = {}
    classes = 0
    for class_id in range(num_classes):
        current_class_indices_list = category_indices_list[class_id]
        if not current_class_indices_list:
            # 如果当前类别没有像素，就跳过这个类别，因为它没有特征可以提取。
            continue
        indices_tensor = torch.tensor(current_class_indices_list)
        rows = indices_tensor[:, 0]  # 形状 (N,)
        cols = indices_tensor[:, 1]  # 形状 (N,)
        extracted_features = seg_logits[0, :, rows, cols]
        category_seg_features[class_id] = extracted_features.T
        classes += 1


    return calculate_contrastive_loss(category_seg_features,classes)



def unsupervised_contrastive_loss(seg_logits,label):

    _, C, H, W = seg_logits.shape
    pixel_features = seg_logits.squeeze(0).permute(1, 2, 0)  # (1, C, H, W) 形状变为 (H, W, C)

    non_negative_mask = (label[0] != -1)
    non_negative_indices = torch.nonzero(non_negative_mask, as_tuple=True)  # 返回 (rows_tensor, cols_tensor)

    similarities_sum = 0.0
    n = 0
    for r, c in zip(non_negative_indices[0], non_negative_indices[1]):
        current_pixel = pixel_features[r, c]

        similarities = []

        neighbor_offsets = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1), (0, 1),
            (1, -1), (1, 0), (1, 1)
        ]

        for dr, dc in neighbor_offsets:
            nr, nc = r + dr, c + dc  # 邻居像素的坐标

            # 检查邻居坐标是否在图像边界内
            if 0 <= nr < H and 0 <= nc < W:
                neighbor_pixel = pixel_features[nr, nc]
                similarity = F.cosine_similarity(current_pixel.unsqueeze(0),neighbor_pixel.unsqueeze(0))
                similarities.append(similarity)

        sorted_similarities = sorted(similarities, reverse=True)
        top_3_values = sorted_similarities[:4]
        top_3_similarities = sum(top_3_values)/4.0
        similarities_sum += top_3_similarities
        n += 1

    loss = - (similarities_sum/n)

    return loss




def head_loss(loss_func,logits,label,r1,r2,r3,align_corners=True):
    # seg_logits = resize(
    #     input=logits,
    #     size=label.shape[1:],
    #     mode='bilinear',
    #     align_corners=align_corners)

    seg_logits = logits

    loss_cl = loss_func(seg_logits, label)
    loss_sc = supervised_contrastive_loss(seg_logits,label)
    # loss_uc = unsupervised_contrastive_loss(seg_logits,label)

    # loss = loss_cl
    loss = r1 * loss_cl + r2 * loss_sc
    # loss = r1 * loss_cl + r2 * loss_sc + r3 * loss_uc

    return loss
