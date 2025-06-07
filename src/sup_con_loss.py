from __future__ import print_function

from collections import defaultdict

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
def sup_con_loss(features, temperature=0.07, contrast_mode='all', base_temperature=0.07, labels=None, mask=None,
                 device=None):
    """Compute loss for model. If both `labels` and `mask` are None,
    it degenerates to SimCLR unsupervised loss:
    https://arxiv.org/pdf/2002.05709.pdf

    Args:
        features: hidden vector of shape [bsz, n_views, ...].
        labels: ground truth of shape [bsz].
        mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
            has the same class as sample i. Can be asymmetric.
    Returns:
        A loss scalar.
    """
    if device is not None:
        device = device
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if len(features.shape) < 3:
        raise ValueError('`features` needs to be [bsz, n_views, ...],'
                         'at least 3 dimensions are required')
    if len(features.shape) > 3:
        features = features.view(features.shape[0], features.shape[1], -1)

    batch_size = features.shape[0]
    if labels is not None and mask is not None:
        raise ValueError('Cannot define both `labels` and `mask`')
    elif labels is None and mask is None:
        mask = torch.eye(batch_size, dtype=torch.float32).to(device)  # 对角线全1的矩阵
    elif labels is not None:
        labels = labels.contiguous().view(-1, 1)
        if labels.shape[0] != batch_size:
            raise ValueError('Num of labels does not match num of features')
        mask = torch.eq(labels, labels.T).float().to(device)
    else:
        mask = mask.float().to(device)

    contrast_count = features.shape[1]  # 4 batch = 10
    contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)  # 40,50
    if contrast_mode == 'one':
        anchor_feature = features[:, 0]
        anchor_count = 1
    elif contrast_mode == 'all':
        anchor_feature = contrast_feature
        anchor_count = contrast_count
    else:
        raise ValueError('Unknown mode: {}'.format(contrast_mode))

    # compute logits
    anchor_dot_contrast = torch.div(
        torch.matmul(anchor_feature, contrast_feature.T),
        temperature)  # 40 40
    # for numerical stability
    logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)  # 40,1
    logits = anchor_dot_contrast - logits_max.detach()

    # tile mask
    mask = mask.repeat(anchor_count, contrast_count)  # 40, 40
    # mask-out self-contrast cases
    logits_mask = torch.scatter(
        torch.ones_like(mask),
        1,
        torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
        0
    )
    mask = mask * logits_mask

    # compute log_prob
    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

    # compute mean of log-likelihood over positive
    mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)  # 40

    # loss
    loss = - (temperature / base_temperature) * mean_log_prob_pos
    loss = loss.view(anchor_count, batch_size).mean()  # 4,10 -> mean

    return loss

def domain_adversarial_loss(domain_pred, true_domains, epoch, total_epochs, max_weight=0.3):
    """
    计算领域对抗损失，并动态调整对抗权重。
    :param domain_pred: 领域分类器的预测 (logits)
    :param true_domains: 真实的领域标签
    :param epoch: 当前训练轮数
    :param total_epochs: 总训练轮数
    :param max_weight: 领域对抗损失的最大权重
    :return: (对抗损失, 交叉熵损失)
    """
    # 计算交叉熵损失
    ce_loss = F.cross_entropy(domain_pred, true_domains)

    # 对抗损失（最大化分类误差）
    adv_loss = -ce_loss

    # 固定权重1.0，因为只在epoch=0使用
    return adv_loss, ce_loss


def domain_confusion_loss(features):
    domain_probs = F.softmax(features, dim=1)
    entropy = -torch.sum(domain_probs * torch.log(domain_probs + 1e-8), dim=1)
    return torch.mean(entropy)

def domain_adversarial_loss1(domain_pred, true_domains, domain_weight=0.5):
    """
    完整领域对抗损失计算
    :param domain_pred: 模型预测的domain概率 [B, num_domains]
    :param true_domains: 真实domain标签 [B]
    :param domain_weight: 对抗损失权重
    :return: (对抗损失, 领域分类损失)
    """
    # 领域分类交叉熵损失
    ce_loss = F.nll_loss(domain_pred, true_domains)

    # 对抗损失（最大化分类误差）
    adv_loss = -ce_loss

    return domain_weight * adv_loss, ce_loss

def soft_sup_con_loss(features, softlabels, hard_labels, temperature=0.07, base_temperature=0.07, device=None):
    """Compute loss for model. 
    Args:
        features: hidden vector of shape [bsz, hide_dim].
        soft_labels : hidden vector of shape [bsz, hide_dim].
        labels: ground truth of shape [bsz].
    Returns:
        A loss scalar.
    """
    if device is not None:
        device = device
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    features_dot_softlabels = torch.div(torch.matmul(features, softlabels.T), temperature)  #
    loss = torch.nn.functional.cross_entropy(features_dot_softlabels, hard_labels)

    return loss


def soft_sup_con_loss_patch(features, softlabels, hard_labels, temperature=0.07, device=None):
    """
    features: [B, N, D] (局部特征)
    softlabels: [B, D] (全局伪标签)
    hard_labels: [B] (真实标签)
    """
    # 计算所有 patches 和所有 softlabels 的相似度
    global_features = features.max(dim=1)[0]  # [B, D]

    # 计算对比损失
    logits = torch.matmul(global_features, softlabels.T) / temperature  # [B, B]
    loss = F.cross_entropy(logits, hard_labels)
    return loss

def cross_modality_local_loss(local_rgb, local_phase, temperature=0.07, device=None):
    """
    跨模态局部特征对比损失（对称版）
    local_rgb   : [B, N, D]
    local_phase : [B, N, D]
    """
    device = device or local_rgb.device
    B, N, D = local_rgb.shape

    rgb_norm = F.normalize(local_rgb, p=2, dim=-1)
    phase_norm = F.normalize(local_phase, p=2, dim=-1)

    # RGB -> Phase
    logits_ab = torch.einsum('bnd,bmd->bnm', rgb_norm, phase_norm) / temperature
    # Phase -> RGB
    logits_ba = torch.einsum('bnd,bmd->bnm', phase_norm, rgb_norm) / temperature

    mask = torch.eye(N, dtype=torch.bool, device=device).unsqueeze(0).expand(B, -1, -1)

    loss_ab = -F.log_softmax(logits_ab, dim=-1)[mask].mean()
    loss_ba = -F.log_softmax(logits_ba, dim=-1)[mask].mean()

    return (loss_ab + loss_ba) / 2

import torch
import torch.nn as nn

class GroupedLoss(nn.Module):
    def __init__(self, device):
        super(GroupedLoss, self).__init__()

        # 主任务：两个损失，两个 log_var
        self.log_var_feature = nn.Parameter(torch.zeros(1, device=device, requires_grad=True))
        self.log_var_phase_feature = nn.Parameter(torch.zeros(1, device=device, requires_grad=True))

        # 辅助任务：可以合理合并为 3 组
        self.log_var_patch = nn.Parameter(torch.zeros(1, device=device, requires_grad=True))  # patch 相关
        self.log_var_triplet = nn.Parameter(torch.zeros(1, device=device, requires_grad=True))  # 所有 triplet
        self.log_var_local = nn.Parameter(torch.zeros(1, device=device, requires_grad=True))  # local 特征

    def forward(self,
                loss_feature,
                loss_phase_feature,
                loss_feature_patch,
                loss_phase_feature_patch,
                loss_triplet_rgb,
                loss_triplet_phase,
                loss_triplet_phase_rgb,
                loss_triplet_rgb_phase,
                loss_feature_local):

        # 主任务损失
        loss_main1 = loss_feature
        loss_main2 = loss_phase_feature

        # 辅助任务损失组合
        loss_patch = loss_feature_patch + loss_phase_feature_patch
        loss_triplet = loss_triplet_rgb + loss_triplet_phase + loss_triplet_phase_rgb + loss_triplet_rgb_phase
        loss_local = loss_feature_local

        # 不确定性加权总损失
        loss = (
            self._weighted_loss(loss_main1, self.log_var_feature) +
            self._weighted_loss(loss_main2, self.log_var_phase_feature) +
            self._weighted_loss(loss_patch, self.log_var_patch) +
            self._weighted_loss(loss_triplet, self.log_var_triplet) +
            self._weighted_loss(loss_local, self.log_var_local)
        )

        return loss

    def _weighted_loss(self, loss, log_var):
        # 不确定性加权核心公式
        precision = torch.exp(-log_var)
        return precision * loss + log_var



def Euclidean_MSE(semantic_emb):
    def mse_loss(enc_out, cls_id):
        gt_cls_sim = torch.cdist(semantic_emb[cls_id], semantic_emb, p=2.0)
        eucdist_logits = torch.cdist(enc_out, semantic_emb, p=2.0)
        cls_euc_scaled = gt_cls_sim / (torch.max(gt_cls_sim, dim=1).values.unsqueeze(-1))
        cls_wts = torch.exp(-1.5 * cls_euc_scaled)
        emb_mse = nn.MSELoss(reduction='none')(eucdist_logits, gt_cls_sim)
        emb_mse *= cls_wts
        return emb_mse.mean()

    return mse_loss

import torch
import torch.nn.functional as F

def triplet_loss_rgb_local(local_image_features, queues, hard_labels, margin=0.2, device=None):
    """
    同模态 RGB 局部特征三元组损失

    参数：
    local_image_features : Tensor - 当前 batch 的 RGB 局部特征 [batch_size, feat_dim]
    queues               : dict   - 历史 RGB 局部特征队列 {class_id: deque([Tensor, ...])}
    hard_labels          : Tensor - 当前 batch 的标签 [batch_size]
    margin               : float  - 三元组间隔
    device               : torch.device - 设备

    返回：
    loss : Tensor - 平均损失
    """
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    losses = []

    for anchor, label in zip(local_image_features, hard_labels):
        cls_label = label.item()

        # ------------------- 获取正样本 -------------------
        positives = list(queues.get(cls_label, []))
        negatives = []
        for cls_id, queue in queues.items():
            if cls_id != cls_label:
                negatives.extend(queue)

        if len(positives) == 0 or len(negatives) == 0:
            continue

        # ------------------- 随机采样 -------------------
        pos_idx = torch.randint(0, len(positives), (1,)).item()
        neg_idx1 = torch.randint(0, len(negatives), (1,)).item()
        neg_idx2 = torch.randint(0, len(negatives), (1,)).item()

        anchor = anchor.unsqueeze(0).to(device)
        positive = positives[pos_idx].unsqueeze(0).to(device)
        negative1 = negatives[neg_idx1].unsqueeze(0).to(device)
        negative2 = negatives[neg_idx2].unsqueeze(0).to(device)

        # ------------------- 三元组损失 -------------------
        pos_dist = F.pairwise_distance(anchor, positive)
        neg_dist1 = F.pairwise_distance(anchor, negative1)
        neg_dist2 = F.pairwise_distance(anchor, negative2)

        loss1 = F.relu(pos_dist - neg_dist1 + margin)
        loss2 = F.relu(pos_dist - neg_dist2 + margin)
        avg_loss = (loss1 + loss2) / 2
        losses.append(avg_loss)

    return torch.mean(torch.stack(losses)) if losses else torch.tensor(0.0).to(device)


def triplet_loss_phase_local(local_image_features, queues, hard_labels, margin=0.2, device=None):
    """
    同模态 RGB 局部特征三元组损失

    参数：
    local_image_features : Tensor - 当前 batch 的 RGB 局部特征 [batch_size, feat_dim]
    queues               : dict   - 历史 RGB 局部特征队列 {class_id: deque([Tensor, ...])}
    hard_labels          : Tensor - 当前 batch 的标签 [batch_size]
    margin               : float  - 三元组间隔
    device               : torch.device - 设备

    返回：
    loss : Tensor - 平均损失
    """
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    losses = []

    for anchor, label in zip(local_image_features, hard_labels):
        cls_label = label.item()

        # ------------------- 获取正样本 -------------------
        positives = list(queues.get(cls_label, []))
        negatives = []
        for cls_id, queue in queues.items():
            if cls_id != cls_label:
                negatives.extend(queue)

        if len(positives) == 0 or len(negatives) == 0:
            continue

        # ------------------- 随机采样 -------------------
        pos_idx = torch.randint(0, len(positives), (1,)).item()
        neg_idx1 = torch.randint(0, len(negatives), (1,)).item()
        neg_idx2 = torch.randint(0, len(negatives), (1,)).item()

        anchor = anchor.unsqueeze(0).to(device)
        positive = positives[pos_idx].unsqueeze(0).to(device)
        negative1 = negatives[neg_idx1].unsqueeze(0).to(device)
        negative2 = negatives[neg_idx2].unsqueeze(0).to(device)

        # ------------------- 三元组损失 -------------------
        pos_dist = F.pairwise_distance(anchor, positive)
        neg_dist1 = F.pairwise_distance(anchor, negative1)
        neg_dist2 = F.pairwise_distance(anchor, negative2)

        loss1 = F.relu(pos_dist - neg_dist1 + margin)
        loss2 = F.relu(pos_dist - neg_dist2 + margin)
        avg_loss = (loss1 + loss2) / 2
        losses.append(avg_loss)

    return torch.mean(torch.stack(losses)) if losses else torch.tensor(0.0).to(device)



def triplet_loss_random(features, queues, hard_labels, margin=0.2, device=None):
    """
    Compute the triplet loss.
    Parameters:
    - anchor: the feature vector of the current sample, shape (D,)
    - queue: the feature vectors of samples in the queue, shape (N, D)
    - labels: the labels of samples in the queue, shape (N,)
    - margin: the margin for triplet loss
    Returns:
    - loss: triplet loss
    """
    # pdb.set_trace()
    if device is not None:
        device = device
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    losses = []
    for i, (image, image_class) in enumerate(zip(features, hard_labels)):
        positive_samples = []
        negative_samples = []
        for cls, queue in queues.items():
            if cls == image_class:
                positive_samples.extend(queue)
            else:
                negative_samples.extend(queue)
        if len(positive_samples) == 0 or len(negative_samples) == 0:
            continue

        random_index_p1 = torch.randint(0, len(positive_samples), (1,)).item()
        random_index_n1 = torch.randint(0, len(negative_samples), (1,)).item()
        random_index_p2 = torch.randint(0, len(positive_samples), (1,)).item()
        random_index_n2 = torch.randint(0, len(negative_samples), (1,)).item()

        positive_samples = torch.stack(positive_samples).to(device)

        random_positive_sample1 = positive_samples[random_index_p1]
        random_positive_sample2 = positive_samples[random_index_p2]

        negative_samples = torch.stack(negative_samples).to(device)

        random_negative_sample1 = negative_samples[random_index_n1]
        random_negative_sample2 = negative_samples[random_index_n2]

        positive_distances1 = torch.cdist(image.unsqueeze(0), random_positive_sample1.unsqueeze(0), p=2.0)
        positive_distances2 = torch.cdist(image.unsqueeze(0), random_positive_sample2.unsqueeze(0), p=2.0)

        # hardest_positive_distance = positive_distances.max()
        negative_distances1 = torch.cdist(image.unsqueeze(0), random_negative_sample1.unsqueeze(0), p=2.0)
        negative_distances2 = torch.cdist(image.unsqueeze(0), random_negative_sample2.unsqueeze(0), p=2.0)

        # hardest_negative_distance = negative_distances.min()
        loss1 = F.relu(positive_distances1 - negative_distances1 + margin)
        loss2 = F.relu(positive_distances2 - negative_distances2 + margin)

        loss = 0.5 * loss1 + 0.5 * loss2

        losses.append(loss)
        # Average the losses for the batch
    return torch.stack(losses).mean()


def triplet_loss_phase(phase_features, ph_queues, hard_labels, margin=0.2, device=None):
    """
    相位特征三元组损失（同模态）

    参数：
    phase_features : Tensor - 当前batch的相位特征 [batch_size, feat_dim]
    ph_queues      : dict   - 历史相位特征队列 {class_id: deque(maxlen=N)}
    hard_labels    : Tensor - 当前batch的标签 [batch_size]
    margin         : float  - 三元组损失间隔

    返回：
    loss : Tensor - 平均损失值
    """
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    losses = []

    # 遍历每个相位特征作为锚点
    for anchor, label in zip(phase_features, hard_labels):
        cls_label = label.item()

        # ------------------------- 正负样本准备 -------------------------
        # 正样本：同类别历史相位特征（同模态）
        positives = list(ph_queues.get(cls_label, []))  # 空列表安全

        # 负样本：其他类别的所有历史相位特征
        negatives = []
        for cls_id, queue in ph_queues.items():
            if cls_id != cls_label:
                negatives.extend(list(queue))

        # 跳过无效样本情况
        if len(positives) == 0 or len(negatives) == 0:
            continue

        # ------------------------- 采样逻辑 -------------------------
        # 随机选择1个正样本和2个负样本
        pos_idx = torch.randint(0, len(positives), (1,)).item()
        neg_idx1 = torch.randint(0, len(negatives), (1,)).item()
        neg_idx2 = torch.randint(0, len(negatives), (1,)).item()

        # 转换为张量并移动到设备
        anchor = anchor.unsqueeze(0).to(device)  # [1, feat_dim]
        positive = positives[pos_idx].unsqueeze(0).to(device)  # [1, feat_dim]
        negative1 = negatives[neg_idx1].unsqueeze(0).to(device)
        negative2 = negatives[neg_idx2].unsqueeze(0).to(device)

        # ------------------------- 距离计算 -------------------------
        pos_dist = F.pairwise_distance(anchor, positive)  # 标量
        neg_dist1 = F.pairwise_distance(anchor, negative1)
        neg_dist2 = F.pairwise_distance(anchor, negative2)

        # ------------------------- 损失计算 -------------------------
        loss1 = F.relu(pos_dist - neg_dist1 + margin)
        loss2 = F.relu(pos_dist - neg_dist2 + margin)
        avg_loss = (loss1 + loss2) / 2

        losses.append(avg_loss)

    # 返回平均损失（若无有效样本则返回0）
    return torch.mean(torch.stack(losses)) if losses else torch.tensor(0.0).to(device)

def triplet_loss_rgb(phase_features, ph_queues, hard_labels, margin=0.2, device=None):
    """
    相位特征三元组损失（同模态）

    参数：
    phase_features : Tensor - 当前batch的相位特征 [batch_size, feat_dim]
    ph_queues      : dict   - 历史相位特征队列 {class_id: deque(maxlen=N)}
    hard_labels    : Tensor - 当前batch的标签 [batch_size]
    margin         : float  - 三元组损失间隔

    返回：
    loss : Tensor - 平均损失值
    """
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    losses = []

    # 遍历每个相位特征作为锚点
    for anchor, label in zip(phase_features, hard_labels):
        cls_label = label.item()

        # ------------------------- 正负样本准备 -------------------------
        # 正样本：同类别历史相位特征（同模态）
        positives = list(ph_queues.get(cls_label, []))  # 空列表安全

        # 负样本：其他类别的所有历史相位特征
        negatives = []
        for cls_id, queue in ph_queues.items():
            if cls_id != cls_label:
                negatives.extend(list(queue))

        # 跳过无效样本情况
        if len(positives) == 0 or len(negatives) == 0:
            continue

        # ------------------------- 采样逻辑 -------------------------
        # 随机选择1个正样本和2个负样本
        pos_idx = torch.randint(0, len(positives), (1,)).item()
        neg_idx1 = torch.randint(0, len(negatives), (1,)).item()
        neg_idx2 = torch.randint(0, len(negatives), (1,)).item()

        # 转换为张量并移动到设备
        anchor = anchor.unsqueeze(0).to(device)  # [1, feat_dim]
        positive = positives[pos_idx].unsqueeze(0).to(device)  # [1, feat_dim]
        negative1 = negatives[neg_idx1].unsqueeze(0).to(device)
        negative2 = negatives[neg_idx2].unsqueeze(0).to(device)

        # ------------------------- 距离计算 -------------------------
        pos_dist = F.pairwise_distance(anchor, positive)  # 标量
        neg_dist1 = F.pairwise_distance(anchor, negative1)
        neg_dist2 = F.pairwise_distance(anchor, negative2)

        # ------------------------- 损失计算 -------------------------
        loss1 = F.relu(pos_dist - neg_dist1 + margin)
        loss2 = F.relu(pos_dist - neg_dist2 + margin)
        avg_loss = (loss1 + loss2) / 2

        losses.append(avg_loss)

    # 返回平均损失（若无有效样本则返回0）
    return torch.mean(torch.stack(losses)) if losses else torch.tensor(0.0).to(device)

def infonce_loss_rgb(phase_features, ph_queues, hard_labels, temperature=0.07, device=None):
    """
    相位特征 InfoNCE 损失（同模态）

    参数：
    phase_features : Tensor - 当前 batch 的相位特征 [batch_size, feat_dim]
    ph_queues      : dict   - 历史相位特征队列 {class_id: deque(maxlen=N)}
    hard_labels    : Tensor - 当前 batch 的标签 [batch_size]
    temperature    : float  - 温度参数，用于缩放 logits
    device         : torch.device - 设备

    返回：
    loss : Tensor - 平均损失值
    """
    device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    losses = []

    # 遍历每个相位特征作为 anchor
    for anchor, label in zip(phase_features, hard_labels):
        cls_label = label.item()

        # ------------------------- 正负样本准备 -------------------------
        # 正样本：同类别历史相位特征（同模态）
        positives = list(ph_queues.get(cls_label, []))

        # 负样本：其他类别的所有历史相位特征
        negatives = []
        for cls_id, queue in ph_queues.items():
            if cls_id != cls_label:
                negatives.extend(list(queue))

        # 跳过无效样本情况
        if len(positives) == 0 or len(negatives) == 0:
            continue

        # ------------------------- 采样逻辑 -------------------------
        # 随机选择 1 个正样本
        pos_idx = torch.randint(0, len(positives), (1,)).item()
        positive = positives[pos_idx]

        # 将负样本堆叠成 tensor
        negatives_tensor = torch.stack(negatives)  # [num_negatives, feat_dim]

        # ------------------------- 特征归一化 -------------------------
        # 注意：通常 InfoNCE 损失中使用归一化后的特征计算余弦相似度
        anchor_norm = F.normalize(anchor.unsqueeze(0).to(device), p=2, dim=1)  # [1, feat_dim]
        positive_norm = F.normalize(positive.unsqueeze(0).to(device), p=2, dim=1)  # [1, feat_dim]
        negatives_norm = F.normalize(negatives_tensor.to(device), p=2, dim=1)  # [num_negatives, feat_dim]

        # ------------------------- 相似度计算 -------------------------
        # 计算 anchor 与正样本之间的相似度 [1,1]
        pos_sim = torch.matmul(anchor_norm, positive_norm.t())
        # 计算 anchor 与所有负样本之间的相似度 [1, num_negatives]
        neg_sim = torch.matmul(anchor_norm, negatives_norm.t())

        # 构造 logits, 正样本放在第一位
        logits = torch.cat([pos_sim, neg_sim], dim=1)  # [1, 1+num_negatives]
        logits /= temperature

        # 构造目标，正确类别为第 0 个位置
        target = torch.zeros(logits.size(0), dtype=torch.long).to(device)

        # ------------------------- 损失计算 -------------------------
        loss = F.cross_entropy(logits, target)
        losses.append(loss)

    # 若无有效样本则返回 0
    return torch.mean(torch.stack(losses)) if losses else torch.tensor(0.0).to(device)


def triplet_loss_phase_rgb(phase_features, queues, ph_queues, hard_labels, margin=0.2, device=None):
    """
    跨模态三元组损失（相位锚点 vs RGB正样本 vs 相位负样本）

    参数：
    phase_features : Tensor - 当前批次相位特征 [batch_size, feat_dim]
    queues         : dict   - 历史RGB特征队列 {class_id: deque}
    ph_queues      : dict   - 历史相位特征队列 {class_id: deque}
    hard_labels    : Tensor - 当前批次标签 [batch_size]
    margin         : float  - 间隔值

    返回：
    loss : Tensor - 平均损失值
    """
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    losses = []

    # 遍历每个样本作为锚点
    for phase_anchor, label in zip(phase_features, hard_labels):
        cls_label = label.item()

        # ===================== 正样本：同类别RGB特征 =====================
        rgb_positives = list(queues.get(cls_label, []))
        if not rgb_positives:  # 空队列检查
            continue

        # ===================== 负样本：其他类别相位特征 =====================
        phase_negatives = []
        for cls_id, queue in ph_queues.items():
            if cls_id != cls_label:
                phase_negatives.extend(list(queue))
        if not phase_negatives:  # 空队列检查
            continue

        # ===================== 随机采样 =====================
        # 随机选择正样本索引
        pos_idx = torch.randint(0, len(rgb_positives), (1,)).item()

        # 随机选择两个负样本索引
        neg_idx1 = torch.randint(0, len(phase_negatives), (1,)).item()
        neg_idx2 = torch.randint(0, len(phase_negatives), (1,)).item()

        # ===================== 张量维度对齐 =====================
        # 锚点相位特征 [feat_dim] -> [1, feat_dim]
        anchor = phase_anchor.unsqueeze(0).to(device)  # [1, D]

        # 正样本RGB特征 [D] -> [1, D]
        pos_rgb = rgb_positives[pos_idx].unsqueeze(0).to(device)

        # 负样本相位特征 [D] -> [1, D]
        neg_phase1 = phase_negatives[neg_idx1].unsqueeze(0).to(device)
        neg_phase2 = phase_negatives[neg_idx2].unsqueeze(0).to(device)


        # ===================== 距离计算 =====================
        # 锚点-正样本距离（跨模态）
        pos_dist = F.pairwise_distance(anchor, pos_rgb)

        # 锚点-负样本距离（同模态）
        neg_dist1 = F.pairwise_distance(anchor, neg_phase1)
        neg_dist2 = F.pairwise_distance(anchor, neg_phase2)

        # ===================== 损失计算 =====================
        loss1 = F.relu(pos_dist - neg_dist1 + margin)
        loss2 = F.relu(pos_dist - neg_dist2 + margin)
        avg_loss = (loss1 + loss2) / 2

        losses.append(avg_loss)

    # 返回平均损失（若无有效样本则返回0）
    return torch.mean(torch.stack(losses)) if losses else torch.tensor(0.0).to(device)


def triplet_loss_rgb_phase(rgb_features, queues, ph_queues, hard_labels, margin=0.2, device=None):
    """
    RGB锚点-相位正样本-RGB负样本三元组损失

    参数：
    rgb_features : Tensor - 当前batch的RGB特征 [batch_size, feat_dim]
    queues       : dict   - 历史RGB特征队列 {class_id: deque}
    ph_queues    : dict   - 历史相位特征队列 {class_id: deque}
    hard_labels  : Tensor - 当前batch的标签 [batch_size]
    margin       : float  - 间隔值

    返回：
    loss : Tensor - 平均损失值
    """
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    losses = []

    # 遍历每个RGB特征作为锚点
    for rgb_anchor, label in zip(rgb_features, hard_labels):
        cls_label = label.item()

        # ===================== 正样本：同类别相位特征 =====================
        ph_positives = list(ph_queues.get(cls_label, []))
        if not ph_positives:  # 空队列检查
            continue

        # ===================== 负样本：其他类别RGB特征 =====================
        rgb_negatives = []
        for cls_id, queue in queues.items():
            if cls_id != cls_label:
                rgb_negatives.extend(list(queue))
        if not rgb_negatives:  # 空队列检查
            continue

        # ===================== 随机采样 =====================
        # 正样本（相位）
        pos_idx = torch.randint(0, len(ph_positives), (1,)).item()

        # 负样本（RGB）
        neg_idx1 = torch.randint(0, len(rgb_negatives), (1,)).item()
        neg_idx2 = torch.randint(0, len(rgb_negatives), (1,)).item()

        # ===================== 张量维度对齐 =====================
        anchor = rgb_anchor.unsqueeze(0).to(device)  # [1, D]
        pos_ph = ph_positives[pos_idx].unsqueeze(0).to(device)  # [1, D]
        neg_rgb1 = rgb_negatives[neg_idx1].unsqueeze(0).to(device)
        neg_rgb2 = rgb_negatives[neg_idx2].unsqueeze(0).to(device)

        # ===================== 距离计算 =====================
        pos_dist = F.pairwise_distance(anchor, pos_ph)  # 跨模态距离
        neg_dist1 = F.pairwise_distance(anchor, neg_rgb1)  # 同模态距离
        neg_dist2 = F.pairwise_distance(anchor, neg_rgb2)

        # ===================== 损失计算 =====================
        loss1 = F.relu(pos_dist - neg_dist1 + margin)
        loss2 = F.relu(pos_dist - neg_dist2 + margin)
        avg_loss = (loss1 + loss2) / 2

        losses.append(avg_loss)

    return torch.mean(torch.stack(losses)) if losses else torch.tensor(0.0).to(device)


import torch
import torch.nn.functional as F

def cross_phase_centroid_loss(feature,
                              phase_feature,
                              cls_id,
                              phase_centroids,
                              temperature=0.07,
                              device=None):
    """
    计算跨相位类别质心损失。

    :param feature: RGB 特征张量，形状 [B, D]。
    :param phase_feature: 相位特征张量，形状 [B, D]。
    :param cls_id: 真实类别标签，形状 [B]。
    :param phase_centroids: 质心字典 {cls_id: Tensor}，每个类别的质心维度 [D]。
    :param temperature: 温度参数，默认 0.07。
    :param device: 计算设备，'cuda' 或 'cpu'。
    :return: 交叉熵损失值。
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # **确保输入为 Tensor**
    feature = torch.as_tensor(feature, dtype=torch.float32, device=device)
    phase_feature = torch.as_tensor(phase_feature, dtype=torch.float32, device=device)
    cls_id = torch.as_tensor(cls_id, dtype=torch.long, device=device)

    # **检查 phase_centroids 是否为空**
    if not phase_centroids:
        print("WARNING: phase_centroids 为空，返回默认损失 0")
        return torch.tensor(0.0, device=device, requires_grad=True)

    # **构造类别质心张量**
    all_class_ids = sorted(phase_centroids.keys())  # 确保类别顺序一致
    centroid_list = []
    centroid_cls_map = {}  # 记录类别索引映射

    for idx, cid in enumerate(all_class_ids):
        if phase_centroids[cid] is not None:
            c = torch.as_tensor(phase_centroids[cid], dtype=torch.float32, device=device).view(1, -1)
            centroid_list.append(c)
            centroid_cls_map[cid] = idx  # 记录类别对应的索引

    # **如果没有有效质心，返回默认损失**
    if not centroid_list:
        print("WARNING: 没有找到有效的类别质心，返回默认损失 0")
        return torch.tensor(0.0, device=device, requires_grad=True)

    # **拼接所有类别质心**
    centroids = torch.cat(centroid_list, dim=0)  # [K, D]，K 为类别数

    # **映射 cls_id 到质心索引**
    valid_mask = torch.tensor([cid in centroid_cls_map for cid in cls_id], dtype=torch.bool, device=device)
    if not valid_mask.any():
        print("WARNING: cls_id 全部无效，返回默认损失 0")
        return torch.tensor(0.0, device=device, requires_grad=True)

    cls_id_mapped = torch.tensor([centroid_cls_map[cid] for cid in cls_id[valid_mask]], dtype=torch.long, device=device)

    # **计算 logits**
    logits_rgb = torch.matmul(feature[valid_mask], centroids.t()) / temperature  # [B, K]
    logits_phase = torch.matmul(phase_feature[valid_mask], centroids.t()) / temperature  # [B, K]

    # **计算交叉熵损失**
    loss_rgb = F.cross_entropy(logits_rgb, cls_id_mapped)
    loss_phase = F.cross_entropy(logits_phase, cls_id_mapped)

    # **最终损失**
    loss = 0.5 * (loss_rgb + loss_phase)

    return loss

def rgb_centroid_loss(phase_feature, cls_id, phase_centroids, temperature=0.07, device=None):
    """
    计算基于类别质心的交叉相位损失。

    :param phase_feature: 输入的相位特征，形状为 [B, D]。
    :param cls_id: 类别 ID，形状为 [B]。
    :param phase_centroids: 每个类别的质心，字典形式 {cls_id: Tensor}，形状为 (D,)。
    :param temperature: 温度超参数，用于调整 logits 的尺度。
    :param device: 计算设备（'cuda' 或 'cpu'）。
    :return: 计算得到的损失值。
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # **确保输入为 PyTorch Tensor**
    phase_feature = torch.as_tensor(phase_feature, dtype=torch.float32, device=device)
    cls_id = torch.as_tensor(cls_id, dtype=torch.long, device=device)

    # **检查 phase_centroids 是否为空**
    if not phase_centroids:
        raise ValueError("phase_centroids 为空，无法计算损失。")

    # **构造类别质心**
    all_class_ids = sorted(phase_centroids.keys())  # 确保类别按顺序排列
    centroid_list, centroid_cls_map = [], {}

    for idx, cid in enumerate(all_class_ids):
        if phase_centroids[cid] is not None:
            c = phase_centroids[cid].view(1, -1)  # 使其形状为 [1, D]
            centroid_list.append(c)
            centroid_cls_map[cid] = idx

    if not centroid_list:
        raise ValueError("没有找到有效的类别质心，检查 phase_centroids 数据。")

    # **计算质心矩阵**
    centroids = torch.cat(centroid_list, dim=0)  # [K, D]

    # **计算 logits**
    logits_phase = torch.matmul(phase_feature, centroids.t()) / temperature  # [B, K]

    # **计算交叉熵损失**
    loss_phase = F.cross_entropy(logits_phase, cls_id)

    return loss_phase

def phase_centroid_loss(phase_feature, cls_id, phase_centroids, temperature=0.07, device=None):
    """
    计算基于类别质心的交叉相位损失。

    :param phase_feature: 输入的相位特征，形状为 [B, D]。
    :param cls_id: 类别 ID，形状为 [B]。
    :param phase_centroids: 每个类别的质心，字典形式 {cls_id: Tensor}，形状为 (D,)。
    :param temperature: 温度超参数，用于调整 logits 的尺度。
    :param device: 计算设备（'cuda' 或 'cpu'）。
    :return: 计算得到的损失值。
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # **确保输入为 PyTorch Tensor**
    phase_feature = torch.as_tensor(phase_feature, dtype=torch.float32, device=device)
    cls_id = torch.as_tensor(cls_id, dtype=torch.long, device=device)

    # **检查 phase_centroids 是否为空**
    if not phase_centroids:
        raise ValueError("phase_centroids 为空，无法计算损失。")

    # **构造类别质心**
    all_class_ids = sorted(phase_centroids.keys())  # 确保类别按顺序排列
    centroid_list, centroid_cls_map = [], {}

    for idx, cid in enumerate(all_class_ids):
        if phase_centroids[cid] is not None:
            c = phase_centroids[cid].view(1, -1)  # 使其形状为 [1, D]
            centroid_list.append(c)
            centroid_cls_map[cid] = idx

    if not centroid_list:
        raise ValueError("没有找到有效的类别质心，检查 phase_centroids 数据。")

    # **计算质心矩阵**
    centroids = torch.cat(centroid_list, dim=0)  # [K, D]

    # **计算 logits**
    logits_phase = torch.matmul(phase_feature, centroids.t()) / temperature  # [B, K]

    # **计算交叉熵损失**
    loss_phase = F.cross_entropy(logits_phase, cls_id)

    return loss_phase

def center_loss(phase_feature, cls_id, ph_queues, device=None):
    """
    计算中心损失（Center Loss），仅基于相位特征和历史相位特征队列

    参数：
      phase_feature : Tensor [batch_size, feat_dim] - 当前batch的相位特征
      cls_id        : Tensor [batch_size] - 当前batch的类别标签（整数）
      ph_queues     : dict {cls_id: deque(maxlen=N)} - 历史相位特征队列
      device        : torch.device, 如果未指定则自动选择

    返回：
      loss          : Tensor - 标量中心损失
    """
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    losses = []

    for pf, cid in zip(phase_feature, cls_id):
        c = cid.item()
        # 如果历史队列中有数据，则计算该类别历史特征的均值作为中心，否则用当前样本自身作为中心
        if c in ph_queues and len(ph_queues[c]) > 0:
            historical = torch.stack(list(ph_queues[c])).to(device)  # shape: [n, feat_dim]
            center = historical.mean(dim=0)
        else:
            center = pf.to(device)

        # 计算当前样本与中心的欧氏距离平方
        diff = pf.to(device) - center
        loss_sample = torch.sum(diff ** 2)
        losses.append(loss_sample)

    return torch.mean(torch.stack(losses)) if losses else torch.tensor(0.0).to(device)

def triplet_loss_hard_sample(features, queues, hard_labels, margin=0.2, device=None):
    """
    Compute the triplet loss.
    Parameters:
    - anchor: the feature vector of the current sample, shape (D,)
    - queue: the feature vectors of samples in the queue, shape (N, D)
    - labels: the labels of samples in the queue, shape (N,)
    - margin: the margin for triplet loss
    Returns:
    - loss: triplet loss
    """
    # pdb.set_trace()
    if device is not None:
        device = device
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    losses = []
    for i, (image, image_class) in enumerate(zip(features, hard_labels)):
        positive_samples = []
        negative_samples = []
        for cls, queue in queues.items():
            # print(f"cls:{cls}")
            # print(f"image_class:{image_class}")
            if cls == image_class:
                positive_samples.extend(queue)
            else:
                negative_samples.extend(queue)
        if len(positive_samples) == 0 or len(negative_samples) == 0:
            continue
        positive_samples = torch.stack(positive_samples).to(device)
        # random_index_p = torch.randint(0, len(positive_samples), (1,)).item()
        # random_positive_sample = positive_samples[random_index_p]
        negative_samples = torch.stack(negative_samples).to(device)
        # random_index_n = torch.randint(0, len(negative_samples), (1,)).item()
        # random_negative_sample = negative_samples[random_index_n]
        positive_distances = torch.cdist(image.unsqueeze(0), positive_samples.unsqueeze(0), p=2.0)
        # hardest_positive_distance = positive_distances.max()
        negative_distances = torch.cdist(image.unsqueeze(0), negative_samples.unsqueeze(0), p=2.0)
        hardest_positive_distance = positive_distances.max()
        hardest_negative_distance = negative_distances.min()
        # hardest_negative_distance = negative_distances.min()
        loss = F.relu(hardest_positive_distance - hardest_negative_distance + margin)

        losses.append(loss)
        # Average the losses for the batch
    return torch.stack(losses).mean()


def itcs_m(features, softlabels, hard_labels, dom_labels, queues, temperature=0.07, base_temperature=0.07, device=None):
    """Compute loss for model.

    Args:

        features: hidden vector of shape [bsz, hide_dim].

        soft_labels : hidden vector of shape [bsz, hide_dim].

        labels: ground truth of shape [bsz].

    Returns:

        A loss scalar.

    """
    losses = []
    sca = 0.1
    text_samples = softlabels
    if device is not None:
        device = device
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    total_similarities = {}
    for i, (image, image_class, image_domain) in enumerate(zip(features, hard_labels, dom_labels)):
        positive_samples = []
        negative_samples = []
        for cls, domain_queues in queues.items():
            if cls == image_class:
                positive_samples.extend(domain_queues[image_domain])
            else:
                negative_samples.extend(domain_queues[image_domain])
        if len(positive_samples) == 0 or len(negative_samples) == 0:
            continue
        positive_samples = torch.stack(positive_samples).to(device)
        negative_samples = torch.stack(negative_samples).to(device)
        text_sim = torch.div(torch.matmul(image.unsqueeze(0), text_samples.T), temperature)
        negative_similarities = torch.div(torch.matmul(image.unsqueeze(0), negative_samples.T), temperature)
        all_similarities = torch.cat([text_sim, negative_similarities], dim=1)
        all_similarities = all_similarities.squeeze(0)
        loss = nn.functional.cross_entropy(all_similarities, image_class)
        losses.append(loss)
    return torch.mean(torch.stack(losses))