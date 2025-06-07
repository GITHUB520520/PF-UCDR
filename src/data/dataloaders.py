import os
import numpy as np
import random
from PIL import Image, ImageOps

import torch
import torch.utils.data as data
import torchvision

from scipy.spatial.distance import cdist
from torchvision import transforms as T

#
# from src.data import _BASE_PATH
_BASE_PATH = ''


class BaselineDataset(data.Dataset):
    def __init__(self, fls, transforms=None):

        self.fls = fls
        self.clss = np.array([f.split('/')[-2] for f in fls])
        self.domains = np.array([f.split('/')[-3] for f in fls])
        self.transforms = transforms
        self.to_tensor = T.ToTensor()

    def phase_img_getter(self, img):
        """生成相位图像（适配通道优先）"""
        # 输入处理
        if isinstance(img, torch.Tensor):
            img_np = img.permute(1, 2, 0).numpy()  # 从 (C, H, W) 转为 (H, W, C)
        else:
            img_np = np.array(img)  # PIL.Image的shape是 (H, W, C)

        # 计算FFT相位
        img_fft = np.fft.fft2(img_np, axes=(0, 1))  # 在空间维度计算FFT
        img_pha = np.angle(img_fft)  # 形状 (H, W, C)

        # 调整img_pha为通道优先 (C, H, W)
        img_pha = np.transpose(img_pha, (2, 0, 1))  # 新形状 (3, 224, 224)

        # 缩放因子形状 (3, 1, 1) [C, H, W]
        scale_factor = np.array([50000, 50000, 50000]).reshape(3, 1, 1)

        # 计算相位图像
        phase_img = scale_factor * (np.e ** (1j * img_pha))  # 广播成功
        phase_img = np.real(np.fft.ifft2(phase_img, axes=(1, 2)))  # 在H,W维度反变换

        # 调整回通道最后并转换为PIL图像
        phase_img = np.transpose(phase_img, (1, 2, 0))  # (H, W, C)
        phase_img = np.clip(phase_img, 0, 255).astype(np.uint8)
        phase_img = Image.fromarray(phase_img)

        # 转换为张量 (C, H, W)
        phase_img = self.to_tensor(phase_img)
        return phase_img

    def __getitem__(self, item):

        sample_domain = self.domains[item]
        if sample_domain == 'sketch' or sample_domain == 'quickdraw':
            sample = ImageOps.invert(Image.open(self.fls[item])).convert(mode='RGB')
        else:
            sample = Image.open(self.fls[item]).convert(mode='RGB')

        clss = self.clss[item]

        if self.transforms is not None:
            # 应用transforms后，sample会是PyTorch Tensor (C, H, W)
            sample = self.transforms(sample)

        # 生成相位图像
        # phase_img = self.phase_img_getter(sample)

        # print(phase_img.shape)

        return sample, clss, sample_domain

    def __len__(self):
        return len(self.fls)


class BaselineEvalDataset(data.Dataset):
    def __init__(self, fls, transforms=None):

        self.fls = fls
        self.clss = np.array([f.split('/')[-2] for f in fls])
        self.domains = np.array([f.split('/')[-3] for f in fls])
        self.transforms = transforms
        self.to_tensor = T.ToTensor()

    def phase_img_getter(self, img):
        """生成相位图像（适配通道优先）"""
        # 输入处理
        if isinstance(img, torch.Tensor):
            img_np = img.permute(1, 2, 0).numpy()  # 从 (C, H, W) 转为 (H, W, C)
        else:
            img_np = np.array(img)  # PIL.Image的shape是 (H, W, C)

        # 计算FFT相位
        img_fft = np.fft.fft2(img_np, axes=(0, 1))  # 在空间维度计算FFT
        img_pha = np.angle(img_fft)  # 形状 (H, W, C)

        # 调整img_pha为通道优先 (C, H, W)
        img_pha = np.transpose(img_pha, (2, 0, 1))  # 新形状 (3, 224, 224)

        # 缩放因子形状 (3, 1, 1) [C, H, W]
        scale_factor = np.array([50000, 50000, 50000]).reshape(3, 1, 1)

        # 计算相位图像
        phase_img = scale_factor * (np.e ** (1j * img_pha))  # 广播成功
        phase_img = np.real(np.fft.ifft2(phase_img, axes=(1, 2)))  # 在H,W维度反变换

        # 调整回通道最后并转换为PIL图像
        phase_img = np.transpose(phase_img, (1, 2, 0))  # (H, W, C)
        phase_img = np.clip(phase_img, 0, 255).astype(np.uint8)
        phase_img = Image.fromarray(phase_img)

        # 转换为张量 (C, H, W)
        phase_img = self.to_tensor(phase_img)
        return phase_img

    def __getitem__(self, item):

        sample_domain = self.domains[item]
        if sample_domain == 'sketch' or sample_domain == 'quickdraw':
            sample = ImageOps.invert(Image.open(self.fls[item])).convert(mode='RGB')
        else:
            sample = Image.open(self.fls[item]).convert(mode='RGB')

        clss = self.clss[item]

        if self.transforms is not None:
            # 应用transforms后，sample会是PyTorch Tensor (C, H, W)
            sample = self.transforms(sample)
        else:
            sample = self.to_tensor(sample)

        # 生成相位图像
        # phase_img = self.phase_img_getter(sample)

        # print(phase_img.shape)

        return sample, clss, sample_domain

    def __len__(self):
        return len(self.fls)


class BaselineDataset_path(data.Dataset):
    def __init__(self, fls, transforms=None):

        self.fls = fls
        self.clss = np.array([f.split('/')[-2] for f in fls])
        self.domains = np.array([f.split('/')[-3] for f in fls])
        self.transforms = transforms

    def __getitem__(self, item):

        sample_domain = self.domains[item]
        if sample_domain == 'sketch' or sample_domain == 'quickdraw':
            sample = ImageOps.invert(Image.open(self.fls[item])).convert(mode='RGB')
        else:
            sample = Image.open(self.fls[item]).convert(mode='RGB')

        clss = self.clss[item]

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample, clss, sample_domain, str(self.fls[item])

    def __len__(self):
        return len(self.fls)


def generate_filter(shape, radius, filtType='LPF'):
    """
    生成理想低通或高通滤波器掩码，shape为图像尺寸 (H, W, C)
    """
    h, w = shape[0], shape[1]
    y, x = np.ogrid[:h, :w]
    center = (h / 2, w / 2)
    distance = np.sqrt((x - center[1]) ** 2 + (y - center[0]) ** 2)
    if filtType == 'LPF':
        mask = distance <= radius
    elif filtType == 'HPF':
        mask = distance > radius
    else:
        raise ValueError("Only LPF and HPF are supported")
    # 扩展最后一个通道维度，使得与图像的通道数匹配
    return mask.astype(np.float32)[..., np.newaxis]


def colorful_spectrum_mix(img1, img2, aug_alpha, hpf_range, hpf_alpha):
    """
    对两张图片在频谱域进行混合：
      - 先计算两幅图像的 FFT（频谱中心化）
      - 随机生成混合比例 lam
      - 混合幅值： lam * abs(img2) + (1 - lam) * abs(img1)
      - 若 hpf_range > 0，则构造低通掩码（LPF）和高通掩码（HPF），
        在低频区域对相位进行加权混合： (1-hpf_alpha)*phase(img1) + hpf_alpha*phase(img2)
      - 对混合后的幅值和相位做逆 FFT 得到混合后的图像
    """
    # 确保两图形状一致
    if img1.shape != img2.shape:
        img2 = np.array(Image.fromarray(img2).resize((img1.shape[1], img1.shape[0])))

    # 混合比例（取值范围[0, aug_alpha]）
    lam = np.random.uniform(0, aug_alpha)

    # 计算 FFT 并中心化
    fft1 = np.fft.fft2(img1, axes=(0, 1))
    fft2 = np.fft.fft2(img2, axes=(0, 1))
    fft1_shift = np.fft.fftshift(fft1)
    fft2_shift = np.fft.fftshift(fft2)

    # 分解出幅值和相位
    mag1, phase1 = np.abs(fft1_shift), np.angle(fft1_shift)
    mag2, phase2 = np.abs(fft2_shift), np.angle(fft2_shift)

    # 混合幅值
    mag_mixed = lam * mag2 + (1 - lam) * mag1

    # 混合相位：在低频区域使用另一张图的相位
    if hpf_range > 0:
        mask_lpf = generate_filter(img1.shape, hpf_range, 'LPF')
        mask_hpf = generate_filter(img1.shape, hpf_range, 'HPF')
        # 在低频区域进行加权混合；高频部分保持 img1 的相位
        phase_mixed = mask_hpf * phase1 + mask_lpf * ((1 - hpf_alpha) * phase1 + hpf_alpha * phase2)
    else:
        phase_mixed = phase1

    # 重构混合后的频谱，并逆变换回空间域
    fft_mixed_shift = mag_mixed * np.exp(1j * phase_mixed)
    fft_mixed = np.fft.ifftshift(fft_mixed_shift)
    img_mixed = np.fft.ifft2(fft_mixed, axes=(0, 1)).real

    # 裁剪至合法像素值
    img_mixed = np.clip(img_mixed, 0, 255).astype(np.uint8)
    return img_mixed


class CuMixloader(data.Dataset):
    def __init__(self, fls, clss, doms, dict_domain, transforms=None, aug_alpha=1.0,
                 hpf_range=50,
                 hpf_alpha=0.4,
                 fft_aug_prob=0.5):
        self.fls = fls
        self.clss = clss
        self.domains = doms
        self.dict_domain = dict_domain
        self.transforms = transforms
        self.to_tensor = T.ToTensor()

        # FFT增强参数
        self.aug_alpha = max(0.0, min(aug_alpha, 2.0))  # 限制在0~2范围
        self.hpf_range = max(1, hpf_range)  # 至少1像素
        self.hpf_alpha = max(0.0, min(hpf_alpha, 1.0))  # 限制在0~1范围
        self.fft_aug_prob = max(0.0, min(fft_aug_prob, 1.0))

    def phase_img_getter(self, img):
        """生成相位图像（适配通道优先）"""
        # 输入处理
        if isinstance(img, torch.Tensor):
            img_np = img.permute(1, 2, 0).numpy()  # 从 (C, H, W) 转为 (H, W, C)
        else:
            img_np = np.array(img)  # PIL.Image的shape是 (H, W, C)

        # 计算FFT相位
        img_fft = np.fft.fft2(img_np, axes=(0, 1))  # 在空间维度计算FFT
        img_pha = np.angle(img_fft)  # 形状 (H, W, C)

        # 调整img_pha为通道优先 (C, H, W)
        img_pha = np.transpose(img_pha, (2, 0, 1))  # 新形状 (3, H, W)

        # 缩放因子形状 (3, 1, 1) [C, H, W]
        scale_factor = np.array([50000, 50000, 50000]).reshape(3, 1, 1)

        # 计算相位图像
        phase_img = scale_factor * (np.e ** (1j * img_pha))  # 广播成功
        phase_img = np.real(np.fft.ifft2(phase_img, axes=(1, 2)))  # 在H,W维度反变换

        # 调整回通道最后并转换为PIL图像
        phase_img = np.transpose(phase_img, (1, 2, 0))  # (H, W, C)
        phase_img = np.clip(phase_img, 0, 255).astype(np.uint8)
        phase_img = Image.fromarray(phase_img)

        # 转换为张量 (C, H, W)
        phase_img = self.to_tensor(phase_img)
        return phase_img

    def __getitem__(self, item):
        sample_domain = self.domains[item]
        if sample_domain in ['sketch', 'quickdraw']:
            sample = ImageOps.invert(Image.open(self.fls[item])).convert('RGB')
        else:
            sample = Image.open(self.fls[item]).convert('RGB')

        clss = self.clss[item]

        if self.transforms is not None:
            sample_trans = self.transforms(sample)
        else:
            sample_trans = self.to_tensor(sample)
        rgb_image = sample_trans
        if random.random() < self.fft_aug_prob:
            idx2 = random.randint(0, len(self.fls) - 1)
            while idx2 == item:
                idx2 = random.randint(0, len(self.fls) - 1)
            sample2_domain = self.domains[idx2]
            if sample2_domain in ['sketch', 'quickdraw']:
                sample2 = ImageOps.invert(Image.open(self.fls[idx2])).convert('RGB')
            else:
                sample2 = Image.open(self.fls[idx2]).convert('RGB')
            if self.transforms is not None:
                sample2_trans = self.transforms(sample2)
            else:
                sample2_trans = sample2
            if isinstance(sample_trans, Image.Image):
                img1 = np.array(sample_trans)
            else:
                img1 = sample_trans.permute(1, 2, 0).numpy()
            if isinstance(sample2_trans, Image.Image):
                img2 = np.array(sample2_trans)
            else:
                img2 = sample2_trans.permute(1, 2, 0).numpy()
            mixed_img = colorful_spectrum_mix(img1, img2, self.aug_alpha, self.hpf_range, self.hpf_alpha)
            sample = Image.fromarray(mixed_img)
            if self.transforms is not None:
                sample = self.transforms(sample)
        else:
            sample = sample_trans

        phase_img = self.phase_img_getter(sample)
        return rgb_image, phase_img, clss, sample_domain

    def __len__(self):
        return len(self.fls)


class PairedContrastiveImageDataset(data.Dataset):
    def __init__(self, fls, clss, doms, dict_domain, dict_clss, transforms, nviews, nothers):
        self.fls = fls
        self.clss = clss
        self.cls_ids = self.convert_text_to_number(self.clss, dict_clss)
        self.domains = doms
        self.dict_domain = dict_domain
        self.dict_clss = dict_clss
        self.transforms = transforms
        self.idx = torch.arange(len(self.cls_ids))
        self.nviews = nviews
        self.nothers = nothers

    def convert_text_to_number(self, clss, dict_clss):
        return torch.tensor([dict_clss[i] for i in clss])

    def __getitem__(self, item):
        sample_domain = self.domains[item]
        if sample_domain == 'sketch' or sample_domain == 'quickdraw':
            sample = ImageOps.invert(Image.open(self.fls[item])).convert(mode='RGB')
        else:
            sample = Image.open(self.fls[item]).convert(mode='RGB')
        clss_name = self.clss[item]
        clss = self.cls_ids[item]

        allSamples = [sample]

        for _ in range(self.nothers):
            allSamples.append(self.pickRandomSample(clss)[0])

        allImages = list()
        for i in allSamples:
            for _ in range(self.nviews):
                allImages.append(self.transforms(i))
        return torch.stack(allImages), clss_name, sample_domain

    def pickRandomSample(self, clss):
        targetIdx = self.idx[self.cls_ids == clss]
        randIdx = targetIdx[torch.randperm(len(targetIdx))[0]]

        sample_domain = self.domains[randIdx]
        if sample_domain == 'sketch' or sample_domain == 'quickdraw':
            sample = ImageOps.invert(Image.open(self.fls[randIdx])).convert(mode='RGB')
        else:
            sample = Image.open(self.fls[randIdx]).convert(mode='RGB')

        return sample, self.dict_domain[self.domains[randIdx]], self.cls_ids[randIdx]

    def __len__(self):
        return len(self.fls)


class PairedContrastiveImageDataset_SameDomain(data.Dataset):
    def __init__(self, fls, clss, doms, dict_domain, dict_clss, transforms, nviews, nothers):
        self.fls = fls
        self.clss = clss
        self.cls_ids = self.convert_text_to_number(self.clss, dict_clss)
        self.domains = doms
        self.dom_ids = self.convert_text_to_number(self.domains, dict_domain)
        self.dict_domain = dict_domain
        self.dict_clss = dict_clss
        self.transforms = transforms
        self.idx = torch.arange(len(self.cls_ids))
        self.nviews = nviews
        self.nothers = nothers

    def convert_text_to_number(self, clss, dict_clss):
        return torch.tensor([dict_clss[i] for i in clss])

    def __getitem__(self, item):
        sample_domain = self.domains[item]
        if sample_domain == 'sketch' or sample_domain == 'quickdraw':
            sample = ImageOps.invert(Image.open(self.fls[item])).convert(mode='RGB')
        else:
            sample = Image.open(self.fls[item]).convert(mode='RGB')

        clss = self.cls_ids[item]

        allSamples = [sample]

        for _ in range(self.nothers):
            allSamples.append(self.pickRandomSample(clss, self.dom_ids[item])[0])

        allImages = list()
        for i in allSamples:
            for _ in range(self.nviews):
                allImages.append(self.transforms(i))
        return torch.stack(allImages), clss

    def pickRandomSample(self, clss, dom):
        targetIdx = self.idx[torch.logical_and(self.cls_ids == clss, self.dom_ids == dom)]
        randIdx = targetIdx[torch.randperm(len(targetIdx))[0]]

        sample_domain = self.domains[randIdx]
        if sample_domain == 'sketch' or sample_domain == 'quickdraw':
            sample = ImageOps.invert(Image.open(self.fls[randIdx])).convert(mode='RGB')
        else:
            sample = Image.open(self.fls[randIdx]).convert(mode='RGB')

        return sample, self.dict_domain[self.domains[randIdx]], self.cls_ids[randIdx]

    def __len__(self):
        return len(self.fls)


class SAKELoader(data.Dataset):
    def __init__(self, fls, cid_mask, transforms=None):

        self.fls = fls
        self.cid_mask = cid_mask
        self.clss = np.array([f.split('/')[-2] for f in fls])
        self.domains = np.array([f.split('/')[-3] for f in fls])
        self.transforms = transforms

    def __getitem__(self, item):

        sample_domain = self.domains[item]
        if sample_domain == 'sketch':
            sample = ImageOps.invert(Image.open(self.fls[item])).convert(mode='RGB')
        else:
            sample = Image.open(self.fls[item]).convert(mode='RGB')

        clss = self.clss[item]

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample, clss, self.cid_mask[clss]

    def __len__(self):
        return len(self.fls)


class SAKELoader_with_domainlabel(data.Dataset):
    def __init__(self, fls, cid_mask=None, transforms=None):

        self.fls = fls
        self.cid_mask = cid_mask
        self.clss = np.array([f.split('/')[-2] for f in fls])
        self.domains = np.array([f.split('/')[-3] for f in fls])
        self.transforms = transforms

    def __getitem__(self, item):

        sample_domain = self.domains[item]
        if sample_domain == 'sketch':
            sample = ImageOps.invert(Image.open(self.fls[item])).convert(mode='RGB')
            domain_label = np.array([0])
        else:
            sample = Image.open(self.fls[item]).convert(mode='RGB')
            domain_label = np.array([1])

        clss = self.clss[item]

        if self.transforms is not None:
            sample = self.transforms(sample)

        if self.cid_mask is not None:
            return sample, clss, self.cid_mask[clss], domain_label
        else:
            return sample, clss, domain_label

    def __len__(self):
        return len(self.fls)


class Doodle2Search_Loader(data.Dataset):
    def __init__(self, fls_sketch, fls_image, semantic_vec, tr_classes, dict_clss, transforms=None):
        self.fls_sketch = fls_sketch
        self.fls_image = fls_image

        self.cls_sketch = np.array([f.split('/')[-2] for f in self.fls_sketch])
        self.cls_image = np.array([f.split('/')[-2] for f in self.fls_image])

        self.tr_classes = tr_classes
        self.dict_clss = dict_clss

        self.semantic_vec = semantic_vec
        # self.sim_matrix = np.exp(-np.square(cdist(self.semantic_vec, self.semantic_vec, 'euclidean'))/0.1)
        cls_euc = cdist(self.semantic_vec, self.semantic_vec, 'euclidean')
        cls_euc_scaled = cls_euc / np.expand_dims(np.max(cls_euc, axis=1), axis=1)
        self.sim_matrix = np.exp(-cls_euc_scaled)

        self.transforms = transforms

    def __getitem__(self, item):
        sketch = ImageOps.invert(Image.open(self.fls_sketch[item])).convert(mode='RGB')
        sketch_cls = self.cls_sketch[item]
        sketch_cls_numeric = self.dict_clss.get(sketch_cls)

        w2v = torch.FloatTensor(self.semantic_vec[sketch_cls_numeric, :])

        # Find negative sample
        possible_classes = self.tr_classes[self.tr_classes != sketch_cls]
        sim = self.sim_matrix[sketch_cls_numeric, :]
        sim = np.array([sim[self.dict_clss.get(x)] for x in possible_classes])

        # norm = np.linalg.norm(sim, ord=1) # Similarity to probability
        # sim = sim/norm
        sim /= np.sum(sim)

        image_neg_cls = np.random.choice(possible_classes, 1, p=sim)[0]
        image_neg = Image.open(
            np.random.choice(self.fls_image[np.where(self.cls_image == image_neg_cls)[0]], 1)[0]).convert(mode='RGB')

        image_pos = Image.open(
            np.random.choice(self.fls_image[np.where(self.cls_image == sketch_cls)[0]], 1)[0]).convert(mode='RGB')

        if self.transforms is not None:
            sketch = self.transforms(sketch)
            image_pos = self.transforms(image_pos)
            image_neg = self.transforms(image_neg)

        return sketch, image_pos, image_neg, w2v

    def __len__(self):
        return len(self.fls_sketch)


class JigsawDataset(data.Dataset):
    def __init__(self, fls, transforms=None, jig_classes=30, bias_whole_image=0.9):

        self.fls = fls
        self.clss = np.array([f.split('/')[-2] for f in fls])
        self.domains = np.array([f.split('/')[-3] for f in fls])

        self.permutations = self.__retrieve_permutations(jig_classes)
        self.grid_size = 3
        self.bias_whole_image = bias_whole_image  # biases the training procedure to show the whole image more often

        self._image_transformer = transforms['image']
        self._augment_tile = transforms['tile']

        def make_grid(x):
            return torchvision.utils.make_grid(x, self.grid_size, padding=0)

        self.returnFunc = make_grid

    def get_tile(self, img, n):
        w = float(img.size[0]) / self.grid_size
        y = int(n / self.grid_size)
        x = n % self.grid_size
        tile = img.crop([x * w, y * w, (x + 1) * w, (y + 1) * w])
        tile = self._augment_tile(tile)
        return tile

    def get_image(self, item):

        sample_domain = self.domains[item]
        if sample_domain == 'sketch' or sample_domain == 'quickdraw':
            sample = ImageOps.invert(Image.open(self.fls[item])).convert(mode='RGB')
        else:
            sample = Image.open(self.fls[item]).convert(mode='RGB')
        return self._image_transformer(sample)

    def __getitem__(self, item):

        img = self.get_image(item)
        n_grids = self.grid_size ** 2
        tiles = [None] * n_grids
        for n in range(n_grids):
            tiles[n] = self.get_tile(img, n)

        order = np.random.randint(len(self.permutations) + 1)  # added 1 for class 0: unsorted
        if self.bias_whole_image:
            if self.bias_whole_image > random():
                order = 0

        if order == 0:
            data = tiles
        else:
            data = [tiles[self.permutations[order - 1][t]] for t in range(n_grids)]

        data = torch.stack(data, 0)
        data = self.returnFunc(data)

        return torch.cat([self._augment_tile(img), data], 0), order, self.clss[item]

    def __len__(self):
        return len(self.fls)

    def __retrieve_permutations(self, classes):
        all_perm = np.load(os.path.join(_BASE_PATH, 'permutations_%d.npy' % (classes)))
        # from range [1,9] to [0,8]
        if all_perm.min() == 1:
            all_perm = all_perm - 1

        return all_perm
