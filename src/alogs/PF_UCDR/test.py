from matplotlib import patches
from tqdm import tqdm
from ...models.PF_UCDR import PF_UCDR
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from src.data.DomainNet import domainnet
from src.data.Sketchy import sketchy_extended
from src.data.TUBerlin import tuberlin_extended
import numpy as np
import torch.backends.cudnn as cudnn
from src.data.dataloaders import CuMixloader, BaselineDataset
from src.data.sampler import BalancedSampler
from src.utils import utils, GPUmanager
from src.utils.metrics import compute_retrieval_metrics
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec

try:
    from torchvision.transforms import InterpolationMode

    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
gm = GPUmanager.GPUManager()
gpu_index = gm.auto_choice()
device = torch.device(f'cuda:{gpu_index}' if torch.cuda.is_available() else 'cpu')
# -*- coding: utf-8 -*-
# !/usr/bin/python
from datetime import datetime

import sys
import os

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
from src.alogs.PF_UCDR.trainer import Trainer
from src.options.options import Options
import os


def main(args):
    trainer = Trainer(args)
    trainer.test()


os.makedirs('retrieval', exist_ok=True)


class Trainer:

    def __init__(self, args):
        self.args = args
        print('\nLoading data...')
        if args.dataset == 'Sketchy':
            data_input = sketchy_extended.create_trvalte_splits(args)
        if args.dataset == 'DomainNet':
            data_input = domainnet.create_trvalte_splits(args)
        if args.dataset == 'TUBerlin':
            data_input = tuberlin_extended.create_trvalte_splits(args)

        self.tr_classes = data_input['tr_classes']
        self.va_classes = data_input['va_classes']
        self.te_classes = data_input['te_classes']
        self.data_splits = data_input['splits']
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        use_gpu = torch.cuda.is_available()
        self.weight_path = args.weight

        if use_gpu:
            cudnn.benchmark = True
            torch.cuda.manual_seed_all(args.seed)

        # Imagenet standards
        im_mean = [0.485, 0.456, 0.406]
        im_std = [0.229, 0.224, 0.225]
        # Image transformations
        self.image_transforms = {
            'train':
                transforms.Compose([
                    transforms.RandomResizedCrop((args.image_size, args.image_size), (0.8, 1.0)),
                    transforms.RandomHorizontalFlip(0.5),
                    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.4),
                    transforms.ToTensor(),
                    transforms.Normalize(im_mean, im_std)
                ]),
            'eval': transforms.Compose([
                transforms.Resize(args.image_size, interpolation=BICUBIC),
                transforms.CenterCrop(args.image_size),
                transforms.ToTensor(),
                # lambda image: image.convert("RGB"),
                transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            ])
        }

        # class dictionary
        self.dict_clss = utils.create_dict_texts(self.tr_classes)
        self.te_dict_class = utils.create_dict_texts(self.tr_classes + self.va_classes + self.te_classes)

        fls_tr = self.data_splits['tr']
        cls_tr = np.array([f.split('/')[-2] for f in fls_tr])
        dom_tr = np.array([f.split('/')[-3] for f in fls_tr])
        tr_domains_unique = np.unique(dom_tr)

        # doamin dictionary
        self.dict_doms = utils.create_dict_texts(tr_domains_unique)
        print(self.dict_doms)
        domain_ids = utils.numeric_classes(dom_tr, self.dict_doms)
        data_train = CuMixloader(fls_tr, cls_tr, dom_tr, self.dict_doms, transforms=self.image_transforms['train'])
        train_sampler = BalancedSampler(domain_ids, args.batch_size // len(tr_domains_unique),
                                        domains_per_batch=len(tr_domains_unique))
        self.train_loader = DataLoader(dataset=data_train, batch_size=args.batch_size, sampler=train_sampler,
                                       num_workers=args.num_workers,
                                       pin_memory=True)
        self.train_loader_for_SP = DataLoader(dataset=data_train, batch_size=100, sampler=train_sampler,
                                              num_workers=args.num_workers,
                                              pin_memory=True)

        print('Loading Done\n')

        self.model = PF_UCDR(self.args, self.dict_clss, self.dict_doms, device, isTest=True)
        weight = torch.load(
            self.weight_path)[
            "model_state_dict"]
        self.model.load_state_dict(weight)
        self.model = self.model.to(device)

        if args.dataset == 'DomainNet':
            self.save_folder_name = 'seen-' + args.seen_domain + '_unseen-' + args.holdout_domain + '_x_' + args.gallery_domain
            if not args.include_auxillary_domains:
                self.save_folder_name += '_noaux'
        elif args.dataset == 'Sketchy':
            if args.is_eccv_split:
                self.save_folder_name = 'eccv_split'
            else:
                self.save_folder_name = 'random_split'
        else:
            self.save_folder_name = ''

        if args.dataset == 'DomainNet' or (args.dataset == 'Sketchy' and args.is_eccv_split):
            self.map_metric = 'mAP@200'
            self.prec_metric = 'prec@200'
        else:
            self.map_metric = 'mAP@all'
            self.prec_metric = 'prec@100'

        self.suffix = 'e-' + str(args.epochs) + '_es-' + str(args.early_stop) + '_opt-' + args.optimizer + \
                      '_bs-' + str(args.batch_size) + '_lr-' + str(args.lr)

        # exit(0)
        path_log = os.path.join(args.root_path, 'logs', args.dataset, self.save_folder_name, self.suffix)
        self.path_cp = os.path.join(args.root_path, 'src/alogs/PF_UCDR/saved_models', args.dataset,
                                    self.save_folder_name)

        # Logger
        print('Setting logger...', end='')
        self.logger = SummaryWriter(path_log)
        print('Done\n')

        self.start_epoch = 0
        self.best_map = 0
        self.early_stop_counter = 0
        self.last_chkpt_name = 'init'

        print("================Start Testing=================")
        print("==================================================")

        # self.resume_from_checkpoint(args.resume_dict)

    def test(self):
        if self.args.dataset == 'DomainNet':
            if self.args.ucddr == 0:
                te_data = []
                # query_domains_for_vis = ['sketch', 'painting', 'clipart', 'infograph', 'quickdraw']
                # print(self.tr_classes)
                # target_class_for_vis = 'airplane'
                # print(self.te_classes)
                # print(target_class_for_vis not in self.te_dict_class)
                # if target_class_for_vis not in self.te_dict_class.keys():
                #     print(f"Warning: Target class '{target_class_for_vis}' not found. Please choose an existing class.")
                #
                # gallery_domain_for_vis = 'real'
                # classes_for_vis = ['ladder', 'boomerang', 'rainbow', 'onion', 'finger']
                # query_indices_to_try = [i for i in range(50)]
                # evaluate_and_visualize_retrieval(
                #     model=self.model,
                #     args=self.args,
                #     te_dict_class=self.te_dict_class,  # 包含所有类别的字典
                #     dict_doms=self.dict_doms,  # 包含所有域的字典
                #     device=device,  # 全局 device
                #     image_transforms_eval=self.image_transforms['eval'],
                #     query_domain_names_vis=query_domains_for_vis,
                #     gallery_domain_name_vis=gallery_domain_for_vis,
                #     target_class_name_vis=target_class_for_vis,
                #     tr_classes=self.tr_classes,
                #     va_classes=self.va_classes,
                #     te_classes=self.te_classes,
                #     data_splits_provider_func=domainnet.trvalte_per_domain,
                #     BaselineDataset_class=BaselineDataset,
                #     utils_module=utils,
                #     index=20
                # )
                # visualize_quickdraw_retrieval_per_class(
                #     model=self.model,
                #     args=args,
                #     te_dict_class=self.te_dict_class,
                #     dict_doms=self.dict_doms,
                #     device=device,
                #     image_transforms_eval=self.image_transforms['eval'],
                #     target_class_names_vis=classes_for_vis,  # 仍然是类别列表
                #     query_domain_name_fixed="quickdraw",
                #     gallery_domain_name_vis="real",
                #     tr_classes=self.tr_classes,
                #     va_classes=self.va_classes,
                #     te_classes=self.te_classes,
                #     data_splits_provider_func=domainnet.trvalte_per_domain,
                #     BaselineDataset_class=BaselineDataset,
                #     utils_module=utils,
                #     num_retrieved=10,
                #     q_idx_to_use_for_query=17
                # )
                # for q_idx_val in query_indices_to_try:
                #     print(f"\n--- Generating visualization using query_index: {q_idx_val} ---")
                #     visualize_quickdraw_retrieval_per_class(
                #         model=self.model,
                #         args=args,
                #         te_dict_class=self.te_dict_class,
                #         dict_doms=self.dict_doms,
                #         device=device,
                #         image_transforms_eval=self.image_transforms['eval'],
                #         target_class_names_vis=classes_for_vis,  # 仍然是类别列表
                #         query_domain_name_fixed="quickdraw",
                #         gallery_domain_name_vis="real",
                #         tr_classes=self.tr_classes,
                #         va_classes=self.va_classes,
                #         te_classes=self.te_classes,
                #         data_splits_provider_func=domainnet.trvalte_per_domain,
                #         BaselineDataset_class=BaselineDataset,
                #         utils_module=utils,
                #         num_retrieved=10,
                #         q_idx_to_use_for_query=q_idx_val  # <--- 传递当前的查询索引
                #     )
                # if hasattr(domainnet, 'trvalte_per_domain'):
                #     query_index = [i + 3 for i in range(20)]
                #
                #     for q_idx_val in query_index:
                #         evaluate_and_visualize_retrieval(
                #             model=self.model,
                #             args=self.args,
                #             te_dict_class=self.te_dict_class,  # 包含所有类别的字典
                #             dict_doms=self.dict_doms,  # 包含所有域的字典
                #             device=device,  # 全局 device
                #             image_transforms_eval=self.image_transforms['eval'],
                #             query_domain_names_vis=query_domains_for_vis,
                #             gallery_domain_name_vis=gallery_domain_for_vis,
                #             target_class_name_vis=target_class_for_vis,
                #             tr_classes=self.tr_classes,
                #             va_classes=self.va_classes,
                #             te_classes=self.te_classes,
                #             data_splits_provider_func=domainnet.trvalte_per_domain,
                #             BaselineDataset_class=BaselineDataset,
                #             utils_module=utils,
                #             index=q_idx_val
                #         )
                # else:
                #     print("Warning: domainnet.trvalte_per_domain function not found. Skipping visualization.")
                # for domain in [self.args.seen_domain, self.args.holdout_domain]:
                for domain in [self.args.holdout_domain]:
                    for includeSeenClassinTestGallery in [0, 1]:
                        test_head_str = 'Query:' + domain + '; Gallery:' + self.args.gallery_domain + '; Generalized:' + str(
                            includeSeenClassinTestGallery)
                        print(test_head_str)
                        # pdb.set_trace()

                        splits_query = domainnet.trvalte_per_domain(self.args, domain, 0, self.tr_classes,
                                                                    self.va_classes,
                                                                    self.te_classes)
                        splits_gallery = domainnet.trvalte_per_domain(self.args, self.args.gallery_domain,
                                                                      includeSeenClassinTestGallery, self.tr_classes,
                                                                      self.va_classes, self.te_classes)

                        data_te_query = BaselineDataset(np.array(splits_query['te']),
                                                        transforms=self.image_transforms['eval'])
                        data_te_gallery = BaselineDataset(np.array(splits_gallery['te']),
                                                          transforms=self.image_transforms['eval'])

                        # PyTorch test loader for query
                        te_loader_query = DataLoader(dataset=data_te_query, batch_size=self.args.batch_size * 10,
                                                     shuffle=False,
                                                     num_workers=self.args.num_workers, pin_memory=True)
                        # PyTorch test loader for gallery
                        te_loader_gallery = DataLoader(dataset=data_te_gallery, batch_size=self.args.batch_size * 10,
                                                       shuffle=False,
                                                       num_workers=self.args.num_workers, pin_memory=True)

                        # print(f'#Test queries:{len(te_loader_query.dataset)}; #Test gallery samples:{len(te_loader_gallery.dataset)}.')
                        result = evaluate(te_loader_query, te_loader_gallery, self.model, self.te_dict_class,
                                          self.dict_doms, 4, self.args)
                        te_data.append(result)

                        out = f"{self.map_metric} = %.4f, {self.prec_metric} = %.4f\n" % (
                            result[self.map_metric], result[self.prec_metric])

                        print(out)
            else:
                if self.args.holdout_domain == 'quickdraw':
                    p = 0.1
                else:
                    p = 0.25
                splits_query = domainnet.seen_cls_te_samples(self.args, self.args.holdout_domain, self.tr_classes, p)
                splits_gallery = domainnet.seen_cls_te_samples(self.args, self.args.gallery_domain, self.tr_classes, p)
                data_te_query = BaselineDataset(np.array(splits_query), transforms=self.image_transforms['eval'])
                data_te_gallery = BaselineDataset(np.array(splits_gallery), transforms=self.image_transforms['eval'])
                # PyTorch test loader for query
                te_loader_query = DataLoader(dataset=data_te_query, batch_size=2048, shuffle=False,
                                             num_workers=self.args.num_workers, pin_memory=True)
                # PyTorch test loader for gallery
                te_loader_gallery = DataLoader(dataset=data_te_gallery, batch_size=2048, shuffle=False,
                                               num_workers=self.args.num_workers, pin_memory=True)
                te_data = evaluate(te_loader_query, te_loader_gallery, self.model, self.te_dict_class, self.dict_doms,
                                   4,
                                   self.args)
                map_ = te_data[self.map_metric]
                prec = te_data[self.prec_metric]
                out = "mAP@200 = %.4f, Prec@200 = %.4f\n" % (map_, prec)
                print(out)
        else:
            data_te_query = BaselineDataset(self.data_splits['query_te'], transforms=self.image_transforms['eval'])
            data_te_gallery = BaselineDataset(self.data_splits['gallery_te'], transforms=self.image_transforms['eval'])

            te_loader_query = DataLoader(dataset=data_te_query, batch_size=self.args.batch_size * 5, shuffle=False,
                                         num_workers=self.args.num_workers, pin_memory=True)
            te_loader_gallery = DataLoader(dataset=data_te_gallery, batch_size=self.args.batch_size * 5, shuffle=False,
                                           num_workers=self.args.num_workers, pin_memory=True)

            print(
                f'#Test queries:{len(te_loader_query.dataset)}; #Test gallery samples:{len(te_loader_gallery.dataset)}.\n')

            te_data = evaluate(te_loader_query, te_loader_gallery, self.model, self.te_dict_class, self.dict_doms, 4,
                               self.args)
            out = f"{self.map_metric} = %.4f, {self.prec_metric} = %.4f\n" % (
                te_data[self.map_metric], te_data[self.prec_metric])
            map_ = te_data[self.map_metric]
            prec = te_data[self.prec_metric]
            out = f"{self.map_metric} = %.4f, {self.prec_metric} = %.4f\n" % (map_, prec)
            print(out)


@torch.no_grad()
def evaluate(loader_sketch, loader_image, model, dict_clss, dict_doms, stage, args):
    sketchEmbeddings = list()
    sketchLabels = list()

    for i, (sk, cls_sk, dom) in tqdm(enumerate(loader_sketch), desc='Extrac query feature', total=len(loader_sketch)):

        sk = sk.float().to(device)
        cls_id = utils.numeric_classes(cls_sk, dict_clss)
        # pdb.set_trace()
        dom_id = utils.numeric_classes(dom, dict_doms)
        sk_em, _ = model.image_encoder(sk, dom_id, cls_id, stage)
        sketchEmbeddings.append(sk_em)

        cls_numeric = torch.from_numpy(cls_id).long().to(device)

        sketchLabels.append(cls_numeric)
        if args.debug_mode == 1 and i == 2:
            break
    sketchEmbeddings = torch.cat(sketchEmbeddings, 0)
    sketchLabels = torch.cat(sketchLabels, 0)

    realEmbeddings = list()
    realLabels = list()

    for i, (im, cls_im, dom) in tqdm(enumerate(loader_image), desc='Extrac gallery feature', total=len(loader_image)):

        im = im.float().to(device)
        cls_id = utils.numeric_classes(cls_im, dict_clss)
        # pdb.set_trace()
        dom_id = utils.numeric_classes(dom, dict_doms)
        im_em, _ = model.image_encoder(im, dom_id, cls_id, stage)
        realEmbeddings.append(im_em)

        cls_numeric = torch.from_numpy(cls_id).long().to(device)

        realLabels.append(cls_numeric)
        if args.debug_mode == 1 and i == 2:
            break
    realEmbeddings = torch.cat(realEmbeddings, 0)
    realLabels = torch.cat(realLabels, 0)

    print('\nQuery Emb Dim:{}; Gallery Emb Dim:{}'.format(sketchEmbeddings.shape, realEmbeddings.shape))
    eval_data = compute_retrieval_metrics(sketchEmbeddings, sketchLabels, realEmbeddings, realLabels)
    return eval_data


@torch.no_grad()
def evaluate_tsne(loader_sketch, loader_image, model, dict_clss, dict_doms, stage, args, device):
    sketchEmbeddings = list()
    sketchLabels = list()

    for i, (sk, cls_sk, dom) in tqdm(enumerate(loader_sketch), desc='Extrac query feature', total=len(loader_sketch)):
        sk = sk.float().to(device)
        cls_id = utils.numeric_classes(cls_sk, dict_clss)
        dom_id = utils.numeric_classes(dom, dict_doms)
        sk_em, _ = model.image_encoder(sk, dom_id, cls_id, stage)
        sketchEmbeddings.append(sk_em)
        cls_numeric = torch.from_numpy(cls_id).long().to(device)
        sketchLabels.append(cls_numeric)
        if args.debug_mode == 1 and i == 2:
            break
    sketchEmbeddings = torch.cat(sketchEmbeddings, 0)
    sketchLabels = torch.cat(sketchLabels, 0)

    realEmbeddings = list()
    realLabels = list()

    for i, (im, cls_im, dom) in tqdm(enumerate(loader_image), desc='Extrac gallery feature', total=len(loader_image)):
        im = im.float().to(device)
        cls_id = utils.numeric_classes(cls_im, dict_clss)
        dom_id = utils.numeric_classes(dom, dict_doms)
        im_em, _ = model.image_encoder(im, dom_id, cls_id, stage)
        realEmbeddings.append(im_em)
        cls_numeric = torch.from_numpy(cls_id).long().to(device)
        realLabels.append(cls_numeric)
        if args.debug_mode == 1 and i == 2:
            break
    realEmbeddings = torch.cat(realEmbeddings, 0)
    realLabels = torch.cat(realLabels, 0)

    print('\nQuery Emb Dim:{}; Gallery Emb Dim:{}'.format(sketchEmbeddings.shape, realEmbeddings.shape))
    eval_data = compute_retrieval_metrics(sketchEmbeddings, sketchLabels, realEmbeddings, realLabels)

    embeddings = torch.cat([sketchEmbeddings, realEmbeddings], 0)
    labels = torch.cat([sketchLabels, realLabels], 0)
    domains = torch.cat([torch.zeros(sketchEmbeddings.size(0), device=device),
                         torch.ones(realEmbeddings.size(0), device=device)], 0)

    chosen_labels = [316, 319, 326, 304, 325, 304, 330, 301]
    selected_labels = torch.tensor(chosen_labels)

    mask = torch.zeros_like(labels, dtype=torch.bool)
    for label_val in selected_labels:
        mask |= (labels == label_val)

    filtered_embeddings = embeddings[mask]
    filtered_labels = labels[mask]
    filtered_domains = domains[mask]

    max_samples_per_class = 100
    sampled_embeddings = []
    sampled_labels = []
    sampled_domains = []

    unique_chosen_labels = selected_labels.unique()
    for label_val in unique_chosen_labels:
        label_mask = (filtered_labels == label_val)
        current_label_embeddings = filtered_embeddings[label_mask]
        current_label_domains = filtered_domains[label_mask]
        current_filtered_labels_for_class = filtered_labels[label_mask]

        num_samples_for_class = len(current_label_embeddings)
        if num_samples_for_class == 0:
            continue

        if num_samples_for_class > max_samples_per_class:
            sampled_indices = torch.randperm(num_samples_for_class, device=current_label_embeddings.device)[
                              :max_samples_per_class]
        else:
            sampled_indices = torch.arange(num_samples_for_class, device=current_label_embeddings.device)

        sampled_embeddings.append(current_label_embeddings[sampled_indices])
        sampled_labels.append(current_filtered_labels_for_class[sampled_indices])
        sampled_domains.append(current_label_domains[sampled_indices])

    if not sampled_embeddings:  # 处理没有样本被选中的情况
        print("Warning: No samples selected for TSNE plot after filtering and sampling.")
        return eval_data

    filtered_embeddings = torch.cat(sampled_embeddings, dim=0)
    filtered_labels = torch.cat(sampled_labels, dim=0)
    filtered_domains = torch.cat(sampled_domains, dim=0)

    # filtered_embeddings = filtered_embeddings.to(device) # 它们应该已经在 device上了
    # filtered_labels = filtered_labels.to(device)
    # filtered_domains = filtered_domains.to(device)

    tsne = TSNE(n_components=2, random_state=0)  # 现在 TSNE 已经被导入
    tsne_results = tsne.fit_transform(filtered_embeddings.cpu().numpy())

    # 绘图代码依赖于 matplotlib 和 datetime
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from datetime import datetime

    plt.figure(figsize=(10, 8))  # 调整了图像大小
    class_colors = ['#e57373', '#64b5f6', '#81c784', '#ba68c8', '#a1887f', '#f06292', '#90a4ae', '#4db6ac']
    domain_markers = {'real': '^', 'sketch': '*'}

    unique_display_labels = filtered_labels.unique()  # 使用实际过滤后的标签进行绘图和图例
    label_to_color_map = {lbl.item(): color for lbl, color in
                          zip(unique_display_labels, class_colors[:len(unique_display_labels)])}

    for unique_label_val_tensor in unique_display_labels:
        unique_label_val = unique_label_val_tensor.item()
        idx = (filtered_labels == unique_label_val_tensor)
        for domain_val_int in [0, 1]:
            domain_name = 'sketch' if domain_val_int == 0 else 'real'
            domain_idx_bool = idx & (filtered_domains == domain_val_int)
            domain_idx_cpu = domain_idx_bool.cpu()

            if torch.any(domain_idx_cpu):
                plt.scatter(tsne_results[domain_idx_cpu, 0], tsne_results[domain_idx_cpu, 1],
                            color=label_to_color_map.get(unique_label_val, '#000000'),  # 提供一个默认颜色
                            marker=domain_markers[domain_name],
                            alpha=0.7)  # 增加透明度以便观察重叠点

    class_legend_elements = [Line2D([0], [0], marker='o', color='w', label=f'Class {lbl.item()}',
                                    markerfacecolor=label_to_color_map.get(lbl.item(), '#000000'), markersize=10)
                             for lbl in unique_display_labels]
    domain_legend_elements = [Line2D([0], [0], marker=marker, color='w', label=domain.title(),
                                     markerfacecolor='black', markersize=10)
                              for domain, marker in domain_markers.items()]

    legend_elements = class_legend_elements + domain_legend_elements
    plt.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.1),
               fancybox=True, shadow=True,
               ncol=max(1, (len(class_legend_elements) + len(domain_legend_elements)) // 3))  # 调整ncol

    plt.title(f"T-SNE: Sketch & Real ({'Train' if stage == 'train' else 'Valid'}) - Cls+Dom")
    plt.xlabel('TSNE-1')
    plt.ylabel('TSNE-2')
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    plt.savefig(f"tsne_plot_detailed_{current_time}.png", bbox_inches='tight')
    plt.show()

    plt.figure(figsize=(10, 8))  # 新的figure
    markers_domain_plot2 = ['o', 'x']
    for unique_label_val_tensor in unique_display_labels:
        unique_label_val = unique_label_val_tensor.item()
        idx = (filtered_labels == unique_label_val_tensor)
        for j, domain_val_int in enumerate([0, 1]):
            domain_idx_bool = idx & (filtered_domains == domain_val_int)
            domain_idx_cpu = domain_idx_bool.cpu()
            if torch.any(domain_idx_cpu):
                plt.scatter(tsne_results[domain_idx_cpu, 0], tsne_results[domain_idx_cpu, 1],
                            color=label_to_color_map.get(unique_label_val, '#000000'),
                            marker=markers_domain_plot2[j],
                            alpha=0.7)

    # 为第二个图创建图例
    legend_handles_plot2 = []
    for lbl_tensor in unique_display_labels:
        lbl_item = lbl_tensor.item()
        legend_handles_plot2.append(Line2D([0], [0], marker='s', color='w',
                                           markerfacecolor=label_to_color_map.get(lbl_item, '#000000'),
                                           label=f'Class {lbl_item}', markersize=8))
    for i, dom_name in enumerate(['Sketch (Dom 0)', 'Real (Dom 1)']):
        legend_handles_plot2.append(Line2D([0], [0], marker=markers_domain_plot2[i], color='w',
                                           markerfacecolor='black', label=dom_name, markersize=8))

    plt.legend(handles=legend_handles_plot2, loc='best', ncol=2)
    plt.title(f"T-SNE: Sketch & Real ({'Train' if stage == 'train' else 'Valid'}) - Simple")
    plt.xlabel('TSNE-1')
    plt.ylabel('TSNE-2')
    current_time_2 = datetime.now().strftime("%Y%m%d-%H%M%S")
    plt.savefig(f"tsne_plot_simple_{current_time_2}.png", bbox_inches='tight')
    plt.show()

    return eval_data


@torch.no_grad()
def evaluate_and_visualize_retrieval(
        model, args, te_dict_class, dict_doms, device, image_transforms_eval,
        query_domain_names_vis, gallery_domain_name_vis, target_class_name_vis,
        tr_classes, va_classes, te_classes,
        data_splits_provider_func,
        BaselineDataset_class,
        utils_module,
        num_retrieved=10,
        index=10
):
    model.eval()
    print(f"\nStarting retrieval visualization for class: '{target_class_name_vis}'")

    num_query_domains = len(query_domain_names_vis)

    cols_per_domain_label = 1
    cols_per_query_image = 1
    cols_per_unseen_gallery_image = 1
    cols_per_mixed_gallery_image = 1

    sep_ratio_query_unseen = 0.8
    sep_ratio_unseen_mixed = 0.8

    gs_width_ratios = []
    gs_width_ratios.append(0.5)
    gs_width_ratios.append(1.0)
    gs_width_ratios.append(sep_ratio_query_unseen)
    for _ in range(num_retrieved):
        gs_width_ratios.append(cols_per_unseen_gallery_image)
    gs_width_ratios.append(sep_ratio_unseen_mixed)
    for _ in range(num_retrieved):
        gs_width_ratios.append(cols_per_mixed_gallery_image)

    num_gs_cols = len(gs_width_ratios)

    fig_width_per_unit_ratio = 1.3
    fig_height_per_row = 1.5
    total_fig_width = sum(gs_width_ratios) * fig_width_per_unit_ratio
    total_fig_height = num_query_domains * fig_height_per_row + fig_height_per_row * 0.6  # 顶部标题空间

    fig = plt.figure(figsize=(total_fig_width, total_fig_height))
    gs = GridSpec(num_query_domains, num_gs_cols, figure=fig, width_ratios=gs_width_ratios)

    if num_query_domains > 0:
        query_img_gs_col_idx_for_title = cols_per_domain_label
        if query_img_gs_col_idx_for_title < num_gs_cols:
            ax_q_title = fig.add_subplot(gs[0, query_img_gs_col_idx_for_title])
            ax_q_title.set_title("Query Domain", fontsize=10, pad=15, fontweight='bold', loc='left')  # loc='left' 使其靠左
            ax_q_title.axis('off')

        unseen_gallery_start_gs_col_idx_for_title = cols_per_domain_label + cols_per_query_image + 1  # 跳过间隔列
        unseen_gallery_title_center_gs_col_idx = unseen_gallery_start_gs_col_idx_for_title + (num_retrieved // 2)
        if num_retrieved > 0 and unseen_gallery_title_center_gs_col_idx < num_gs_cols:
            ax_u_title = fig.add_subplot(gs[0, unseen_gallery_title_center_gs_col_idx])
            ax_u_title.set_title("Unseen Gallery", fontsize=10, pad=15, fontweight='bold', loc='center')
            ax_u_title.axis('off')

        mixed_gallery_start_gs_col_idx_for_title = unseen_gallery_start_gs_col_idx_for_title + num_retrieved + 1  # 跳过间隔列
        mixed_gallery_title_center_gs_col_idx = mixed_gallery_start_gs_col_idx_for_title + (num_retrieved // 2)
        if num_retrieved > 0 and mixed_gallery_title_center_gs_col_idx < num_gs_cols:
            ax_m_title = fig.add_subplot(gs[0, mixed_gallery_title_center_gs_col_idx])
            ax_m_title.set_title("Mixed Gallery", fontsize=10, pad=15, fontweight='bold', loc='center')
            ax_m_title.axis('off')

    for i, query_domain_name in enumerate(query_domain_names_vis):
        print(f"  Processing Query Domain: {query_domain_name}...")
        current_gs_col_idx = 0

        ax_domain_label = fig.add_subplot(gs[i, current_gs_col_idx])
        ax_domain_label.text(0.5, 0.5, query_domain_name, ha='center', va='center', fontsize=9);
        ax_domain_label.axis('off')
        current_gs_col_idx += cols_per_domain_label

        query_candidate_splits = \
            data_splits_provider_func(args, query_domain_name, 0, tr_classes, va_classes, te_classes)['te']
        # print(query_candidate_splits)
        matching_query_paths = [
            pth for pth in query_candidate_splits
            if pth.split(os.sep)[-2].lower() == target_class_name_vis.lower()
        ]
        query_img_path_vis = matching_query_paths[index]
        if not query_img_path_vis:
            print(f"    Warning: No query image for '{target_class_name_vis}' in '{query_domain_name}'.")
            ax_q_placeholder = fig.add_subplot(gs[i, current_gs_col_idx])
            ax_q_placeholder.text(0.5, 0.5, "N/A", ha='center', va='center');
            ax_q_placeholder.axis('off')
            current_gs_col_idx += cols_per_query_image
            current_gs_col_idx += 1
            for _ in range(num_retrieved):
                if current_gs_col_idx < num_gs_cols:
                    ax_placeholder = fig.add_subplot(gs[i, current_gs_col_idx]);
                    ax_placeholder.text(0.5, 0.5, "-");
                    ax_placeholder.axis('off');
                    current_gs_col_idx += 1
            current_gs_col_idx += 1
            for _ in range(num_retrieved):
                if current_gs_col_idx < num_gs_cols:
                    ax_placeholder = fig.add_subplot(gs[i, current_gs_col_idx]);
                    ax_placeholder.text(0.5, 0.5, "-");
                    ax_placeholder.axis('off');
                    current_gs_col_idx += 1
            continue

        query_img_pil = Image.open(query_img_path_vis).convert("RGB")
        query_tensor = image_transforms_eval(query_img_pil).unsqueeze(0).to(device)
        query_cls_id_numeric_for_model = utils_module.numeric_classes([target_class_name_vis], te_dict_class)
        query_dom_id_numeric_for_model = utils_module.numeric_classes([query_domain_name], dict_doms)
        query_feat, _ = model.image_encoder(query_tensor, query_dom_id_numeric_for_model,
                                            query_cls_id_numeric_for_model, stage=4)

        ax_query = fig.add_subplot(gs[i, current_gs_col_idx])
        ax_query.imshow(query_img_pil, aspect='auto');
        ax_query.axis('off')
        current_gs_col_idx += cols_per_query_image

        current_gs_col_idx += 1

        for j, gallery_type_vis in enumerate(["Unseen", "Mixed"]):
            if j == 1:
                current_gs_col_idx += 1

            print(f"    Building {gallery_type_vis} Gallery from '{gallery_domain_name_vis}'...")
            include_seen_in_gallery = 1 if gallery_type_vis == "Mixed" else 0
            gallery_splits = \
                data_splits_provider_func(args, gallery_domain_name_vis, include_seen_in_gallery, tr_classes,
                                          va_classes,
                                          te_classes)['te']

            gallery_group_start_temp_gs_idx = current_gs_col_idx

            if not gallery_splits:
                print(f"    Warning: Gallery for {gallery_domain_name_vis} ({gallery_type_vis}) is empty.")
                for k_ax_offset in range(num_retrieved):
                    if gallery_group_start_temp_gs_idx + k_ax_offset < num_gs_cols:
                        ax_g_placeholder = fig.add_subplot(gs[i, gallery_group_start_temp_gs_idx + k_ax_offset])
                        ax_g_placeholder.text(0.5, 0.5, "N/A", ha='center', va='center');
                        ax_g_placeholder.axis('off')
                current_gs_col_idx += num_retrieved
                continue

            gallery_dataset_vis = BaselineDataset_class(np.array(gallery_splits), transforms=image_transforms_eval)
            gallery_loader_vis = DataLoader(gallery_dataset_vis, batch_size=args.batch_size * 5, shuffle=False,
                                            num_workers=args.num_workers, pin_memory=True)
            gallery_feats_list_vis, gallery_labels_str_list_vis = [], []
            for gal_batch_imgs, gal_batch_cls_str, gal_batch_dom_str in tqdm(gallery_loader_vis,
                                                                             desc=f'Ext. Gallery ({gallery_type_vis})',
                                                                             leave=False,
                                                                             total=len(gallery_loader_vis)):
                gal_batch_imgs = gal_batch_imgs.float().to(device)
                gal_cls_id_numeric_for_model = utils_module.numeric_classes(gal_batch_cls_str, te_dict_class)
                gal_dom_id_numeric_for_model = utils_module.numeric_classes(gal_batch_dom_str, dict_doms)
                gal_feat_batch, _ = model.image_encoder(gal_batch_imgs, gal_dom_id_numeric_for_model,
                                                        gal_cls_id_numeric_for_model, stage=4)
                gallery_feats_list_vis.append(gal_feat_batch)
                gallery_labels_str_list_vis.extend(list(gal_batch_cls_str))

            if not gallery_feats_list_vis:
                print(f"    Warning: No features for {gallery_domain_name_vis} ({gallery_type_vis}).")
                for k_ax_offset in range(num_retrieved):
                    if gallery_group_start_temp_gs_idx + k_ax_offset < num_gs_cols:
                        ax_g_placeholder = fig.add_subplot(gs[i, gallery_group_start_temp_gs_idx + k_ax_offset])
                        ax_g_placeholder.text(0.5, 0.5, "N/A", ha='center', va='center');
                        ax_g_placeholder.axis('off')
                current_gs_col_idx += num_retrieved
                continue

            gallery_feats_all_vis = torch.cat(gallery_feats_list_vis, dim=0)
            similarities = torch.matmul(query_feat, gallery_feats_all_vis.T).squeeze(0)
            actual_k = min(num_retrieved, len(gallery_labels_str_list_vis))

            if actual_k == 0:
                for k_ax_offset in range(num_retrieved):
                    if gallery_group_start_temp_gs_idx + k_ax_offset < num_gs_cols:
                        ax_g_placeholder = fig.add_subplot(gs[i, gallery_group_start_temp_gs_idx + k_ax_offset])
                        ax_g_placeholder.text(0.5, 0.5, "N/A", ha='center', va='center');
                        ax_g_placeholder.axis('off')
                current_gs_col_idx += num_retrieved
                continue

            top_k_scores, top_k_indices = torch.topk(similarities, k=actual_k, largest=True)

            for k in range(num_retrieved):
                retrieved_image_gs_idx_in_group = gallery_group_start_temp_gs_idx + k
                if retrieved_image_gs_idx_in_group >= num_gs_cols: continue

                ax_retrieved = fig.add_subplot(gs[i, retrieved_image_gs_idx_in_group])
                ax_retrieved.axis('off')
                if k < actual_k:
                    retrieved_idx_in_gallery = top_k_indices[k].item()
                    retrieved_img_path = gallery_splits[retrieved_idx_in_gallery]
                    retrieved_label_str = gallery_labels_str_list_vis[retrieved_idx_in_gallery]
                    retrieved_img_pil = Image.open(retrieved_img_path).convert("RGB")
                    ax_retrieved.imshow(retrieved_img_pil, aspect='auto')
                    is_correct = (retrieved_label_str.lower() == target_class_name_vis.lower())
                    border_color = 'green' if is_correct else 'red'
                    rect = patches.Rectangle((0, 0), 1, 1, linewidth=3.0, edgecolor=border_color, facecolor='none',
                                             transform=ax_retrieved.transAxes, clip_on=False, zorder=10)
                    ax_retrieved.add_patch(rect)
                else:
                    ax_retrieved.text(0.5, 0.5, "-", ha='center', va='center', fontsize=10, fontweight='bold')
            current_gs_col_idx += num_retrieved

    sep1_gs_col_idx = cols_per_domain_label + cols_per_query_image

    sep2_gs_col_idx = cols_per_domain_label + cols_per_query_image + 1 + num_retrieved  # 1是第一个间隔列

    line_y_start, line_y_end = 0.05, 0.91


    norm_gs_widths = np.array(gs_width_ratios) / sum(gs_width_ratios)
    norm_gs_boundaries = np.cumsum(np.insert(norm_gs_widths, 0, 0))

    sep1_left_norm = norm_gs_boundaries[sep1_gs_col_idx]
    sep1_right_norm = norm_gs_boundaries[sep1_gs_col_idx + 1]
    sep1_center_norm = (sep1_left_norm + sep1_right_norm) / 2.0

    fig.subplots_adjust(left=0.03, right=0.98, top=0.90, bottom=0.05, wspace=0.1, hspace=0.4)  # 初始调整

    fig_plot_width = fig.subplotpars.right - fig.subplotpars.left
    sep1_x_fig = fig.subplotpars.left + sep1_center_norm * fig_plot_width
    fig.add_artist(
        plt.Line2D([sep1_x_fig, sep1_x_fig], [line_y_start, line_y_end], color='black', lw=1.5, linestyle='--'))

    sep2_left_norm = norm_gs_boundaries[sep2_gs_col_idx]
    sep2_right_norm = norm_gs_boundaries[sep2_gs_col_idx + 1]
    sep2_center_norm = (sep2_left_norm + sep2_right_norm) / 2.0
    sep2_x_fig = fig.subplotpars.left + sep2_center_norm * fig_plot_width
    fig.add_artist(
        plt.Line2D([sep2_x_fig, sep2_x_fig], [line_y_start, line_y_end], color='black', lw=1.5, linestyle='--'))

    fig.subplots_adjust(
        left=0.03,
        right=0.98,
        top=0.90,
        bottom=0.05,
        wspace=0.05,
        hspace=0.35
    )

    save_path = os.path.join('retrieval',
                             f"retrieval_vis_{args.dataset}_{index}_{target_class_name_vis.replace(' ', '_')}.png")
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"  Visualization saved to: {save_path}")
    plt.show()


def make_square_pil(pil_img, target_size, fill_color=(255, 255, 255, 0)):
    w, h = pil_img.size
    if w == h: return pil_img.resize((target_size, target_size),
                                     Image.Resampling.LANCZOS) if w != target_size else pil_img
    nw, nh = (target_size, int(target_size * h / w)) if w > h else (int(target_size * w / h), target_size)
    rsz_img = pil_img.resize((nw, nh), Image.Resampling.LANCZOS)
    mode = 'RGBA' if len(fill_color) == 4 else 'RGB'
    if rsz_img.mode != mode and mode == 'RGBA':
        rsz_img = rsz_img.convert('RGBA')
    elif rsz_img.mode != 'RGB' and mode == 'RGB':
        if rsz_img.mode == 'RGBA':
            tmp_bg = Image.new('RGB', (nw, nh), fill_color[:3] if len(fill_color) == 3 else (255, 255, 255))
            try:
                tmp_bg.paste(rsz_img, (0, 0), rsz_img)
            except ValueError:
                alpha = rsz_img.split()[-1]; tmp_bg.paste(rsz_img, (0, 0), alpha)
            rsz_img = tmp_bg
        else:
            rsz_img = rsz_img.convert('RGB')
    new_img = Image.new(mode, (target_size, target_size), fill_color)
    px, py = (target_size - nw) // 2, (target_size - nh) // 2
    new_img.paste(rsz_img, (px, py));
    return new_img


@torch.no_grad()
def visualize_quickdraw_retrieval_per_class(
        model, args, te_dict_class, dict_doms, device, image_transforms_eval,
        target_class_names_vis, query_domain_name_fixed, gallery_domain_name_vis,
        tr_classes, va_classes, te_classes, data_splits_provider_func,
        BaselineDataset_class, utils_module, num_retrieved=10, q_idx_to_use_for_query=0):
    model.eval()
    print(f"Vis: classes={target_class_names_vis}, query_idx={q_idx_to_use_for_query}")
    num_cats = len(target_class_names_vis)
    if num_cats == 0: return

    N_CAT_LBL, N_Q_IMG, N_SEP, N_GAL_IMG = 1, 1, 1, num_retrieved
    N_COLS = N_CAT_LBL + N_Q_IMG + N_SEP + N_GAL_IMG
    IMG_SZ_IN, PIL_SZ = 0.65, 256

    ratio_cat_label_rel_to_image = 1.8
    BASE_RATIO_FOR_IMAGE_CELL = 1.0
    ratio_separator_rel_to_image = 0.1

    ratios = []
    ratios.append(ratio_cat_label_rel_to_image * BASE_RATIO_FOR_IMAGE_CELL)
    ratios.append(BASE_RATIO_FOR_IMAGE_CELL)
    ratios.append(ratio_separator_rel_to_image * BASE_RATIO_FOR_IMAGE_CELL)
    for _ in range(N_GAL_IMG):
        ratios.append(BASE_RATIO_FOR_IMAGE_CELL)

    fig_w = sum(r * IMG_SZ_IN for r in ratios)
    fig_row_h = IMG_SZ_IN
    extra_h_inch = IMG_SZ_IN * 2.5
    fig_h = (num_cats * fig_row_h) + extra_h_inch

    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = GridSpec(num_cats, N_COLS, figure=fig, width_ratios=ratios,
                  wspace=0.15,
                  hspace=0.30,
                  left=0.03, right=0.97,
                  bottom=0.08,
                  top=0.85)

    title_fs = 9
    title_pad_points = 30

    if num_cats > 0:
        ax_tqc = fig.add_subplot(gs[0, 0])
        ax_tqc.set_title("Query Category", fontsize=title_fs, pad=title_pad_points, loc='center', fontweight='bold')
        ax_tqc.axis('off')


        g_sidx = N_CAT_LBL + N_Q_IMG + N_SEP
        g_cidx = 0
        if N_GAL_IMG > 0: g_cidx = g_sidx + (N_GAL_IMG // 2)
        if g_cidx < N_COLS:
            ax_tmg = fig.add_subplot(gs[0, g_cidx])
            ax_tmg.set_title("Mixed Gallery", fontsize=title_fs, pad=title_pad_points, loc='center',
                             fontweight='bold')
            ax_tmg.axis('off')

    axes_bounds = []
    for i, cls_name in enumerate(target_class_names_vis):
        gs_cidx = 0;
        print(f"  Row {i + 1}: {cls_name}")
        ax_cl = fig.add_subplot(gs[i, gs_cidx]);
        ax_cl.text(0.05, 0.5, cls_name.replace('_', ' ').title(), ha='left', va='center', fontsize=7);
        ax_cl.axis('off');
        axes_bounds.append(ax_cl);
        gs_cidx += N_CAT_LBL

        q_path, q_ok, q_feat = None, False, None
        q_cands = data_splits_provider_func(args, query_domain_name_fixed, 0, tr_classes, va_classes, te_classes)['te']
        q_match = [p for p in q_cands if p.split(os.sep)[-2].lower() == cls_name.lower()]
        if q_match: q_path = q_match[q_idx_to_use_for_query if q_idx_to_use_for_query < len(q_match) else 0]

        ax_q = fig.add_subplot(gs[i, gs_cidx]);
        ax_q.axis('off');
        axes_bounds.append(ax_q)
        if q_path:
            try:
                q_pil_o = Image.open(q_path).convert("RGB");
                q_pil_s = make_square_pil(q_pil_o, PIL_SZ)
                ax_q.imshow(q_pil_s, aspect='auto')
                q_tensor = image_transforms_eval(q_pil_o).unsqueeze(0).to(device)
                q_cid = utils_module.numeric_classes([cls_name], te_dict_class)
                q_did = utils_module.numeric_classes([query_domain_name_fixed], dict_doms)
                q_f, _ = model.image_encoder(q_tensor, q_did, q_cid, stage=4)
                if q_f is not None and q_f.numel() > 0:
                    q_feat, q_ok = q_f, True
                else:
                    ax_q.text(0.5, 0.5, "F_Err", ha='center', va='center', fontsize=4)
            except Exception:
                ax_q.text(0.5, 0.5, "Q_Err", ha='center', va='center', fontsize=4)
        else:
            ax_q.text(0.5, 0.5, "NoPth", ha='center', va='center', fontsize=4)
        gs_cidx += N_Q_IMG;
        gs_cidx += N_SEP

        gal_start_cidx = gs_cidx
        if not q_ok:
            for k_ph in range(N_GAL_IMG):
                ph_cidx = gal_start_cidx + k_ph
                if ph_cidx < N_COLS: ax_gph = fig.add_subplot(gs[i, ph_cidx]);ax_gph.text(0.5, 0.5, "-");ax_gph.axis(
                    'off');axes_bounds.append(ax_gph)
            gs_cidx += N_GAL_IMG;
            continue

        gal_s_all = data_splits_provider_func(args, gallery_domain_name_vis, 1, tr_classes, va_classes, te_classes)[
            'te']
        if not gal_s_all:
            for k_ph in range(N_GAL_IMG):
                ph_cidx = gal_start_cidx + k_ph
                if ph_cidx < N_COLS: ax_gph = fig.add_subplot(gs[i, ph_cidx]);ax_gph.text(0.5, 0.5, "G_NA");ax_gph.axis(
                    'off');axes_bounds.append(ax_gph)
            gs_cidx += N_GAL_IMG;
            continue

        gal_dset = BaselineDataset_class(np.array(gal_s_all), transforms=image_transforms_eval)
        gal_load = DataLoader(gal_dset, batch_size=args.batch_size * 2, shuffle=False, num_workers=args.num_workers)
        gal_f_lst, gal_l_lst, gal_p_proc = [], [], []
        for bn, (g_img_b, g_cls_b, g_dom_b) in enumerate(
                tqdm(gal_load, desc=f"Gal {cls_name[:5]}", leave=False, ncols=70)):
            s_idx_b = bn * gal_load.batch_size;
            g_img_b = g_img_b.to(device)
            g_cid_b = utils_module.numeric_classes(g_cls_b, te_dict_class);
            g_did_b = utils_module.numeric_classes(g_dom_b, dict_doms)
            try:
                g_feat_b, _ = model.image_encoder(g_img_b, g_did_b, g_cid_b, stage=4)
                gal_f_lst.append(g_feat_b);
                gal_l_lst.extend(list(g_cls_b))
                for k_b in range(len(g_cls_b)): gal_p_proc.append(gal_s_all[s_idx_b + k_b])
            except Exception as e_gf:
                print(f"Err gal feat: {e_gf}");continue

        if not gal_f_lst:
            for k_ph in range(N_GAL_IMG):
                ph_cidx = gal_start_cidx + k_ph
                if ph_cidx < N_COLS: ax_gph = fig.add_subplot(gs[i, ph_cidx]);ax_gph.text(0.5, 0.5, "F_NA");ax_gph.axis(
                    'off');axes_bounds.append(ax_gph)
            gs_cidx += N_GAL_IMG;
            continue

        gal_f_tensor = torch.cat(gal_f_lst, dim=0);
        gal_sims = torch.matmul(q_feat, gal_f_tensor.T).squeeze(0)
        gal_ret_idx_proc = []
        if gal_sims.numel() > 0:
            k_topk_g = min(N_GAL_IMG, gal_f_tensor.shape[0])
            if k_topk_g > 0: _, topk_g_t_idx = torch.topk(gal_sims,
                                                          k=k_topk_g);gal_ret_idx_proc = topk_g_t_idx.cpu().tolist()

        for k_disp_g in range(N_GAL_IMG):
            plot_g_idx = gal_start_cidx + k_disp_g
            if plot_g_idx < N_COLS:
                ax_g_ret = fig.add_subplot(gs[i, plot_g_idx]);
                ax_g_ret.axis('off');
                axes_bounds.append(ax_g_ret)
                if k_disp_g < len(gal_ret_idx_proc):
                    idx_proc = gal_ret_idx_proc[k_disp_g]
                    if idx_proc < len(gal_p_proc) and idx_proc < len(gal_l_lst):
                        g_p_ret = gal_p_proc[idx_proc];
                        g_l_ret = gal_l_lst[idx_proc]
                        try:
                            g_pil_o_ret = Image.open(g_p_ret).convert("RGB");
                            g_pil_s_ret = make_square_pil(g_pil_o_ret, PIL_SZ)
                            ax_g_ret.imshow(g_pil_s_ret, aspect='auto')
                            g_corr = (g_l_ret.lower() == cls_name.lower());
                            g_bdr_c = 'green' if g_corr else 'red'
                            g_rect = patches.Rectangle((0, 0), 1, 1, lw=1.8, edgecolor=g_bdr_c, facecolor='none',
                                                       transform=ax_g_ret.transAxes, clip_on=False)
                            ax_g_ret.add_patch(g_rect)
                        except Exception:
                            ax_g_ret.text(0.5, 0.5, "R_Err", ha='center', va='center', fontsize=4)
                    else:
                        ax_g_ret.text(0.5, 0.5, "Idx!", ha='center', va='center', fontsize=4)
                else:
                    ax_g_ret.text(0.5, 0.5, "-", ha='center', va='center', fontsize=10, fw='bold')
        gs_cidx += N_GAL_IMG

    if num_cats > 0 and len(axes_bounds) > 0:
        v_axes = [ax for ax in axes_bounds if
                  ax.get_figure() is fig and hasattr(ax, 'get_position') and ax.get_position() is not None]
        if v_axes:
            all_y0 = [ax.get_position().y0 for ax in v_axes];
            all_y1 = [ax.get_position().y1 for ax in v_axes]
            cmin_y, cmax_y = min(all_y0), max(all_y1);
            marg_r, c_h = 0.03, cmax_y - cmin_y
            ly_s, ly_e = cmin_y + marg_r * c_h, cmax_y - marg_r * c_h
            sep_cidx_line = N_CAT_LBL + N_Q_IMG
            if sep_cidx_line < N_COLS:
                r_sep_ax = min(num_cats // 2, num_cats - 1)
                if r_sep_ax >= 0:
                    try:
                        tmp_ax_s = fig.add_subplot(gs[r_sep_ax, sep_cidx_line])
                        lx_c = tmp_ax_s.get_position().x0 + tmp_ax_s.get_position().width / 2.0;
                        tmp_ax_s.remove()
                        fig.add_artist(plt.Line2D([lx_c, lx_c], [ly_s, ly_e], color='black', lw=1.0, ls='--',
                                                  transform=fig.transFigure))
                    except Exception as e_ln:
                        print(f"Warn: Sep line err: {e_ln}")
        else:
            print("Warn: No valid axes for sep line.")

    sub_y = 0.01;
    fig.text(0.5, sub_y, f"(b) Retr. Results In Query Domain \"{query_domain_name_fixed.title()}\"", ha='center',
             va='bottom', fontsize=7)

    s_dir = f'retrieval_vis_final_q{q_idx_to_use_for_query}_more_title_space'  # New save dir
    os.makedirs(s_dir, exist_ok=True)
    f_cls_part = "_".join(target_class_names_vis[:1]).replace(' ', '_')
    f_name = f"qdraw_{args.dataset}_{f_cls_part}_q{q_idx_to_use_for_query}.png"
    s_path = os.path.join(s_dir, f_name)
    try:
        plt.savefig(s_path, dpi=300);print(f"  Vis saved: {s_path}")
    except Exception as e_sv:
        print(f"ERR save fig '{s_path}': {e_sv}")
    plt.show(block=False);
    plt.pause(0.1);
    plt.close(fig)


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    # Parse options
    args = Options().parse()
    print('Parameters:\t' + str(args))
    main(args)
