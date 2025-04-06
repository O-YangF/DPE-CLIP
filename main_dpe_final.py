import random
import argparse
import wandb
from tqdm import tqdm
from datetime import datetime
from copy import deepcopy
import math

import torch
import torch.nn.functional as F
import operator
import torch.nn as nn
from info_nce import InfoNCE
from sklearn.manifold import TSNE 
import matplotlib.pyplot as plt

import clip
from utils import *

import open_clip

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'



def get_arguments():
    """Get arguments of the test-time adaptation."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config', required=True, help='settings of DPE on specific dataset in yaml format.')
    parser.add_argument('--wandb-log', dest='wandb', action='store_true', help='Whether you want to log to wandb. Include this flag to enable logging.')
    parser.add_argument('--datasets', dest='datasets', type=str, required=True, help="Datasets to process, separated by a slash (/). Example: I/A/V/R/S")
    parser.add_argument('--data-root', dest='data_root', type=str, default='./dataset/', help='Path to the datasets directory. Default is ../data/')
    parser.add_argument('--backbone', dest='backbone', type=str, choices=['RN50', 'ViT-B/16', 'SigLIP', 'OpenCLIP'], required=True, help='CLIP model backbone to use: RN50 or ViT-B/16.')
    parser.add_argument('--coop', dest='coop', action='store_true', help='Whether you want to use CoOp weights for initialization.')

    args = parser.parse_args()

    return args

def InfoNCELoss(A, B):
    loss = InfoNCE(temperature=0.01, reduction='mean')
    return loss(A, B)

def update_cache(cache, pred, features_loss, shot_capacity, monitor=None, include_prob_map=False, similarity_threshold=0.9):
    """
    更新缓存，结合熵值和相似度进行更精细的替换。
    Args:
        cache: 当前缓存 (dict: class_id -> list of [feature, float_entropy, Optional[prob_map]])
        pred: 当前样本的预测类别
        features_loss: 包含 [图像特征, float_entropy, Optional[概率图]] 的列表
        shot_capacity: 每个类别缓存的最大容量
        monitor: CacheMonitor实例，用于记录
        include_prob_map: 是否在缓存项中包含概率图 (用于负缓存)
        similarity_threshold: 用于判断样本是否高度相似的余弦相似度阈值
    """
    # 根据之前的分析，加上 no_grad 更安全
    with torch.no_grad():
        # 准备要缓存的项
        item_to_cache = features_loss if not include_prob_map else features_loss[:2] + [features_loss[2]]
        new_feature = item_to_cache[0]
        new_loss = item_to_cache[1] # 这个已经是 float 了

        if pred not in cache:
            cache[pred] = []

        if len(cache[pred]) < shot_capacity:
            cache[pred].append(item_to_cache)
            if monitor:
                # 直接传递 float
                monitor.record(None, pred, None, new_loss)
        else:
            higher_loss_group = [] # 存储 (index, sample)
            lower_loss_group = []  # 存储 (index, sample)

            # 1. 分离缓存为两组
            for idx, existing_item in enumerate(cache[pred]):
                # 直接比较 float
                if existing_item[1] > new_loss:
                    higher_loss_group.append((idx, existing_item))
                else:
                    lower_loss_group.append((idx, existing_item))

            # 2. 检查 higher_loss_group 中是否存在高相似度样本
            highly_similar_in_higher = [] # 存储 (index, sample, similarity)
            for idx, existing_item in higher_loss_group:
                # similarity 是 Tensor，这里保留 .item()
                similarity = F.cosine_similarity(new_feature.squeeze(0), existing_item[0].squeeze(0), dim=0).item()
                if similarity > similarity_threshold:
                    highly_similar_in_higher.append((idx, existing_item, similarity))

            # Action A: 如果找到高相似度样本，替换其中熵最高的那个
            if highly_similar_in_higher:
                # 找到这些高相似度样本中熵最高的那个
                # 在 key 函数中直接使用 float 比较
                best_candidate_tuple = max(highly_similar_in_higher, key=lambda x: x[1][1])

                target_idx = best_candidate_tuple[0]
                item_to_replace = best_candidate_tuple[1]

                # 直接获取 float，移除 .item()
                old_entropy = item_to_replace[1]
                # new_entropy = new_loss # 已经是 float 了

                # 执行替换
                cache[pred][target_idx] = item_to_cache

                if monitor:
                    # 直接传递 float
                    monitor.record(pred, pred, old_entropy, new_loss)
                # 替换发生，结束此样本的处理
                return cache

            # 3. 如果 Action A 未执行，检查 lower_entropy_group 中是否存在高相似度样本
            found_high_similarity_in_lower = False
            for idx, existing_item in lower_loss_group:
                 # similarity 是 Tensor，这里保留 .item()
                similarity = F.cosine_similarity(new_feature.squeeze(0), existing_item[0].squeeze(0), dim=0).item()
                if similarity > similarity_threshold:
                    found_high_similarity_in_lower = True
                    break # 找到一个就足够判断

            # Action B: 如果找到，直接返回，不做任何更改
            if found_high_similarity_in_lower:
                # 不做任何操作，因为新样本与一个已经很好的样本太相似
                return cache

            # 4. 如果 Action A 和 B 都未执行 (即新样本与现有样本都不高度相似)
            # Action C: 检查 higher_entropy_group 是否为空。如果不为空，替换其中熵最高的那个样本
            if higher_loss_group: # 确保有比新样本更差的样本存在
                 # 找到 higher_entropy_group 中熵最高的样本
                 # 在 key 函数中直接使用 float 比较
                target_idx, item_to_replace = max(higher_loss_group, key=lambda x: x[1][1]) # x[1]是sample, x[1][1]是loss (float)

                # 直接获取 float
                old_entropy = item_to_replace[1]
                # new_entropy = new_loss # 已经是 float 了

                # 执行替换
                cache[pred][target_idx] = item_to_cache

                if monitor:
                     # 直接传递 float
                    monitor.record(pred, pred, old_entropy, new_loss)
            # else: # 如果 higher_entropy_group 为空，意味着新样本不比缓存中任何样本更好，则不执行任何操作

    return cache
    
def visualize_cache(cache, iter):
    # t-SNE visualization of cache features
    with torch.no_grad():
        cache_features = []
        cache_labels = []
        for class_index in sorted(cache.keys()):
            for item in cache[class_index]:
                cache_features.append(item[0].reshape(-1))
                cache_labels.append(class_index)
        cache_features = torch.stack(cache_features, dim=0)
        cache_labels = torch.Tensor(cache_labels).to(torch.int64)
        cache_features = F.normalize(cache_features, dim=1)
        cache_features = cache_features.cpu().numpy()
        cache_labels = cache_labels.cpu().numpy()
        tsne = TSNE(n_components=2)
        print(cache_features.shape)
        cache_features_fit = tsne.fit_transform(cache_features)
        
        # Assign different colors to different cache_labels
        colors = [
            '#00429d',  # Strong Blue
            '#93003a',  # Deep Red
            '#007d34',  # Vivid Green
            '#ff6800',  # Vivid Orange
            '#e30022',  # Bright Red
            '#a6bdd7',  # Light Periwinkle
            '#ffcc00',  # Vivid Yellow
            '#540d6e',  # Dark Violet
            '#7f180d',  # Dark Red
            '#00939c',  # Cyan Process
            '#5f3c99',  # Purplish Blue
            '#ff4a46',  # Bright Red-Orange
            '#8f0075',  # Strong Purple
            '#ff3c38',  # Bright Red
            '#83a697',  # Muted Cyan
            '#1e96be',  # Strong Cyan
            '#d9e021',  # Vivid Lime Green
            '#f18d05',  # Rich Orange
            '#f6e120',  # Bright Yellow
            '#8f2d56',  # Strong Rose
            '#006837',  # Dark Green
            '#e7298a',  # Bright Pink
            '#ce1256',  # Dark Pink
            '#01665e',  # Dark Teal
            '#dfc27d',  # Pale Gold
            '#35978f',  # Muted Teal
            '#bf812d',  # Mustard Brown
            '#543005',  # Dark Brown
            '#8c510a',  # Light Brown
            '#80cdc1',  # Soft Turquoise
        ]
        colors_others = 'gray'
        figure, ax = plt.subplots(1, 1, dpi=600, figsize=(5, 5))
        patch = ax.patch
        patch.set_color("#f5f5f5")
        ax.tick_params(axis='both',          # Changes apply to both x and y axes
               which='both',         # Apply changes to both major and minor ticks
               bottom=False,         # No ticks along the bottom edge
               top=False,            # No ticks along the top edge
               left=False,           # No ticks along the left edge
               right=False,          # No ticks along the right edge
               labelbottom=False,    # No labels along the bottom edge
               labelleft=False)      # No labels along the left edge
        plt.grid(color='w', zorder=0, linewidth=2)
        plt.gca().spines['bottom'].set_color('gray')
        plt.gca().spines['left'].set_color('gray')
        plt.gca().spines['top'].set_color('gray')
        plt.gca().spines['right'].set_color('gray')
        # In Food-101, we have 101 classes
        for i in range(101):
            if i < 30:
                plt.scatter(cache_features_fit[cache_labels == i, 0], cache_features_fit[cache_labels == i, 1], c=colors[i], s=15, marker='x', zorder=5)
            else:
                plt.scatter(cache_features_fit[cache_labels == i, 0], cache_features_fit[cache_labels == i, 1], c=colors_others, s=5, zorder=5)
        save_path = 'fig/cache_features_iter_{}.png'.format(iter)
        plt.savefig(save_path)
        plt.close()
        

def cache_key_value(image_features, cache, alpha, beta, clip_weights):
    """Compute logits using positive/negative cache."""
    with torch.no_grad():
        cache_keys = []
        cache_values = []
        all_classes = []
        for class_index in sorted(cache.keys()):
            num_items = len(cache[class_index])
            # Compute the prototype of the class
            image_prototype = torch.zeros_like(image_features)
            for item in cache[class_index]:
                image_prototype += item[0] / num_items
            cache_keys.append(image_prototype)
            cache_values.append(class_index)
            all_classes.append(class_index)

        cache_keys = torch.cat(cache_keys, dim=0).permute(1, 0)
        cache_values = (F.one_hot(torch.Tensor(cache_values).to(torch.int64), num_classes=clip_weights.size(1))).cuda().half()
            
        return cache_keys, cache_values, all_classes
    
def compute_cache_logits(image_features, cache_keys, cache_values, alpha, beta, clip_weights):
    affinity = image_features @ cache_keys
    cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
    return alpha * cache_logits
    


class TextResidue(nn.Module):
    def __init__(self, clip_weights, rank=8):
        super(TextResidue, self).__init__()
        self.feat_dim, self.cate_num = clip_weights.shape
        
        self.rank = rank 
        dtype = clip_weights.dtype 
        device = clip_weights.device

        # --- 定义低秩矩阵 U 和 V (替换了原来的 residual) ---
        # U 初始化
        self.U = nn.Parameter(torch.empty(self.feat_dim, self.rank, dtype=dtype, device=device))
        nn.init.kaiming_uniform_(self.U, a=math.sqrt(5)) 

        # V 初始化为零
        self.V = nn.Parameter(torch.zeros(self.cate_num, self.rank, dtype=dtype, device=device))
        
    def forward(self, x):
        # --- 计算低秩残差 U @ V.T ---
        low_rank_residual = self.U @ self.V.T 

        # --- 应用残差并归一化 ---
        new_clip_weights = x.clone() + low_rank_residual 
        new_clip_weights = F.normalize(new_clip_weights, dim=0) 
        return new_clip_weights
    
    def reset(self):
        # --- 重置 V 为零 ---
        nn.init.zeros_(self.V)
        # 重新初始化 U
        nn.init.kaiming_uniform_(self.U, a=math.sqrt(5))
        



class PositiveCacheResidue(nn.Module):
    def __init__(self, pos_cache_keys, rank):
        super(PositiveCacheResidue, self).__init__()
        self.feat_dim, self.cache_size = pos_cache_keys.shape 
        # rank 现在是必需的，直接存储
        self.rank = rank
        dtype = pos_cache_keys.dtype
        device = pos_cache_keys.device

        # --- 定义低秩矩阵 U_cache 和 V_cache (替换了原来的 residual) ---
        self.U_cache = nn.Parameter(torch.empty(self.feat_dim, self.rank, dtype=dtype, device=device))
        nn.init.kaiming_uniform_(self.U_cache, a=math.sqrt(5))

        self.V_cache = nn.Parameter(torch.zeros(self.cache_size, self.rank, dtype=dtype, device=device))
        
    def forward(self, x):
        # --- 计算低秩残差 U_cache @ V_cache.T ---
        low_rank_residual = self.U_cache @ self.V_cache.T 

        # --- 应用残差并归一化 ---
        new_pos_cache_keys = x.clone() + low_rank_residual
        new_pos_cache_keys = F.normalize(new_pos_cache_keys, dim=0) 
        return new_pos_cache_keys

    def reset(self):
        # --- 重置 V_cache 为零 ---
        nn.init.zeros_(self.V_cache)
        # 重新初始化 U_cache
        nn.init.kaiming_uniform_(self.U_cache, a=math.sqrt(5))



class SmoothCrossEntropy(nn.Module):
    def __init__(self, alpha=0.0):
        super(SmoothCrossEntropy, self).__init__()
        self.alpha = alpha

    def forward(self, logits, labels):
        num_classes = logits.shape[-1]
        alpha_div_k = self.alpha / num_classes
        target_probs = F.one_hot(labels, num_classes=num_classes).float() * \
            (1. - self.alpha) + alpha_div_k
        loss = -(target_probs * torch.log_softmax(logits, dim=-1)).sum(dim=-1)
        return loss.mean()

def run_test_dpe(pos_cfg, lr_cfg, loader, clip_model, clip_weights, dataset_name):   
    with torch.cuda.amp.autocast():
        pos_cache, accuracies = {}, []
        
        # Unpack all hyperparameters
        pos_enabled = pos_cfg['enabled']
        


        try:
            rank = pos_cfg['low_rank']['rank']
            print(f"Mandatory Low Rank Adaptation enabled with rank = {rank}")
            if not isinstance(rank, int) or rank <= 0:
                 raise ValueError(f"Configured rank must be a positive integer. Found: {rank}")
        except KeyError:
            # 如果配置中确实找不到 rank，则抛出更明确的错误
            raise KeyError("Mandatory parameter 'rank' not found in configuration under 'positive: low_rank: rank'. Please add it to the YAML file.")






        if pos_enabled:
            pos_params = {k: pos_cfg[k] for k in ['shot_capacity', 'alpha', 'beta']}
        
        clip_weights_global = clip_weights.clone()
        num_avg = 0
        total = len(loader)
        
        losses = []
        all_clip_weights = []
        distances = []

        # Test-time adaptation
        for i, (images, target) in enumerate(tqdm(loader, desc='Processed test images: ')):
            clip_weights_local = clip_weights_global.clone().detach()
            text_residue = TextResidue(clip_weights_local, rank=rank)

            # (获取初始 logits) - 注意这里可能需要调用一次 text_residue 来获取初始调整后的权重
            initial_adapted_weights = text_residue(clip_weights_local).detach() # 获取初始状态，不计算梯度
            image_features_x, clip_logits, entropy, prob_map, pred = get_clip_logits(images, clip_model, initial_adapted_weights)
            target = target.cuda()
            
            pos_cache_residue = None # 初始化
            pos_cache_keys = None    # 初始化
            pos_cache_values = None  # 初始化
            all_classes = []         # 初始化

            if pos_enabled:
                entropy = get_entropy(entropy, clip_weights)
                update_cache(pos_cache, pred, [image_features_x, entropy], pos_params['shot_capacity'])

                # 检查缓存是否非空再计算 key-value 和实例化 PositiveCacheResidue
                if pred in pos_cache and pos_cache[pred]: # 确保缓存中有内容
                    # 注意：cache_key_value 可能需要修改以适应低秩，但目前看它只依赖 image_features_x 和 cache 内容
                    # 这里假设 cache_key_value 返回的 pos_cache_keys 形状正确
                    pos_cache_keys, pos_cache_values, all_classes = cache_key_value(image_features_x, pos_cache, pos_params['alpha'], pos_params['beta'], clip_weights)

                    if pos_cache_keys is not None and pos_cache_keys.numel() > 0:
                         # --- 实例化 PositiveCacheResidue (低秩模式)，传递必需的 rank ---
                         pos_cache_residue = PositiveCacheResidue(pos_cache_keys, rank=rank)
                    # else: pos_cache_residue 保持为 None

            steps = 1 # Update step, set to 1 in default
            for j in range(steps):
                new_clip_weights = text_residue(clip_weights_local)
                final_logits = clip_logits.clone()

                current_pos_cache_residue = None # 重置以捕获本次迭代的实例
                
                if pos_enabled and pos_cache:
                    new_pos_cache_keys = pos_cache_residue(pos_cache_keys)

                    current_pos_cache_residue = pos_cache_residue # 记录供优化器使用

                    final_logits += compute_cache_logits(image_features_x, new_pos_cache_keys, pos_cache_values, pos_params['alpha'], pos_params['beta'], clip_weights)
                    loss = avg_entropy(final_logits)

                    if all_classes: # 确保对齐损失的输入有效
                         image2text_loss = InfoNCELoss(new_pos_cache_keys.T, new_clip_weights[:, all_classes].T)
                         loss += image2text_loss * lr_cfg['align']
                else:
                    # 如果没有启用或没有缓存，损失基于 new_clip_weights 计算的 logits (需要重新计算或调整逻辑)
                    # 为保持与原代码相似性，如果不用缓存，则基于初始 logits 计算熵？ 或基于 new_clip_weights 计算？
                    # 假设基于 new_clip_weights 更合理
                    _, current_clip_logits, _, _, _ = get_clip_logits(images, clip_model, new_clip_weights)
                    loss = avg_entropy(current_clip_logits) # 基于当前文本适应后的 logits 计算熵
                
                lr_text = lr_cfg['text']
                lr_image = lr_cfg['image']


                # --- 优化器部分无需修改 ---
                params_to_optimize = [{'params': text_residue.parameters(), 'lr': lr_text, 'eps': 1e-3, 'weight_decay': 1e-1}]
                if pos_enabled and current_pos_cache_residue is not None:
                    params_to_optimize.append({'params': current_pos_cache_residue.parameters(), 'lr': lr_image, 'eps': 1e-3, 'weight_decay': 1e-1})

                if params_to_optimize: # 确保有参数可优化
                    optimizer = torch.optim.AdamW(params_to_optimize)
                    optimizer.zero_grad()
                    if j == steps - 1:
                        loss.backward()
                    else:
                        loss.backward(retain_graph=True) # 如果 steps > 1 可能需要
                    optimizer.step()

            # Actual inference
            text_residue.eval()
            final_pos_cache_residue = None

            if pos_enabled and pos_cache:
                pos_cache_residue.eval()
                final_pos_cache_residue = pos_cache_residue

            with torch.no_grad():
                final_clip_weights = text_residue(clip_weights_local)

                # (获取推理 logits)
                if dataset_name == 'A':
                    image_features, clip_logits_inf, _, _, _ = get_clip_logits(images, clip_model, final_clip_weights)
                else:
                    img_input = images[0] if isinstance(images, list) else images
                    image_features, clip_logits_inf, _, _, _ = get_clip_logits(img_input, clip_model, final_clip_weights)

                final_logits_inf = clip_logits_inf.clone()


                if pos_enabled and final_pos_cache_residue is not None and pos_cache_keys is not None:
                    # --- 使用最终优化后的 pos_cache_residue ---
                    final_pos_cache_keys = final_pos_cache_residue(pos_cache_keys)
                    final_logits_inf += compute_cache_logits(image_features, final_pos_cache_keys, pos_cache_values, pos_params['alpha'], pos_params['beta'], clip_weights)   
                    
                acc = cls_acc(final_logits_inf, target.cuda())
                accuracies.append(acc)
                if wandb.run is not None:
                    wandb.log({"Averaged test accuracy": sum(accuracies)/len(accuracies)}, commit=True)
                
                final_loss_inf = avg_entropy(final_logits_inf)
                
                # Global update step: textual prototype evolution
                if get_entropy(final_loss_inf, clip_weights) < 0.1:
                    num_avg += 1
                    clip_weights_global = clip_weights_global * (num_avg / (num_avg + 1)) + final_clip_weights * (1 / (num_avg + 1))
                    # clip_weights_global = F.normalize(clip_weights_global, dim=0)

            if i % 1000 == 0:
                print("---- DPE's test accuracy: {:.2f}. ----\n".format(sum(accuracies)/len(accuracies)))

        print("---- DPE's test accuracy: {:.2f}. ----\n".format(sum(accuracies)/len(accuracies)))
        return sum(accuracies)/len(accuracies)

def main():
    args = get_arguments()
    config_path = args.config

    # Initialize CLIP model
    if args.backbone == 'RN50' or args.backbone == 'ViT-B/16':
        clip_model, preprocess = clip.load(args.backbone)
    elif args.backbone == 'SigLIP':
        clip_model, preprocess = create_model_from_pretrained('hf-hub:timm/ViT-SO400M-14-SigLIP')
    elif args.backbone == 'OpenCLIP':
        clip_model, _, preprocess = open_clip.create_model_and_transforms('hf-hub:laion/CLIP-ViT-L-14-laion2B-s32B-b82K')
        clip_model = clip_model.to('cuda')

    if args.wandb:
        date = datetime.now().strftime("%b%d_%H-%M-%S")
        group_name = f"{args.backbone}_{args.datasets}_{date}"
    
    # Run DPE on each dataset
    datasets = args.datasets.split('/')
    for dataset_name in datasets:
        # Set random seed
        random.seed(1)
        torch.manual_seed(1)
        print(f"Processing {dataset_name} dataset.")
        
        cfg = get_config_file(config_path, dataset_name)
        print("\nRunning dataset configurations:")
        print(cfg, "\n")
        print(args.coop)
        print(args.backbone)
        
        test_loader, classnames, template, cupl_path = build_test_data_loader(dataset_name, args.data_root, preprocess)
        clip_weights = clip_classifier(classnames, template, cupl_path, clip_model, args.coop, args.backbone)

        if args.wandb:
            run_name = f"{dataset_name}"
            run = wandb.init(project="20250405-DPE", config=cfg, group=group_name, name=run_name)

        acc = run_test_dpe(cfg['positive'], cfg['learning_rate'], test_loader, clip_model, clip_weights, dataset_name)

        if args.wandb:
            wandb.log({f"{dataset_name}": acc})
            run.finish()

if __name__ == "__main__":
    main()