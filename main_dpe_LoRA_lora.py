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

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'  # 内存优化

def get_arguments():
    """Get arguments of the test-time adaptation."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config', required=True, help='settings of DPE on specific dataset in yaml format.')
    parser.add_argument('--wandb-log', dest='wandb', action='store_true', help='Whether to log to wandb.')
    parser.add_argument('--datasets', dest='datasets', type=str, required=True, help="Datasets to process, separated by a slash (/). Example: I/A/V/R/S")
    parser.add_argument('--data-root', dest='data_root', type=str, default='./dataset/', help='Path to the datasets directory.')
    parser.add_argument('--backbone', dest='backbone', type=str, choices=['RN50', 'ViT-B/16', 'SigLIP', 'OpenCLIP'], required=True, help='CLIP model backbone.')
    parser.add_argument('--coop', dest='coop', action='store_true', help='Whether to use CoOp weights for initialization.')
    args = parser.parse_args()
    return args

def InfoNCELoss(A, B):
    loss = InfoNCE(temperature=0.01, reduction='mean')
    return loss(A, B)

def update_cache(cache, pred, features_loss, shot_capacity, monitor=None, include_prob_map=False, similarity_threshold=0.9):
    """
    Update cache with new features and loss, using entropy and similarity-based conditions.
    Args:
        cache: dict of class_id -> list of [feature, float_entropy, Optional[prob_map]]
        pred: predicted class
        features_loss: list of [image_feature, float_entropy, Optional[prob_map]]
        shot_capacity: max capacity per class
        monitor: CacheMonitor instance for logging (optional)
        include_prob_map: whether to include prob_map in cache
        similarity_threshold: cosine similarity threshold from config
    """
    with torch.no_grad():
        item_to_cache = features_loss if not include_prob_map else features_loss[:2] + [features_loss[2]]
        new_feature, new_loss = item_to_cache[0], item_to_cache[1]

        if pred not in cache:
            cache[pred] = []

        if len(cache[pred]) < shot_capacity:
            cache[pred].append(item_to_cache)
            if monitor:
                monitor.record(None, pred, None, new_loss)
        else:
            higher_loss_group, lower_loss_group = [], []
            for idx, item in enumerate(cache[pred]):
                if item[1] > new_loss:
                    higher_loss_group.append((idx, item))
                else:
                    lower_loss_group.append((idx, item))

            highly_similar_in_higher = []
            for idx, item in higher_loss_group:
                similarity = F.cosine_similarity(new_feature.squeeze(0), item[0].squeeze(0), dim=0).item()
                if similarity > similarity_threshold:
                    highly_similar_in_higher.append((idx, item, similarity))

            if highly_similar_in_higher:
                target_idx, item_to_replace = max(highly_similar_in_higher, key=lambda x: x[1][1])[0:2]
                old_entropy = item_to_replace[1]
                cache[pred][target_idx] = item_to_cache
                if monitor:
                    monitor.record(pred, pred, old_entropy, new_loss)
                return cache

            if any(F.cosine_similarity(new_feature.squeeze(0), item[0].squeeze(0), dim=0).item() > similarity_threshold 
                   for _, item in lower_loss_group):
                return cache

            if higher_loss_group:
                target_idx, item_to_replace = max(higher_loss_group, key=lambda x: x[1][1])
                old_entropy = item_to_replace[1]
                cache[pred][target_idx] = item_to_cache
                if monitor:
                    monitor.record(pred, pred, old_entropy, new_loss)

        return cache

def visualize_cache(cache, iter):
    with torch.no_grad():
        cache_features = [item[0].reshape(-1) for class_index in sorted(cache.keys()) for item in cache[class_index]]
        cache_labels = [class_index for class_index in sorted(cache.keys()) for _ in cache[class_index]]
        cache_features = torch.stack(cache_features, dim=0)
        cache_labels = torch.tensor(cache_labels, dtype=torch.int64)
        cache_features = F.normalize(cache_features, dim=1).cpu().numpy()
        cache_labels = cache_labels.cpu().numpy()
        tsne = TSNE(n_components=2)
        print(cache_features.shape)
        cache_features_fit = tsne.fit_transform(cache_features)
        
        colors = ['#00429d', '#93003a', '#007d34', '#ff6800', '#e30022', '#a6bdd7', '#ffcc00', '#540d6e', '#7f180d', 
                  '#00939c', '#5f3c99', '#ff4a46', '#8f0075', '#ff3c38', '#83a697', '#1e96be', '#d9e021', '#f18d05', 
                  '#f6e120', '#8f2d56', '#006837', '#e7298a', '#ce1256', '#01665e', '#dfc27d', '#35978f', '#bf812d', 
                  '#543005', '#8c510a', '#80cdc1']
        colors_others = 'gray'
        fig, ax = plt.subplots(1, 1, dpi=600, figsize=(5, 5))
        ax.patch.set_color("#f5f5f5")
        ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, 
                       labelbottom=False, labelleft=False)
        plt.grid(color='w', zorder=0, linewidth=2)
        for spine in plt.gca().spines.values():
            spine.set_color('gray')
        for i in range(101):
            plt.scatter(cache_features_fit[cache_labels == i, 0], cache_features_fit[cache_labels == i, 1], 
                        c=colors[i] if i < 30 else colors_others, s=15 if i < 30 else 5, marker='x', zorder=5)
        plt.savefig(f'fig/cache_features_iter_{iter}.png')
        plt.close()

def cache_key_value(image_features, cache, alpha, beta, clip_weights):
    with torch.no_grad():
        cache_keys = [torch.zeros_like(image_features) for _ in range(len(cache))]
        cache_values = []
        all_classes = []
        for i, class_index in enumerate(sorted(cache.keys())):
            num_items = len(cache[class_index])
            for item in cache[class_index]:
                cache_keys[i] += item[0] / num_items
            cache_values.append(class_index)
            all_classes.append(class_index)
        cache_keys = torch.cat(cache_keys, dim=0).permute(1, 0)
        cache_values = F.one_hot(torch.tensor(cache_values, dtype=torch.int64), num_classes=clip_weights.size(1)).cuda().half()
        return cache_keys, cache_values, all_classes

def compute_cache_logits(image_features, cache_keys, cache_values, alpha, beta, clip_weights):
    affinity = image_features @ cache_keys
    cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
    return alpha * cache_logits

class TextResidue(nn.Module):
    def __init__(self, clip_weights, rank=16):
        super(TextResidue, self).__init__()
        self.feat_dim, self.cate_num = clip_weights.shape
        self.rank = rank
        dtype, device = clip_weights.dtype, clip_weights.device
        self.U = nn.Parameter(torch.empty(self.feat_dim, self.rank, dtype=dtype, device=device))
        nn.init.kaiming_uniform_(self.U, a=math.sqrt(5))
        self.V = nn.Parameter(torch.zeros(self.cate_num, self.rank, dtype=dtype, device=device))

    def forward(self, x):
        residual = self.U @ self.V.T
        new_clip_weights = x.clone() + residual
        return F.normalize(new_clip_weights, dim=0)

    def reset(self):
        nn.init.zeros_(self.V)
        nn.init.kaiming_uniform_(self.U, a=math.sqrt(5))

class PositiveCacheResidue(nn.Module):
    def __init__(self, pos_cache_keys, rank=16):
        super(PositiveCacheResidue, self).__init__()
        self.feat_dim, self.cache_size = pos_cache_keys.shape
        self.rank = rank
        dtype, device = pos_cache_keys.dtype, pos_cache_keys.device 
        self.U_cache = nn.Parameter(torch.empty(self.feat_dim, self.rank, dtype=dtype, device=device))
        nn.init.kaiming_uniform_(self.U_cache, a=math.sqrt(5))
        self.V_cache = nn.Parameter(torch.zeros(self.cache_size, self.rank, dtype=dtype, device=device))

    def forward(self, x):
        residual = self.U_cache @ self.V_cache.T
        new_pos_cache_keys = x.clone() + residual
        return F.normalize(new_pos_cache_keys, dim=0)

    def reset(self):
        nn.init.zeros_(self.V_cache)
        nn.init.kaiming_uniform_(self.U_cache, a=math.sqrt(5))

class SmoothCrossEntropy(nn.Module):
    def __init__(self, alpha=0.0):
        super(SmoothCrossEntropy, self).__init__()
        self.alpha = alpha

    def forward(self, logits, labels):
        num_classes = logits.shape[-1]
        alpha_div_k = self.alpha / num_classes
        target_probs = F.one_hot(labels, num_classes=num_classes).float() * (1. - self.alpha) + alpha_div_k
        return -(target_probs * torch.log_softmax(logits, dim=-1)).sum(dim=-1).mean()

def run_test_dpe(pos_cfg, lr_cfg, loader, clip_model, clip_weights, dataset_name):
    with torch.cuda.amp.autocast():
        pos_cache, accuracies = {}, []
        pos_enabled = pos_cfg['enabled']

        # Dynamic rank with default fallback
        rank = pos_cfg.get('low_rank', {}).get('rank', 16)
        if not isinstance(rank, int) or rank <= 0:
            print(f"Invalid rank {rank}, using default 16")
            rank = 16

        # Dynamic similarity threshold with default fallback
        similarity_threshold = pos_cfg.get('similarity_threshold', 0.9)
        if not isinstance(similarity_threshold, (int, float)) or similarity_threshold < 0 or similarity_threshold > 1:
            print(f"Invalid similarity_threshold {similarity_threshold}, using default 0.9")
            similarity_threshold = 0.9

        if pos_enabled:
            pos_params = {k: pos_cfg[k] for k in ['shot_capacity', 'alpha', 'beta']}

        clip_weights_global = clip_weights.clone()
        num_avg = 0

        for i, (images, target) in enumerate(tqdm(loader, desc='Processed test images: ')):
            clip_weights_local = clip_weights_global.clone().detach()
            text_residue = TextResidue(clip_weights_local, rank=rank)
            initial_weights = text_residue(clip_weights_local).detach()
            image_features_x, clip_logits, entropy, prob_map, pred = get_clip_logits(images, clip_model, initial_weights)
            target = target.cuda()

            pos_cache_residue = None
            pos_cache_keys, pos_cache_values, all_classes = None, None, []

            if pos_enabled:
                entropy = get_entropy(entropy, clip_weights)
                update_cache(pos_cache, pred, [image_features_x, entropy], pos_params['shot_capacity'], 
                             similarity_threshold=similarity_threshold)
                if pred in pos_cache and pos_cache[pred]:
                    pos_cache_keys, pos_cache_values, all_classes = cache_key_value(image_features_x, pos_cache, 
                                                                                   pos_params['alpha'], pos_params['beta'], clip_weights)
                    if pos_cache_keys is not None and pos_cache_keys.numel() > 0:
                        pos_cache_residue = PositiveCacheResidue(pos_cache_keys, rank=rank)

            steps = 1
            for j in range(steps):
                new_clip_weights = text_residue(clip_weights_local)
                final_logits = clip_logits.clone()
                current_pos_cache_residue = None

                if pos_enabled and pos_cache and pos_cache_residue:
                    new_pos_cache_keys = pos_cache_residue(pos_cache_keys)
                    current_pos_cache_residue = pos_cache_residue
                    final_logits += compute_cache_logits(image_features_x, new_pos_cache_keys, pos_cache_values, 
                                                        pos_params['alpha'], pos_params['beta'], clip_weights)
                    loss = avg_entropy(final_logits)
                    if all_classes:
                        loss += InfoNCELoss(new_pos_cache_keys.T, new_clip_weights[:, all_classes].T) * lr_cfg['align']
                else:
                    _, current_logits, _, _, _ = get_clip_logits(images, clip_model, new_clip_weights)
                    loss = avg_entropy(current_logits)

                lr_text, lr_image = lr_cfg['text'], lr_cfg['image']
                params_to_optimize = [{'params': text_residue.parameters(), 'lr': lr_text, 'eps': 1e-3, 'weight_decay': 1e-1}]
                if pos_enabled and current_pos_cache_residue:
                    params_to_optimize.append({'params': current_pos_cache_residue.parameters(), 'lr': lr_image, 
                                              'eps': 1e-3, 'weight_decay': 1e-1})
                optimizer = torch.optim.AdamW(params_to_optimize)
                optimizer.zero_grad()
                loss.backward() if j == steps - 1 else loss.backward(retain_graph=True)
                optimizer.step()

            text_residue.eval()
            final_pos_cache_residue = pos_cache_residue if pos_enabled and pos_cache else None
            with torch.no_grad():
                final_clip_weights = text_residue(clip_weights_local)
                img_input = images if dataset_name == 'A' else (images[0] if isinstance(images, list) else images)
                image_features, clip_logits_inf, _, _, _ = get_clip_logits(img_input, clip_model, final_clip_weights)
                final_logits_inf = clip_logits_inf.clone()
                if final_pos_cache_residue and pos_cache_keys is not None:
                    final_logits_inf += compute_cache_logits(image_features, final_pos_cache_residue(pos_cache_keys), 
                                                            pos_cache_values, pos_params['alpha'], pos_params['beta'], clip_weights)
                
                acc = cls_acc(final_logits_inf, target.cuda())
                accuracies.append(acc)
                if wandb.run:
                    wandb.log({"Averaged test accuracy": sum(accuracies) / len(accuracies)})

                final_loss = avg_entropy(final_logits_inf)
                if get_entropy(final_loss, clip_weights) < 0.1:
                    num_avg += 1
                    clip_weights_global = clip_weights_global * (num_avg / (num_avg + 1)) + final_clip_weights * (1 / (num_avg + 1))

            if i % 1000 == 0:
                print(f"---- DPE's test accuracy: {sum(accuracies)/len(accuracies):.2f} ----")

        print(f"---- DPE's final test accuracy: {sum(accuracies)/len(accuracies):.2f} ----")
        return sum(accuracies) / len(accuracies)

def main():
    args = get_arguments()
    config_path = args.config

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

    datasets = args.datasets.split('/')
    for dataset_name in datasets:
        random.seed(1)
        torch.manual_seed(1)
        print(f"Processing {dataset_name} dataset.")
        
        cfg = get_config_file(config_path, dataset_name)
        print("\nRunning dataset configurations:\n", cfg, "\n", args.coop, "\n", args.backbone)
        
        test_loader, classnames, template, cupl_path = build_test_data_loader(dataset_name, args.data_root, preprocess)
        clip_weights = clip_classifier(classnames, template, cupl_path, clip_model, args.coop, args.backbone)

        if args.wandb:
            run = wandb.init(project="20250406-DPE", config=cfg, group=group_name, name=dataset_name)

        acc = run_test_dpe(cfg['positive'], cfg['learning_rate'], test_loader, clip_model, clip_weights, dataset_name)

        if args.wandb:
            wandb.log({dataset_name: acc})
            run.finish()

if __name__ == "__main__":
    main()