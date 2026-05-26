import os
import pickle

import math
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import MPNetModel, MPNetConfig, AutoTokenizer, AutoModel, BertModel, BertTokenizerFast
from tqdm import tqdm
import json
from dataloader import create_dataloader
import argparse
import torch.nn.functional as F
# tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
import pickle
import os
import numpy as np
# 仅在初始化PCA投影时需要sklearn
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

class r0TrainableMultimodalLaBSE(nn.Module):

    def __init__(self,
                 initial_alpha=0.88,  # 将融合权重作为参数
                 image_feature_path="/Data/ljm/HLMEA-main/HLMEA-main/dataset_1011/pkls/zh_en_GA_id_img_feature_dict.pkl",
                 ent_ids_path="/Data/ljm/HLMEA-main/HLMEA-main/dataset/DBP15K_ZH_EN/ent_ids_merged",
                 max_length=512,
                 use_pca_projection=True):  # 添加一个开关以决定是否使用PCA
        super().__init__()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.alpha = nn.Parameter(torch.tensor(initial_alpha), requires_grad=False)

        self._uri_to_id = self.load_ent_ids(ent_ids_path)


        # --- 加载核心模型 ---
        print("Loading LaBSE model...")
        self.labse = BertModel.from_pretrained("models/labse")
        self.labse.to(self.device)

        # --- 数据加载 ---
        print(f"Loading image features from {image_feature_path}...")
        self._image_features_dict = self._load_pickle(image_feature_path)
        if not self._image_features_dict:
            print("Warning: Image features dictionary is empty.")
            self._image_feature_dim = 2048  # 默认值
        else:
            self._image_features_dict = {k: torch.from_numpy(v).to(self.device) for k, v in self._image_features_dict.items()}
            self._image_feature_dim = next(iter(self._image_features_dict.values())).shape[0]

        self._text_feature_dim = self.labse.config.hidden_size

        if self._image_features_dict:
            print("Calculating the mean of all image features for missing value imputation...")
            all_features = torch.stack(list(self._image_features_dict.values()))
            self.mean_image_feature = torch.mean(all_features, dim=0)
            print("Mean image feature calculated.")
        else:
            # 如果没有任何图像特征，则回退到使用零向量
            self.mean_image_feature = torch.zeros(self._image_feature_dim, device=self.device)

        # --- 投影层初始化 ---
        if use_pca_projection:
            print("Initializing projection layer using PCA...")
            self.image_projection = self._create_pca_projection()
        else:
            print("Initializing projection layer with random weights...")
            self.image_projection = nn.Linear(self._image_feature_dim, self._text_feature_dim)

        self.image_projection.to(self.device)

        # --- 设置为验证模式 ---
        self.eval()
        for param in self.parameters():
            param.requires_grad = False
        print("Model is set to evaluation mode. All gradients are turned off.")

    def load_ent_ids(self,file_path):
        """加载 ent_ids 文件，返回 URI 到序号的映射字典"""
        uri_to_id = {}
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')  # 按制表符分割
                if len(parts) == 2:
                    ent_id, uri = parts
                    uri_to_id[uri] = int(ent_id)  # 转换为整数
        return uri_to_id

    def _get_image_batch_features(self, entity_ids):
        # ... (此方法现在使用修改后的填充策略和已在GPU上的Tensor字典) ...
        feature_vectors = []
        has_image_flags = []

        for eid in entity_ids:
            xuhao = self._uri_to_id.get(eid)

            if xuhao is not None and xuhao in self._image_features_dict:

                has_image_flags.append(True)
                feature_vector = self._image_features_dict[xuhao]
            else:
                has_image_flags.append(False)
                feature_vector = self.mean_image_feature  # 使用平均特征填充

            feature_vectors.append(feature_vector)

        image_features_tensor = torch.stack(feature_vectors)
        has_image_mask = torch.tensor(has_image_flags, device=self.device)

        return image_features_tensor, has_image_mask



    def _load_pickle(self, path):
        try:
            with open(path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Error loading pickle file {path}: {e}")
            return {}

    def _create_pca_projection(self):
        if not self._image_features_dict:
            return nn.Linear(self._image_feature_dim, self._text_feature_dim)

        all_features = np.stack([v.cpu().numpy() for v in self._image_features_dict.values()])
        print(f"Fitting PCA on {all_features.shape[0]} image features...")
        pca = PCA(n_components=self._text_feature_dim, random_state=42)
        pca.fit(all_features)
        print("PCA fitting complete.")

        projection_layer = nn.Linear(self._image_feature_dim, self._text_feature_dim, bias=False)
        projection_layer.weight.data = torch.tensor(pca.components_, dtype=torch.float32)
        return projection_layer

    def forward(self, entity_ids, text_inputs):
        """
        修改后的前向传播逻辑，使用新的图像特征获取方式
        entity_ids: 实体URI列表
        text_inputs: LaBSE tokenizer的输出
        """
        # 1. 获取文本特征
        text_outputs = self.labse(**text_inputs)
        text_embeds = text_outputs.pooler_output

        # 2. 获取并投影图像特征
        image_feats, _ = self._get_image_batch_features(entity_ids)
        projected_image_embeds = self.image_projection(image_feats)

        # 3. 融合前先归一化
        norm_text_embeds = F.normalize(text_embeds, p=2, dim=1)
        norm_image_embeds = F.normalize(projected_image_embeds, p=2, dim=1)

        # 4. 加权融合
        fused_embeds = (self.alpha * norm_text_embeds) + ((1 - self.alpha) * norm_image_embeds)

        return fused_embeds


class CrossAttentionFusion(nn.Module):
    """
    Cross-Attention module to fuse text and image embeddings.
    Text is Query, Image is Key+Value.
    """
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim*4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim*4, embed_dim)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, text_emb, image_emb):
        text_seq = text_emb.unsqueeze(1)
        image_seq = image_emb.unsqueeze(1)

        attn_output, _ = self.cross_attention(query=text_seq, key=image_seq, value=image_seq)
        x = self.norm1(text_emb + self.dropout1(attn_output.squeeze(1)))
        ffn_output = self.ffn(x)
        fused_output = self.norm2(x + self.dropout2(ffn_output))
        return fused_output  # (B, embed_dim)


class r1TrainableMultimodalLaBSE(nn.Module):
    """
    最终优化版的多模态LaBSE模型。
    - 使用基于相似度的门控进行图文融合。
    - 包含用于自适应损失加权的可学习参数。
    """

    def __init__(self,
                 image_feature_path="/Data/ljm/HLMEA-main/HLMEA-main/dataset_1011/pkls/zh_en_GA_id_img_feature_dict.pkl",
                 ent_ids_path="/Data/ljm/HLMEA-main/HLMEA-main/dataset/DBP15K_ZH_EN/ent_ids_merged",
                 embed_dim=768,
                 dropout: float = 0.3,
                 temperature: float = 0.07):  # 增加了Dropout以防止过拟合
        super().__init__()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.temperature = temperature
        # --- 模型核心组件 ---
        self.labse = BertModel.from_pretrained("models/labse")
        # self.image_projection = nn.Linear(2048, embed_dim)

        # self.image_projection = nn.Sequential(
        #     nn.Linear(512, 768),
        #     # nn.ReLU(),
        #     # nn.Dropout(0.4),
        #     # nn.Linear(1024, 768)
        # )

        self._image_feature_path = image_feature_path
        self._image_features_dict, self.mean_image_feature_cpu, self.std_image_feature_cpu, self._image_feature_dim = self._load_image_features_and_stats()

        self.image_projection = nn.Sequential(
            nn.Linear(2048, embed_dim * 2),  # 放大特征
            nn.GELU(),  # 引入非线性
            nn.Dropout(0.3),  # 增加正则化
            nn.Linear(embed_dim * 2, embed_dim),  # 映射回目标维度
            nn.LayerNorm(embed_dim)  # !!! 关键：增加层归一化
        )

        self.fusion_module = CrossAttentionFusion(
            embed_dim=embed_dim,
            num_heads=8,
            dropout=dropout
        )

        self.adaptive_gate_network = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 4, 1),
            nn.Sigmoid()
        )
        self.adaptive_gate_network[-2].bias.data.fill_(2.0)

        self.lambda_gate_reg = 1.0  # 加强正则

        self.log_var_main = nn.Parameter(torch.tensor(0.0))
        self.log_var_contrastive = nn.Parameter(torch.tensor(0.0))

        self._ent_ids_path = ent_ids_path
        self._uri_to_id = self.load_ent_ids(ent_ids_path)
        self.last_fusion_weights = None

    def _load_image_features_and_stats(self):
        """
        ### MODIFIED (v2) ###
        加载图像特征, 计算并返回:
        1. 特征字典 (dict)
        2. 平均特征向量 (torch.Tensor on CPU)
        3. 标准差向量 (torch.Tensor on CPU)
        4. 特征维度 (int)
        """
        print(f"正在加载图像特征文件: {self._image_feature_path}")
        if not os.path.exists(self._image_feature_path):
            raise FileNotFoundError(f"图像特征文件未找到: {self._image_feature_path}")

        with open(self._image_feature_path, 'rb') as f:
            features_dict_np = pickle.load(f)

        if not features_dict_np:
            print("警告：图像特征文件为空。将使用2048维的 N(0, 1) 进行采样。")
            dim = 2048  # 回退到默认维度
            mean_feature = torch.zeros(dim, dtype=torch.float32)
            std_feature = torch.ones(dim, dtype=torch.float32)  # 使用 1 作为标准差
            return {}, mean_feature, std_feature, dim

        features_dict = {k: torch.tensor(v, dtype=torch.float32) for k, v in features_dict_np.items()}

        # 计算平均特征和维度
        all_features = torch.stack(list(features_dict.values()), dim=0)

        # --- MODIFIED: 计算 Mean 和 Std ---
        mean_feature = torch.mean(all_features, dim=0)
        # 加上一个很小的 epsilon 防止 std 为 0
        std_feature = torch.std(all_features, dim=0) + 1e-6
        # --- END MODIFIED ---

        dim = mean_feature.shape[0]

        print(f"图像特征加载完成。特征维度: {dim}，已计算平均值和标准差。")
        return features_dict, mean_feature, std_feature, dim

    def load_ent_ids(self,file_path):
        """加载 ent_ids 文件，返回 URI 到序号的映射字典"""
        uri_to_id = {}
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')  # 按制表符分割
                if len(parts) == 2:
                    ent_id, uri = parts
                    uri_to_id[uri] = int(ent_id)  # 转换为整数
        return uri_to_id

    def _get_image_batch_features(self, entity_ids):
        """
        ### MODIFIED (v2) ###
        - 图像缺失时，从 N(mean, std) 分布中随机采样一个向量进行填充。
        - 仍然返回 has_image_mask，供辅助损失函数 (info_nce_loss) 使用。
        """
        feature_vectors_on_device = []
        has_image_flags = []  # 仍然需要这个掩码
        image_features_dict_cpu = self._image_features_dict

        # 使用在 __init__ 中计算好的统计数据
        mean_feature_cpu = self.mean_image_feature_cpu
        std_feature_cpu = self.std_image_feature_cpu

        uri_to_id = self._uri_to_id

        for eid in entity_ids:
            xuhao = uri_to_id.get(eid)
            if xuhao is not None and xuhao in image_features_dict_cpu:
                # print(777777777)
                feature_cpu = image_features_dict_cpu[xuhao]
                has_image_flags.append(True)  # 标记为有真实图像
            else:
                # --- MODIFIED: 从 N(mean, std) 随机采样 ---
                # print(8888888)
                feature_cpu = torch.normal(mean_feature_cpu, std_feature_cpu)
                # feature_cpu = torch.zeros(self._image_feature_dim, dtype=torch.float32)
                # --- END MODIFIED ---
                has_image_flags.append(False)  # 标记为使用随机填充

            feature_gpu = feature_cpu.to(device=self.device)  # 直接在to()中指定device
            feature_vectors_on_device.append(feature_gpu)

        image_features_tensor = torch.stack(feature_vectors_on_device)
        has_image_mask = torch.tensor(has_image_flags, device=self.device)

        return image_features_tensor, has_image_mask

    def forward(self, entity_ids, input_ids, attention_mask):
        # 1. 获取文本嵌入
        text_outputs = self.labse(input_ids=input_ids, attention_mask=attention_mask)
        text_embeddings = text_outputs.pooler_output


        image_emb, has_image_mask = self._get_image_batch_features(entity_ids)
        image_embeddings = self.image_projection(image_emb)

        # 3. 统一融合（无分支）
        #    - 真实图像: fused_increment = fusion(text, real_image_proj)
        #    - 缺失图像: fused_increment = fusion(text, random_image_proj)
        # fused_increment = self.fusion_module(text_embeddings, image_embeddings)

        # Cross-Attention 融合
        fused_increment = self.fusion_module(text_embeddings, image_embeddings)
        fused_increment = fused_increment * 2.0  # 输出缩放，增强门控梯度



        gate_input = torch.cat([text_embeddings, fused_increment], dim=-1)
        fusion_gate = self.adaptive_gate_network(gate_input)
        self.last_fusion_weights = fusion_gate


        output_embeddings = text_embeddings + fusion_gate * fused_increment

        # 门控正则
        gate_reg_loss = self.lambda_gate_reg * ((fusion_gate.mean() - 0.7) ** 2)  # 偏向使用图像

        return output_embeddings, text_embeddings, image_embeddings, has_image_mask, gate_reg_loss




def pairwise_margin_loss(embeddings_anchor, embeddings_positive, embeddings_negative, margin=1.0):
    """主任务：对比学习损失"""
    positive_distance = (embeddings_anchor - embeddings_positive).pow(2).sum(1)
    negative_distance = (embeddings_anchor - embeddings_negative).pow(2).sum(1)
    losses = torch.relu(positive_distance - negative_distance + margin)
    return losses.mean()

def info_nce_loss(image_features, text_features, temperature):
    """标准 InfoNCE"""
    image_features = F.normalize(image_features, p=2, dim=-1)
    text_features = F.normalize(text_features, p=2, dim=-1)

    logits_per_image = torch.matmul(image_features, text_features.t()) / temperature
    logits_per_text = logits_per_image.t()
    labels = torch.arange(len(logits_per_image), device=logits_per_image.device)
    return (F.cross_entropy(logits_per_image, labels) + F.cross_entropy(logits_per_text, labels)) / 2



def tokenize_and_convert_to_tensor(tokenizer: str, texts, max_length=512, device=None):
    # if 'e5' in ptm_model_name:
    #     tokenizer = AutoTokenizer.from_pretrained('intfloat/multilingual-e5-large')
    # elif 'labse' in ptm_model_name:
    #     tokenizer = BertTokenizerFast.from_pretrained("models/labse")
    # elif 'mpnet' in ptm_model_name:
    #     tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
    # elif 'minilm' in ptm_model_name:
    #     tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    # else:
    #     raise RuntimeError('Invalid PTM model name %s for tokenizer' % ptm_model_name)
    ## Tokenize all texts at once, ensuring consistent padding
    try:
        inputs = tokenizer(texts, padding=True, truncation=True, return_tensors='pt', max_length=max_length)
    except Exception as e:
        print('Error:', e)
    input_ids, attention_mask = inputs['input_ids'], inputs['attention_mask']
    if device:
        input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
    return input_ids, attention_mask


# evaluate 函数也需要修改以适应新的模型输出格式
def evaluate(tokenizer, model, dataloader, device):
    """评估函数，修正了模型输出的处理方式"""
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # ... (数据加载部分与train函数相同) ...
            text_entity, pos_entity, neg_entity = batch['text_entity'], batch['pos_entity'], batch['neg_entity']
            text_desc, desc_pos, desc_neg = batch['text_description'], batch['description_pos'], batch[
                'description_neg']
            text_ids, text_mask = tokenize_and_convert_to_tensor(tokenizer, text_desc, device=device)
            pos_ids, pos_mask = tokenize_and_convert_to_tensor(tokenizer, desc_pos, device=device)
            neg_ids, neg_mask = tokenize_and_convert_to_tensor(tokenizer, desc_neg, device=device)

            final_anchor, _, _, _, _ = model(text_entity, text_ids, text_mask)
            final_pos, _, _, _, _ = model(pos_entity, pos_ids, pos_mask)
            final_neg, _, _, _, _ = model(neg_entity, neg_ids, neg_mask)

            loss = pairwise_margin_loss(final_anchor, final_pos, final_neg)
            total_loss += loss.item()

    model.train()
    return total_loss / len(dataloader) if len(dataloader) > 0 else 0


def train(ptm_name,ptm_model, train_dataloader, dev_dataloader, optimizer, device,
          epochs=50, patience=3, save_path='best_model.pth'):
    ptm_model.train()
    ptm_model.to(device)
    best_dev_loss = float('inf')
    no_improvement = 0
    tokenizer = BertTokenizerFast.from_pretrained("models/labse")

    for epoch in range(epochs):
        total_main_loss, total_contrastive_loss,total_gate_reg= 0, 0, 0
        weight_contrastive = min(1.0, (epoch + 1) / 50.0)
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{epochs}"):
            optimizer.zero_grad()

            text_entity, pos_entity, neg_entity = batch['text_entity'], batch['pos_entity'], batch['neg_entity']
            text_desc, desc_pos, desc_neg = batch['text_description'], batch['description_pos'], batch[
                'description_neg']
            text_ids, text_mask = tokenize_and_convert_to_tensor(tokenizer, text_desc, device=device)
            pos_ids, pos_mask = tokenize_and_convert_to_tensor(tokenizer, desc_pos, device=device)
            neg_ids, neg_mask = tokenize_and_convert_to_tensor(tokenizer, desc_neg, device=device)

            # 前向
            final_anchor, txt_anchor, img_anchor, has_img_mask, gate_reg_loss = ptm_model(text_entity, text_ids,
                                                                                          text_mask)
            final_pos, _, _, _, _ = ptm_model(pos_entity, pos_ids, pos_mask)
            final_neg, _, _, _, _ = ptm_model(neg_entity, neg_ids, neg_mask)

            # 主任务三元组损失
            main_loss = pairwise_margin_loss(final_anchor, final_pos, final_neg)

            # 对所有样本 InfoNCE (缺失图像弱权重)
            contrastive_loss_full = info_nce_loss(img_anchor, txt_anchor, ptm_model.temperature)
            contrastive_loss_real = info_nce_loss(img_anchor[has_img_mask], txt_anchor[has_img_mask],
                                                  ptm_model.temperature) if has_img_mask.sum() > 1 else torch.tensor(
                0.0, device=device)
            contrastive_loss = 0.1 * contrastive_loss_full + 0.9 * contrastive_loss_real

            # 自适应权重
            loss_main_w = main_loss * torch.exp(-ptm_model.log_var_main) + ptm_model.log_var_main
            loss_contrastive_w = contrastive_loss * torch.exp(
                -ptm_model.log_var_contrastive) + ptm_model.log_var_contrastive

            total_loss = 0.5 * loss_main_w + weight_contrastive * loss_contrastive_w + gate_reg_loss
            total_loss.backward()
            optimizer.step()

            total_main_loss += main_loss.item()
            total_contrastive_loss += contrastive_loss.item()
            total_gate_reg += gate_reg_loss.item()

            # 评估
        dev_loss = evaluate(tokenizer, ptm_model, dev_dataloader, device)

        print(f"Epoch {epoch + 1}: Train Main Loss={total_main_loss / len(train_dataloader):.4f}, "
              f"Contrastive Loss={total_contrastive_loss / len(train_dataloader):.4f}, "
              f"Gate Reg Loss={total_gate_reg / len(train_dataloader):.4f}, Dev Loss={dev_loss:.4f}")

        if ptm_model.last_fusion_weights is not None:
            with torch.no_grad():
                weights = ptm_model.last_fusion_weights.cpu()
                print(f"Gate Stats -> Mean: {weights.mean():.4f}, Std: {weights.std():.4f}, "
                      f"Min: {weights.min():.4f}, Max: {weights.max():.4f}")

        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            no_improvement = 0
            torch.save(ptm_model.state_dict(), save_path)
            print(f"Model saved at epoch {epoch + 1} with Dev Loss: {dev_loss:.4f}")
        else:
            no_improvement += 1
            if no_improvement >= patience:
                print(f"No improvement for {patience} epochs. Stopping early.")
                break


def test_gate_distribution_triplet(model, dataloader, device, save_dir):
    """
    Q3 实验代码：验证 Cross-Attention 和 Gating 的协同机制
    修改版：增加了数据保存功能
    """
    print("\n" + "="*50)
    print("Running Q3 Experiment: Gate Distribution Analysis")
    print("="*50)
    
    model.eval()
    # 注意：请确保 BertTokenizerFast 已正确导入
    tokenizer = BertTokenizerFast.from_pretrained("models/labse")
    
    gates_real = []
    gates_fill = []
    gates_noise = []
    
    target_count = 100 
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Analysis Experiment"):
            if len(gates_real) >= target_count and len(gates_fill) >= target_count:
                break
                
            text_entity = batch['text_entity'] 
            text_desc = batch['text_description']
            
            # Tokenize
            text_ids, text_mask = tokenize_and_convert_to_tensor(tokenizer, text_desc, device=device)
            
            # 1. 正常前向传播
            text_outputs = model.labse(input_ids=text_ids, attention_mask=text_mask)
            text_embeddings = text_outputs.pooler_output
            
            image_emb_raw, has_image_mask = model._get_image_batch_features(text_entity)
            image_embeddings = model.image_projection(image_emb_raw)
            
            # 2. 计算当前 Batch 的门控值
            fused_inc = model.fusion_module(text_embeddings, image_embeddings)
            fused_inc = fused_inc * 2.0
            
            gate_input = torch.cat([text_embeddings, fused_inc], dim=-1)
            gate_values = model.adaptive_gate_network(gate_input)
            
            # 3. 分流收集数据
            batch_gates = gate_values.cpu().numpy().flatten()
            batch_mask = has_image_mask.cpu().numpy().astype(bool)
            
            # 收集真实图片
            if len(gates_real) < target_count:
                gates_real.extend(batch_gates[batch_mask])
            
            # 收集填充图片
            if len(gates_fill) < target_count:
                gates_fill.extend(batch_gates[~batch_mask])
                
            # 4. 构造高斯噪声对照组
            if batch_mask.sum() > 0:
                real_indices = torch.where(has_image_mask)[0]
                noise_feats = torch.randn_like(image_embeddings[real_indices]) 
                
                fused_inc_noise = model.fusion_module(text_embeddings[real_indices], noise_feats)
                fused_inc_noise = fused_inc_noise * 2.0
                
                gate_in_noise = torch.cat([text_embeddings[real_indices], fused_inc_noise], dim=-1)
                gate_val_noise = model.adaptive_gate_network(gate_in_noise)
                
                gates_noise.extend(gate_val_noise.cpu().numpy().flatten())

    # 截断数据
    gates_real = gates_real[:target_count]
    gates_fill = gates_fill[:target_count]
    gates_noise = gates_noise[:target_count]
    
    print(f"Stats - Real Mean: {np.mean(gates_real):.4f}, Noise Mean: {np.mean(gates_noise):.4f}")

    # ==========================================
    # [新增部分] 保存数据到本地 JSON 文件
    # ==========================================
    data_save_path = os.path.join(save_dir, "Q3_experiment_data.json")
    
    # Numpy float 类型不能直接被 JSON 序列化，需要转为 Python float
    data_to_save = {
        "gates_real": [float(x) for x in gates_real],
        "gates_fill": [float(x) for x in gates_fill],
        "gates_noise": [float(x) for x in gates_noise]
    }
    
    with open(data_save_path, 'w', encoding='utf-8') as f:
        json.dump(data_to_save, f, ensure_ascii=False, indent=4)
        
    print(f"Data saved successfully to: {data_save_path}")
    # ==========================================

    # 原有的绘图代码（可保留用于实时查看，也可注释掉）
    plt.figure(figsize=(10, 6))
    plt.hist(gates_real, bins=30, alpha=0.5, label=f'Real Images (Mean: {np.mean(gates_real):.2f})', color='green')
    if len(gates_fill) > 0:
        plt.hist(gates_fill, bins=30, alpha=0.5, label=f'Filled Images (Mean: {np.mean(gates_fill):.2f})', color='orange')
    plt.hist(gates_noise, bins=30, alpha=0.5, label=f'Gaussian Noise (Mean: {np.mean(gates_noise):.2f})', color='red')
    
    plt.title("Adaptive Gating Response (Q3 Experiment)")
    plt.xlabel("Gate Value")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    plot_path = os.path.join(save_dir, "Q3_gate_analysis.png")
    plt.savefig(plot_path)
    print(f"Experimental plot saved to: {plot_path}")
    plt.close()