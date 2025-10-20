import torch
from TextModel import Transformer
from transformers import logging, AutoTokenizer, AutoModel
from tqdm import tqdm
import os
import torch.nn as nn
NUM_CLASSES = 5  # 类别数量

import torch


def adjust_cls_feats(text_outputs, image_results):
    """
    调整 cls_feats 向量方向，使其略微偏向图像模型分类结果对应的类别向量
    :param text_outputs: 文本模型的输出
    :param image_results: 图像模型的分类结果
    :param alpha: 调整的权重列表，控制偏向的程度，每个元素对应一个分类
    :return: 调整后的 cls_feats
    """
    cls_feats = text_outputs['cls_feats']
    label_feats = text_outputs['label_feats']
    batch_size = cls_feats.size(0)

    # 提取图像模型分类结果对应的类别向量
    adjusted_cls_feats = []
    for i in range(batch_size):
        class_index = image_results[i]
        if class_index == 5:
            # 如果分类结果为 5，使用空向量
            adjusted_cls_feats.append(torch.zeros_like(cls_feats[i]))
        else:
            # 否则，提取对应类别的特征向量           
            adjusted_cls_feats.append(label_feats[i, class_index])

    adjusted_cls_feats = torch.stack(adjusted_cls_feats)

    return adjusted_cls_feats





import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CrossMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0, bias=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        
        # 线性变换层
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        
        self.dropout = dropout

    def forward(self, query, key_value, key_padding_mask=None, need_weights=True):
        """
        参数:
            query: 文本特征 [batch_size, tgt_len, embed_dim]
            key_value: 图像特征 [batch_size, src_len, embed_dim]
            key_padding_mask: 可选的key padding掩码 [batch_size, src_len]
            need_weights: 是否返回注意力权重
        """
        batch_size, tgt_len, embed_dim = query.size()
        # print(f"query.size(): {query.size()}")
        # print(f"key_value.size(): {key_value.size()}")
        src_len = key_value.size(1)
        
        # 线性投影
        q = self.q_proj(query)
        k = self.k_proj(key_value)
        v = self.v_proj(key_value)
        
        # 重塑为多头
        q = q.view(batch_size, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, src_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, src_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 缩放点积注意力
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        # 应用key padding掩码（如果提供）
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, src_len]
            attn_weights = attn_weights.masked_fill(key_padding_mask, float('-inf'))
        
        # 应用softmax获取注意力权重
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)
        
        # 应用注意力权重
        output = torch.matmul(attn_weights, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, tgt_len, embed_dim)
        
        # 输出投影
        output = self.out_proj(output)

        # print(f"output.size(): {output.size()}")
        
        if need_weights:
            # 平均多头注意力权重
            attn_weights = attn_weights.mean(dim=1)
            return output, attn_weights
        else:
            return output, None

# Add&Norm层
class AddNorm(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super(AddNorm, self).__init__()
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    """
    参数:
        x: 经过自注意力或前馈网络后的张量 [batch_size, seq_len, hidden_size]
        residual: 残差连接的输入张量 [batch_size, seq_len, hidden_size]
    返回:
        经过Add&Norm处理后的张量 [batch_size, seq_len, hidden_size]
    """
    def forward(self, x, sublayer_output):
        # 残差连接与层归一化
        return self.norm(x + self.dropout(sublayer_output))

# 前馈神经网络
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # 前馈网络实现
        return self.fc2(self.dropout(self.relu(self.fc1(x))))
class GatingNetwork(nn.Module):
    def __init__(self, input_dim, num_modalities=3):
        super(GatingNetwork, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_modalities)
        )

    def forward(self, x_cat):
        logits = self.linear(x_cat)                      # [batch_size, 3]
        weights = F.softmax(logits, dim=-1)              # 每行归一化
        return weights                                   # [batch_size, 3]

class GatingNetworkMoE(nn.Module):
    def __init__(self, input_dim, num_experts, k=2):
        super(GatingNetworkMoE, self).__init__()
        self.linear = nn.Linear(input_dim, num_experts)
        self.k = k
    def forward(self, x):
        # Step 1: Generate logits
        logits = self.linear(x)
        # Step 2: Add noise (optional)
        noise = torch.randn_like(logits) * 0.1  # 添加高斯噪声
        noisy_logits = logits + noise
        # Step 3: Top-K selection
        top_k_values, top_k_indices = torch.topk(noisy_logits, k=self.k, dim=-1)
        # Step 4: Apply Softmax
        weights = F.softmax(top_k_values, dim=-1)
        # Step 5: Create sparse weights
        sparse_weights = torch.zeros_like(logits)
        sparse_weights.scatter_(-1, top_k_indices, weights)
        return sparse_weights
class Expert(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Expert, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)             # 保持输入维度一致便于加权融合
        )

    def forward(self, x):
        return self.layer(x)                              # [batch_size, input_dim]
class MoE(nn.Module):
    def __init__(self, input_dim, num_experts=4, k=2, hidden_dim=128):
        super(MoE, self).__init__()
        self.num_experts = num_experts
        self.k = k
        self.experts = nn.ModuleList([Expert(input_dim, hidden_dim) for _ in range(num_experts)])
        self.gating = GatingNetworkMoE(input_dim, num_experts, k)

    def forward(self, x):
        # x: [batch_size, input_dim]
        gating_weights = self.gating(x)                             # [batch_size, num_experts]

        # 获取所有专家输出
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)  # [batch_size, num_experts, input_dim]

        # 加权求和
        gating_weights = gating_weights.unsqueeze(-1)               # [batch_size, num_experts, 1]
        output = torch.sum(gating_weights * expert_outputs, dim=1) # [batch_size, input_dim]
        return output

    # 融合分类器
class MultiModalClassifier(nn.Module):
    def __init__(self, text_dim=768, audio_dim=128, hidden_dim=512, n_head=4, num_classes=NUM_CLASSES):
        super(MultiModalClassifier, self).__init__()
        self.audio_proj = nn.Linear(audio_dim, text_dim)  # 将音频特征映射到与文本相同维度
        self.cross_attention = CrossMultiheadAttention(
            embed_dim=text_dim,
            num_heads=n_head,
            dropout=0.1
        )
        self.add_norm = AddNorm(text_dim)
        self.ffn = PositionwiseFeedForward(text_dim, hidden_dim)
        self.fc = nn.Linear(text_dim, num_classes)
        self.softmax = nn.Softmax(dim=1)
        #MOE
        self.moe=MoE(text_dim, 4)


        # 门控机制
        self.gate= GatingNetwork(text_dim*3)  # 文本、音频、图像特征的门控融合
        




    
    def forward(self, outputs, audio_embedding, image_results):
                
        
        x_aud = self.audio_proj(audio_embedding)
        image_embedding = adjust_cls_feats(outputs, image_results)
        x_img = image_embedding.unsqueeze(1)  # (B, 1, D)
        x_text = outputs['hiddens']  # (B, seq_len, D)
        x_cat = torch.cat([x_img, x_aud, x_text], dim=-1)  # shape: [batch_size, 3d]
        #门控
        weights = self.gate(x_cat)                                # [batch_size, 3]
        w_img, w_aud, w_text = weights[:, 0:1], weights[:, 1:2], weights[:, 2:3]
        x_img = w_img * x_img
        x_aud = w_aud * x_aud
        x_text = w_text * x_text

        
        

        for i in range(4):
            # 第一步：分别做 cross-attention，得到两个模态的注意输出
            y, _ = self.cross_attention(x_text, x_img)  # expert 1
            y, _ = self.cross_attention(y, x_aud)  # expert 2

            # 第二步：MoE 门控融合
            y=self.moe(y)

            # 第三步：Add & Norm
            x = self.add_norm(y, x)

            # 第四步：FFN
            y = self.ffn(x)
            x = self.add_norm(y, x)

        # 第五步：取CLS token做分类
        outputs = self.fc(x[:, 0, :])
        outputs = self.softmax(outputs)
        return outputs
        
