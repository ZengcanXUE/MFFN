import torch
from TextModel import Transformer
from transformers import logging, AutoTokenizer, AutoModel
from tqdm import tqdm
import os
import torch.nn as nn
NUM_CLASSES = 5  # 类别数量

import torch


def adjust_cls_feats(text_outputs, image_results, alpha):
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
    selected_label_feats = []
    adjusted_cls_feats = []
    for i in range(batch_size):
        class_index = image_results[i]
        if class_index == 5:
            # 如果分类结果为 5，使用原 cls_feats 向量
            selected_label_feats.append(cls_feats[i])
            adjusted_cls_feats.append(cls_feats[i])
        else:
            # 否则，提取对应类别的特征向量
            selected_label_feats.append(label_feats[i, class_index])
            # 获取对应分类的 alpha 值
            current_alpha = alpha[class_index]
            # 调整 cls_feats 向量
            adjusted_feat = (1 - current_alpha) * cls_feats[i] + current_alpha * label_feats[i, class_index]
            adjusted_cls_feats.append(adjusted_feat)

    adjusted_cls_feats = torch.stack(adjusted_cls_feats)

    return adjusted_cls_feats


# 融合分类器
class MultiModalClassifier(nn.Module):
    def __init__(self, text_dim=768, audio_dim=41, hidden_dim=512, n_head=4, num_classes=NUM_CLASSES):
        super(MultiModalClassifier, self).__init__()
        self.audio_proj = nn.Linear(audio_dim, text_dim)  # 将音频特征映射到与文本相同维度
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=text_dim,
            num_heads=n_head,
            batch_first=True
        )
        # 门控机制
        self.gate_linear = nn.Linear(text_dim * 2, 1)
        self.sigmoid = nn.Sigmoid()

        self.fc1 = nn.Linear(text_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        # 初始化4个可训练的 alpha 参数，每个对应一个分类
        self.alpha = nn.Parameter(torch.tensor([0.5, 0.5, 0.8, 0.1], requires_grad=True))

    
    def forward(self, outputs, audio_embedding, image_results):
        # 调整 cls_feats 向量方向，使其略微偏向图像模型分类结果对应的类别向量
        audio_embedding = self.audio_proj(audio_embedding)  # (B, D)
        audio_embedding = audio_embedding.unsqueeze(1)      # (B, 1, D)

        x = outputs['cls_feats']  # (B, D)

        x = adjust_cls_feats(outputs, image_results,self.alpha)
        
        # 加上seq_len维度使其成为query
        query = x.unsqueeze(1)  # (B, 1, D)

        # Cross-Attention: 让文本向量 attend 到音频特征
        attn_output, _ = self.cross_attention(
            query=query,
            key=audio_embedding,
            value=query
        )  # (B, 1, D)

        attn_output = attn_output.squeeze(1)  # (B, D)

        # 门控融合
        gate_input = torch.cat([x, attn_output], dim=-1)  # (B, 2D)
        gate = self.sigmoid(self.gate_linear(gate_input))  # (B, 1)
        x = gate * x + (1 - gate) * attn_output  # (B, D)
        # x = torch.cat((x, audio_embedding), dim=1)
        
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


