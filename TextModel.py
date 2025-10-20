import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer



# 定义一个Transformer模型的包装类，通常用于分类任务（含支持对比学习方法）
class Transformer(nn.Module):
    def __init__(self, base_model, num_classes=5, method='DualCL'):
        super().__init__()

        # 基础模型（例如BERT、RoBERTa等Transformer模型）
        self.base_model = base_model

        # 类别数量，用于分类器输出维度
        self.num_classes = num_classes

        # 方法类型，可能是 'ce'（交叉熵），'scl'（监督对比学习）或其他支持 DualCL 的方法
        self.method = method
        
        #注意力层
        self.att_fc = nn.Linear(base_model.config.hidden_size, 1)
        
        # 小型 MLP 层：Linear -> ReLU -> Dropout -> Linear
        self.mlp = nn.Sequential(
            nn.Linear(base_model.config.hidden_size, base_model.config.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.4),  # 你可以调整
            nn.Linear(base_model.config.hidden_size, base_model.config.hidden_size)
        )

        # 分类头：将Transformer输出映射到类别空间（hidden_size -> num_classes）
        self.linear = nn.Linear(base_model.config.hidden_size, num_classes)

        # Dropout用于防止过拟合
        self.dropout = nn.Dropout(0.5)

        # 设置基础模型的所有参数都可以训练（不是冻结的）
        for param in base_model.parameters():
            param.requires_grad_(True)

    def forward(self, inputs):
        # 获取base_model的输出，通常是BERT的last_hidden_state
        raw_outputs = self.base_model(**inputs)

        # 取出所有Token的位置输出 (batch_size, seq_len, hidden_dim)
        hiddens = raw_outputs.last_hidden_state 
        # print("hiddens 的形状:", hiddens.shape)

        # 取出 [CLS] 位置的向量作为句子整体表示（一般在第0个位置）
        # cls_feats = hiddens[:, 0, :]  # (batch_size, hidden_dim)
        # attention pooling
        scores = torch.tanh(self.att_fc(hiddens))             # (batch_size, seq_len, 1)
        attn_weights = torch.softmax(scores, dim=1)           # (batch_size, seq_len, 1)
        cls_feats = torch.sum(attn_weights * hiddens, dim=1)  # (batch_size, hidden_dim)
        
        
        
        #增强最大值池化的特征
        # max_pooled = torch.max(hiddens, dim=1).values # 所有 token 的最大值池化
        # cls_feats = cls_feats + max_pooled

        # 如果是标准交叉熵分类或SCL对比学习方法
        if self.method in ['ce', 'scl']:
            label_feats = None  # 不使用额外的标签特征
            # 进行分类：先dropout，然后全连接输出分类结果
           

        else:
            # 如果使用DualCL等其他方法，使用标签感知的特征
            # 假设将第1到num_classes位置的Token用作类别特征（注意：输入要事先构造好）
            label_feats = hiddens[:, 1:self.num_classes + 1, :]  # (batch_size, num_classes, hidden_dim)
            label_feats = self.mlp(label_feats)  # (batch_size, num_classes, hidden_dim)
            

             # MLP 提取更深层次的句子表示
            cls_feats2 = self.mlp(cls_feats)  # (batch_size, hidden_dim)
            # print("cls_feats2 的形状:", cls_feats2.shape)
            
            

            # 使用cls_feats与label_feats做内积，得到类别的匹配得分（类似soft-label attention）
            # einsum表示：
            #   b d  -> batch, dim
            #   b c d -> batch, num_classes, dim
            #   输出为 b c -> batch, num_classes
            predicts = torch.einsum('bd,bcd->bc', cls_feats2, label_feats)

        # 返回一个包含预测值、[CLS]特征、标签特征的字典
        outputs = {
            'predicts': predicts,         # 分类分数
            'cls_feats': cls_feats2,       # [CLS]位置的特征表示
            'label_feats': label_feats,    # 标签感知特征（如果使用了）
            'hiddens': hiddens              # 原始Transformer输出

        }
        # print("label_feats 的形状:", outputs['label_feats'].shape)
        return outputs
    
# debug
# base_model = AutoModel.from_pretrained(r'C:\Users\HUAWEI\Desktop\双重对比学习\Dual-Contrastive-Learning-main\Dual-Contrastive-Learning-main\Chinesebert')  # 这里需要替换为你的实际 base_model
# tokenizer=AutoTokenizer.from_pretrained(r'C:\Users\HUAWEI\Desktop\双重对比学习\Dual-Contrastive-Learning-main\Dual-Contrastive-Learning-main\Chinesebert')
# model = Transformer(base_model)
# # 示例文本
# text = "这是一个示例句子。"
# # 对文本进行编码，得到字典形式的输入
# inputs = tokenizer(text, return_tensors='pt')

# # 调用模型
# outputs = model(inputs)

# print("label_feats 的形状:", outputs['label_feats'].shape)