import torch
import torch.nn as nn
import numpy as np
import  shutil
model = torch.hub.load('harritaylor/torchvggish', 'vggish')
model.eval()
import os
# Download an example audio file
# import urllib
# url, filename = ("http://soundbible.com/grab.php?id=1698&type=wav", "bus_chatter.wav")
# try: urllib.URLopener().retrieve(url, filename)
# except: urllib.request.urlretrieve(url, filename)


###############################################################################
# audio_path = "bus_chatter.wav"  # 替换为你的音频文件路径
# output = model(audio_path)
# max_feature, _ = torch.max(output, dim=1)
# print(output.shape)  # 输出特征的形状
# print(max_feature.shape)  # 输出最大特征的形状

def extract_audio_features(audio_path, max_len=10):
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"音频文件不存在: {audio_path}")
    
    # print("Extracting audio features from:", audio_path)
    output = model(audio_path)  # 或 model.forward(filename)，视库而定
    # if output == torch([]) or output.shape[0] == 0:
    #     # 若模型输出为空，返回全零向量
    #     return torch.zeros((max_len, model.feature_dim))  # 你需要指定 model.feature_dim 或固定值
    seq_len, feature_dim = output.shape
    # 截断（前 max_len）
    if seq_len > max_len:
        output = output[:max_len, :]
    
    # padding（后 pad）
    elif seq_len < max_len:
        pad_len = max_len - seq_len
        pad = torch.zeros((pad_len, feature_dim), device=output.device)
        output = torch.cat([output, pad], dim=0)
    
    # shape: [max_len, feature_dim]
    return output

    # max_feature, _ = torch.max(output, dim=0)  # dim=0帧维度
    # print(max_feature.shape)
    # if max_feature.shape[0] != 128:
    #     raise ValueError(f"特征维度不符合要求: 预期128，实际{max_feature.shape[0]}")
    

# 预处理数据
# num=0
# target_path = r"C:\Users\HUAWEI\Desktop\Problem"
# path=r"C:\Users\HUAWEI\Desktop\teacher\多模态数据集\【线上教师】科学课1-数据集"
# label_dict = {'鼓励或表扬': 1, '讲授': 0, '提问': 3, '演示或示范': 2,'课堂管理':4}
# for dir in os.listdir(path):
#         if not label_dict.get(dir) is None:
#             datafile=os.path.join(path, dir)
#             for audiofile in os.listdir(datafile):
#                 if audiofile.endswith('.wav'):
#                     audio_path = os.path.join(datafile, audiofile)
#                     audio_feature = extract_audio_features(audio_path)
#                     if audio_feature.shape != torch.Size([128]):
#                         print(audio_feature.shape)
#                         shutil.move(audio_path, target_path)
#                         textfile = os.path.splitext(audiofile)[0] + '.txt'
#                         text_path = os.path.join(datafile, textfile)
#                         shutil.move(text_path, target_path)
#                         picturefile = os.path.splitext(audiofile)[0] + '.jpg'
#                         picture_path = os.path.join(datafile, picturefile)
#                         shutil.move(picture_path, target_path)
#                         print(f"移动了不符合要求的音频文件: {audio_path}")
#                         num+=1
# print(f"移动了{num}个不符合要求的音频文件和对应的文本、图片文件到目标路径: {target_path}")
# 移动了22个   
    
            


# class FintuneModel(nn.Module):
#     def __init__(self):
#         super(FintuneModel, self).__init__()
#         urls = {
#             'vggish': "https://github.com/harritaylor/torchvggish/releases/download/v0.1/vggish-10086976.pth"
#         }
#         self.pretrain = vggish.VGGish(urls, preprocess=False, postprocess=False)
#         self.classifier = classifier()

#     def forward(self, x):
#         """
#         :param x: [bs, num_frames, 96, 64]
#         :return:
#         """
#         bs, num_frames, _, _ = x.size()
#         x = x.view(bs*num_frames, 1, x.size(2), x.size(3))
#         x = self.pretrain(x) # [bs*num_frames, 128]
#         x = x.view(bs, x.size(1), num_frames)
#         x = self.classifier(x)
#         return x