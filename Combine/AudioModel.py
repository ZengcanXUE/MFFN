import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import json


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 音频特征提取

def extract_audio_features(audio_path, sr=16000, n_mfcc=13):
    # 采样率（默认16000Hz，常用于语音识别）, 提取13维MFCC特征(梅尔频率倒谱系数)
    # 1. 加载音频文件
    # audio_path是音频文件路径，sr是采样率（一般用16000Hz标准语音采样率）
    y, sr = librosa.load(audio_path, sr=sr)
    
    # 2. 计算MFCC特征
    # 从音频信号中提取n_mfcc维的MFCC特征矩阵，shape是 (n_mfcc, 时间帧数)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    
    # 3. 计算一阶差分特征（MFCC的变化率）
    # delta特征可以描述语音在时间轴上的变化趋势
    delta_mfcc = librosa.feature.delta(mfcc)
    
    # 4. 计算二阶差分特征（变化率的变化率）
    # 更进一步捕捉声音变化的动态信息
    delta2_mfcc = librosa.feature.delta(mfcc, order=2)
    
    # 5. 合并MFCC、ΔMFCC、ΔΔMFCC特征
    # 纵向堆叠成一个新的特征矩阵，shape是(39, 时间帧数)
    # 原始MFCC：13维,加上ΔMFCC（变化率）和ΔΔMFCC（变化率的变化），一共就39维。
    # 这是因为原始MFCC是对频率域的描述，而ΔMFCC和ΔΔMFCC是对时间域的描述。
    mfcc_features = np.vstack([mfcc, delta_mfcc, delta2_mfcc])

    # 6. 提取音高特征（pitch）
    # 使用Librosa的piptrack提取音高曲线（pitches）和对应的幅度（magnitudes）
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    # 只取所有大于0的音高的均值（因为piptrack会输出一堆0），如果没有音高则返回0
    pitch_feature = np.mean(pitches[pitches > 0]) if np.any(pitches > 0) else 0.0

    # 7. 提取能量特征（energy）
    # 计算每一帧的短时能量（rms），再取能量均值，反映整体声音的强弱
    energy_feature = np.mean(librosa.feature.rms(y=y))
    
    # 8. 对39维的MFCC+Δ+ΔΔ特征，在时间轴上取均值
    # 得到一个固定长度的39维向量，去除时间帧数的影响
    mfcc_mean = np.mean(mfcc_features, axis=1)

    # 9. 把 MFCC特征均值（39维）+ pitch（1维）+ energy（1维） 拼接成最终特征
    # 得到一个固定长度的41维特征向量
    audio_feature = np.hstack([mfcc_mean, pitch_feature, energy_feature])
    
    # 10. 返回最终的音频特征向量
    return audio_feature









