import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer
from ultralytics import YOLO
from sklearn.model_selection import train_test_split
# from AudioModel import extract_audio_features
from audiomodel2 import extract_audio_features
from TextModel import Transformer

MAX_LEN = 128  # 文本最大长度
num_classes = 4  # 类别数量
label_dict = {'鼓励或表扬': 1, '讲授': 0, '提问': 3, '演示或示范': 2,'demonstration': 2, 'question': 3, 'praise': 1, 'teach': 0,'课堂管理':4} 
label_dict2 = {'鼓励或表扬': 1, '讲授': 0, '提问': 3, '演示或示范': 2,'课堂管理':4}
# 自定义数据集
class MultiModalDataset(Dataset):
    def __init__(self, data_list, max_len=MAX_LEN):
        self.data_list = data_list  # [(text, audio_path, image_path, label), ...]
        self.max_len = max_len
        self.base_model = AutoModel.from_pretrained(r'C:\Users\HUAWEI\Desktop\双重对比学习\Dual-Contrastive-Learning-main\Dual-Contrastive-Learning-main\Chinesebert')
        self.tokenizer = AutoTokenizer.from_pretrained(r'C:\Users\HUAWEI\Desktop\双重对比学习\Dual-Contrastive-Learning-main\Dual-Contrastive-Learning-main\Chinesebert')
        label_list = list(label_dict2.keys())
        sep_token = ['[SEP]']
        self.dataset=[]
        # 初始化模型
        TextModel = Transformer(self.base_model, num_classes=5, method='DualCL')
        state_dict = torch.load('best_model (alpha0.4-best94%).pth', map_location=torch.device('cpu'))
        TextModel.load_state_dict(state_dict)
        TextModel.eval()  # 设置模型为评估模式

        ImgModel = YOLO(model=r'C:\Users\HUAWEI\Desktop\ultralytics-main\ultralytics-main\runs\train\exp86\weights\best.pt')
        ImgModel.eval()  # 设置模型为评估模式

        for data in data_list:
            text, audio_path, image_path, label = data
            tokens = self.tokenizer.tokenize(text)
            text_input = label_list + sep_token + tokens

            inputs = self.tokenizer(
                text_input,
                padding='max_length',
                truncation=True,
                max_length=256,
                is_split_into_words=True,
                add_special_tokens=True,
                return_tensors='pt'
            )

            positions = torch.zeros_like(inputs['input_ids'])
            positions[:, num_classes:] = torch.arange(0, inputs['input_ids'].size(1) - num_classes)
            inputs['position_ids'] = positions

            with torch.no_grad():
                outputs = TextModel(inputs)

            outputs['cls_feats'] = outputs['cls_feats'].squeeze(0)
            outputs['label_feats'] = outputs['label_feats'].squeeze(0)
            outputs['predicts'] = outputs['predicts'].squeeze(0)
            outputs['hiddens'] = outputs['hiddens'].squeeze(0)

            audio_feature = extract_audio_features(audio_path)
            audio_feature = torch.tensor(audio_feature).float()

            # Perform prediction
            results = ImgModel.predict(source=image_path,
                                       conf=0.4,
                                       iou=0.5
                                       )
            # 提取预测的分类
            predicted_classes = []
            for result in results:
                # 获取预测的类别索引
                class_indices = result.boxes.cls.cpu().numpy().astype(int)
                # 获取类别名称（假设模型有对应的类别名称列表）
                class_names = ImgModel.names
                # 提取预测的分类名称
                predicted_classes.extend([class_names[idx] for idx in class_indices])

            if predicted_classes:
                predicted_classes = label_dict[predicted_classes[0]]
            else:
                predicted_classes = 5  # 如果没有检测到任何目标，设置为5（无效类别）

            label = torch.tensor(label).long()
            print("text shape:", outputs['hiddens'].shape)
            print("audio shape:", audio_feature.shape)

            self.dataset.append((outputs, audio_feature, predicted_classes, label))
        self.data_list = self.dataset



    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        

        return self.data_list[idx]
        
        
        
    
    # 准备数据
def prepare_data():
    # 加载数据
    path=r"C:\Users\HUAWEI\Desktop\teacher\多模态数据集\【线上教师】科学课1-数据集"

    data=[]
    
    # ['teach','praise','demonstration','question']

    for dir in os.listdir(path):
        if not label_dict.get(dir) is None:
            for textfile in os.listdir(os.path.join(path, dir)):
                if textfile.endswith('.txt'):
                    name = os.path.splitext(textfile)[0]
                    textfile = os.path.join(path, dir, textfile)
                    with open(textfile, 'r', encoding='utf-8') as f:
                        text = f.read()
                    data.append((text, os.path.join(path, dir, name+'.wav'), os.path.join(path, dir, name+'.jpg'),  label_dict[dir]))
    # print(data)               

    """
    自己构建一个列表: [(text, audio_path, label), ...]
    例如:
    data = [
        ("老师讲授新的知识点", "./audios/1.wav", 0),
        ("教师提出问题让学生思考", "./audios/2.wav", 1),
        ...
    ]
    """

    train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)
    # print(train_data)
    return train_data, val_data

# if __name__ == '__main__':
#     train_data, val_data = prepare_data()
#     train_dataset = MultiModalDataset(train_data)
#     val_dataset = MultiModalDataset(val_data)
#     print(len(train_dataset))
#     print(len(val_dataset))
    # print(train_dataset[0])
    # print(val_dataset[0])
    # print(train_dataset[0][0].shape)