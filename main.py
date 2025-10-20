import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
# from CombineModel import MultiModalClassifier
from CombineModel2 import MultiModalClassifier
from tqdm import tqdm
from datasets_unit import MultiModalDataset, prepare_data
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import seaborn as sns
import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BERT_MODEL = 'bert-base-chinese'  # BERT模型
BATCH_SIZE = 4  # 批量大小
LEARNING_RATE = 1e-5  # 学习率
EPOCHS = 50  # 训练轮数

# 训练函数
def train_one_epoch(model, dataloader, criterion, optimizer):
    model.train()
    total_loss = 0
    total_correct = 0

    for text_outputs, audio_feature, Image_classes, label in tqdm(dataloader):
        text_outputs = {k: v.to(DEVICE) if hasattr(v, 'to') else v for k, v in text_outputs.items()}
        audio_feature = audio_feature.to(DEVICE)
        Image_classes = Image_classes.to(DEVICE)
        label = label.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(text_outputs, audio_feature, Image_classes)

        loss = criterion(outputs, label)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = outputs.argmax(dim=1)
        total_correct += (preds == label).sum().item()

    return total_loss / len(dataloader), total_correct / len(dataloader.dataset)

# 验证函数
def evaluate(model, dataloader, criterion):
    model.eval()
    total_loss = 0
    total_correct = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for text_outputs, audio_feature, Image_classes, label in tqdm(dataloader):
            text_outputs = {k: v.to(DEVICE) if hasattr(v, 'to') else v for k, v in text_outputs.items()}
            audio_feature = audio_feature.to(DEVICE)
            Image_classes = Image_classes.to(DEVICE)
            label = label.to(DEVICE)

            outputs = model(text_outputs, audio_feature, Image_classes)

            loss = criterion(outputs, label)

            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            total_correct += (preds == label).sum().item()
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(label.cpu().tolist())

    cm = confusion_matrix(all_labels, all_preds)
    return total_loss / len(dataloader), total_correct / len(dataloader.dataset), cm , all_preds, all_labels
# 计算模型参数数量
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
# 主程序
def main():
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    best_train_acc = 0  # 初始化最佳训练准确率
    best_val_acc = 0  # 初始化最佳验证准确率
    best_val_precision = 0
    best_val_recall = 0
    best_val_f1 = 0
    best_val_epoch = 0
    train_data, val_data = prepare_data()

    train_dataset = MultiModalDataset(train_data)
    val_dataset = MultiModalDataset(val_data)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    model = MultiModalClassifier().to(DEVICE)
    param_count = count_parameters(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(list(model.parameters()), lr=LEARNING_RATE)
    torch.autograd.set_detect_anomaly(True)
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch+1}/{EPOCHS}")
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc, val_cm, val_preds, val_labels = evaluate(model, val_loader, criterion)
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        val_losses.append(val_loss) 
        val_accuracies.append(val_acc)
        # 计算精确率、召回率和F1分数
        precision, recall, f1, _ = precision_recall_fscore_support(
            val_labels, val_preds, average='weighted', zero_division=0
        )
        # 在验证集上检查是否达到最佳准确率
        # 在验证集上检查是否达到最佳性能
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_precision = precision
            best_val_recall = recall
            best_val_f1 = f1
            best_val_epoch = epoch + 1
        if train_acc > best_train_acc:
            best_train_acc = train_acc
            # best_model_state = model.state_dict()  # 保存最佳模型的参数
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val Precision: {precision:.4f} | Val Recall: {recall:.4f} | Val F1: {f1:.4f}")
    print("\n=== 最终结果 ===")
    print(f"最佳训练准确率: {best_train_acc:.4f}")
    print(f"最佳验证性能 (Epoch {best_val_epoch}):")
    print(f"验证准确率: {best_val_acc:.4f}")
    print(f"验证精确率: {best_val_precision:.4f}")
    print(f"验证召回率: {best_val_recall:.4f}")
    print(f"验证F1分数: {best_val_f1:.4f}")
    print(f"模型参数数量: {param_count:,}")
    print("训练后的 alpha 参数: ", model.alpha.detach().numpy())    
    # 绘制损失曲线
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()

    # 绘制准确率曲线
    plt.plot(train_accuracies, label='Train Acc')
    plt.plot(val_accuracies, label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.show()

    # 绘制混淆矩阵
    plt.figure(figsize=(10, 7))
    sns.heatmap(val_cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Validation Confusion Matrix')
    plt.show()

if __name__ == '__main__':
    main()