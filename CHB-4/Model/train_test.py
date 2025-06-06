import torch
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
# --------------------------------------------------------
from Datasets_Train import train_dataloader
from Datasets_Test import test_dataloader
from model_seizure import EEGLightNet
#%%
# 设置随机种子
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

#%%
def train_and_test_model(model, train_loader, test_loader, criterion, optimizer, device, num_epochs=10, seed=42):
    set_seed(seed)  # 固定随机种子
    model.to(device)

    # 初始化记录列表
    train_losses = []
    train_accuracies = []
    test_accuracies = []
    learning_rates = []

    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    best_acc = 0.0

    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        batch_count = 0

        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs} (Training)')

        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            correct_predictions += (predicted == labels).sum().item()
            total_predictions += labels.size(0)
            batch_count += 1

            progress_bar.set_postfix({
                'loss': running_loss / batch_count,
                'acc': correct_predictions / total_predictions,
                'lr': optimizer.param_groups[0]['lr']
            })

        # 记录训练集性能
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct_predictions / total_predictions
        current_lr = optimizer.param_groups[0]['lr']

        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)
        learning_rates.append(current_lr)

        # 测试阶段
        model.eval()
        correct_predictions = 0
        total_predictions = 0

        with torch.no_grad():
            for inputs, labels in tqdm(test_loader, desc=f'Epoch {epoch + 1}/{num_epochs} (Testing)'):
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)

                total_predictions += labels.size(0)
                correct_predictions += (predicted == labels).sum().item()

        test_accuracy = correct_predictions / total_predictions
        test_accuracies.append(test_accuracy)
        print(f"Test Accuracy: {test_accuracy:.4f}")

        scheduler.step(test_accuracy)  # 根据测试准确率调整学习率

        if test_accuracy > best_acc:
            best_acc = test_accuracy
            torch.save(model.state_dict(), 'model_seizure_5(92.2).pth')

    # 绘图
    plt.figure(figsize=(12, 7))

    plt.subplot(2, 2, 1)
    plt.plot(train_losses, label='Training Loss', color='r')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(train_accuracies, label='Training Accuracy', color='b')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy Curve')
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(test_accuracies, label='Test Accuracy', color='g')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Test Accuracy Curve')
    plt.legend()

    plt.tight_layout()
    plt.show()

    print(f'Training and testing completed. Best accuracy: {best_acc:.4f}')
    return train_losses, train_accuracies, test_accuracies

#%%
# 初始化模型和优化器
model = EEGLightNet()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#%%
# (model, train_loader, test_loader, criterion, optimizer, device, num_epochs)
train_and_test_model(model=model,
                     train_loader=train_dataloader,
                     test_loader=test_dataloader,
                     criterion=criterion,
                     optimizer=optimizer,
                     device=device,
                     num_epochs=10)