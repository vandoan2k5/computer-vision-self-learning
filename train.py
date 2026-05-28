# %% [cell 1]: Import modules tự tạo & thiết lập seed
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

import config
from model import CIFAR100Net
from utils.dataset import CIFAR100CustomDataset, get_transforms
from utils.engine import train_one_epoch, validate

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

set_seed(config.SEED)

# %% [cell 2]: Setup Data (Train, Validation DataLoader)
df = pd.read_csv(config.CSV_PATH)
train_df_full = df[df['split'] == 'train'].reset_index(drop=True)

# Chia dữ liệu có stratify theo nhãn lớp để đảm bảo cân bằng
train_df, val_df = train_test_split(
    train_df_full, 
    test_size=0.1, 
    stratify=train_df_full['label_idx'], 
    random_state=config.SEED
)
train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)

train_tf, val_tf = get_transforms()

# Cấu hình tối ưu nạp dữ liệu cho GPU T4x2
train_loader = torch.utils.data.DataLoader(
    CIFAR100CustomDataset(train_df, config.DATA_DIR, train_tf), 
    batch_size=config.BATCH_SIZE, shuffle=True,
    num_workers=4, pin_memory=True
)

val_loader = torch.utils.data.DataLoader(
    CIFAR100CustomDataset(val_df, config.DATA_DIR, val_tf), 
    batch_size=config.BATCH_SIZE, shuffle=False,
    num_workers=4, pin_memory=True
)

print(f"Data Loaded! Train batch size: {config.BATCH_SIZE}")

# %% [cell 3]: Init Model, Optimizer, Scheduler & Training Loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CIFAR100Net(num_classes=config.NUM_CLASSES)

if torch.cuda.device_count() > 1:
    print(f"Sử dụng {torch.cuda.device_count()} GPUs với nn.DataParallel")
    model = nn.DataParallel(model)
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=config.LR, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()

# Khởi tạo Cosine Learning Rate Scheduler giúp hội tụ mượt mà
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.EPOCHS)

# SỬA LỖI: Khởi tạo biến lưu lịch sử và độ chính xác tốt nhất
history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
best_val_acc = 0.0

print("Bắt đầu huấn luyện...")
for epoch in range(config.EPOCHS):
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = validate(model, val_loader, criterion, device)
    
    # Cập nhật và lấy thông tin Learning Rate hiện tại
    scheduler.step()
    current_lr = optimizer.param_groups[0]['lr']
    
    # Lưu lịch sử học tập
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)
    
    print(f"Epoch [{epoch+1}/{config.EPOCHS}] | LR: {current_lr:.6f} | "
          f"Train Loss: {train_loss:.4f} - Train Acc: {train_acc:.2f}% | "
          f"Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.2f}%")
    
    # SỬA LỖI: Kiểm tra và lưu checkpoint tốt nhất theo đường dẫn config.MODEL_SAVE_PATH
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
        torch.save(state_dict, config.MODEL_SAVE_PATH)
        print(f"--> Đã lưu mô hình mới tốt nhất vào {config.MODEL_SAVE_PATH} với Val Acc: {best_val_acc:.2f}%")

print("Huấn luyện hoàn tất!")