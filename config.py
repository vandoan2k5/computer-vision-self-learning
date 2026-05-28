import torch
import os

DATA_DIR = "data/cifar100_images"
CSV_PATH = "data/cifar100_images/labels.csv"
MODEL_SAVE_PATH = ".checkpoints/ckpt_001/best_cifar100_model.pth"

os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

BATCH_SIZE = 128 * torch.cuda.device_count()
EPOCHS = 50
LR = 0.001
NUM_CLASSES = 100
SEED = 42