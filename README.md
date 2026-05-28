# CIFAR-100 Image Classification from Scratch with PyTorch (Dual GPU T4)

Dự án này xây dựng một pipeline phân loại hình ảnh bài bản từ đầu (**from scratch**) trên bộ dữ liệu **CIFAR-100** sử dụng cấu trúc mạng CNN tùy biến dạng VGG/ResNet. Mã nguồn được thiết kế theo dạng mô-đun hóa (modularized) chuyên nghiệp, tối ưu hóa để chạy song song trên hệ thống **Dual Nvidia T4 GPU** (Kaggle/Colab Environment).

## 📌 Cấu trúc thư mục (Project Architecture)

```text
project/
│
├── data/
│   └── cifar100_images/
│       ├── train/              # Thư mục chứa ảnh train
│       ├── test/               # Thư mục chứa ảnh test
│       └── labels.csv          # File quản lý nhãn và phân chia tập dữ liệu
│
├── model/
│   ├── __init__.py
│   ├── net.py                  # Định nghĩa luồng mạng mạng chính (CIFAR100Net)
│   └── components/
│       ├── __init__.py
│       └── residual_block.py   # Khối mạng Residual Block tùy biến
│
├── utils/
│   ├── __init__.py
│   ├── dataset.py              # Custom Dataset & Data Augmentation
│   └── engine.py               # Thao tác huấn luyện (Train & Validation loops)
│
├── .checkpoints/               # Thư mục tự động tạo để lưu weights
│   └── ckpt_001/
│       └── best_cifar100_model.pth
│
├── config.py                   # Quản lý Hyperparameters & Paths
├── train.py                    # Script chạy chính (hoặc chuyển đổi thành Notebook)
└── README.md

```

---

## 🛠️ Tính năng nổi bật (Key Features)

* **Modular Design:** Code tách biệt rõ ràng giữa cấu hình, dữ liệu, kiến trúc mô hình và vòng lặp huấn luyện giúp dễ dàng nâng cấp, debug và bảo trì.
* **Multi-GPU Ready:** Tự động phát hiện cấu hình phần cứng và phân phối tải dữ liệu đều lên cả 2 GPU bằng `nn.DataParallel`.
* **From Scratch Architecture:** Không sử dụng pre-trained model. Mạng CNN được tích hợp khối **Residual Connection** giúp giải quyết triệt để hiện tượng tiêu biến đạo hàm (Vanishing Gradient) khi mạng sâu hơn.
* **Tối ưu hóa pipeline dữ liệu:** Tích hợp `pin_memory=True` và `num_workers=4` trong `DataLoader` loại bỏ hiện tượng nghẽn cổ chai (bottleneck) khi CPU nạp dữ liệu lên GPU.
* **Học tập mượt mà:** Sử dụng `AdamW` kết hợp với `CosineAnnealingLR` giúp mô hình hội tụ sâu và đạt độ chính xác cao ở các epoch cuối.

---

## 🚀 Hướng dẫn cài đặt và Sử dụng

### 1. Chuẩn bị môi trường

Yêu cầu cài đặt Python $\ge$ 3.10 và thư viện PyTorch phù hợp với phiên bản CUDA hiện tại của bạn.

```bash
pip install torch torchvision pandas scikit-learn pillow matplotlib

```

### 2. Cấu hình Hyperparameters

Bạn có thể tùy chỉnh các tham số huấn luyện (Batch size, Learning rate, Epochs,...) trực tiếp trong file `config.py` trước khi chạy:

```python
# config.py
BATCH_SIZE = 128 * torch.cuda.device_count() # Tự động tối ưu hóa theo số lượng GPU
EPOCHS = 50
LR = 0.001

```

### 3. Huấn luyện mô hình (Training)

Chạy script huấn luyện chính bằng cách thực thi lệnh sau:

```bash
python train.py

```

Trong quá trình chạy, hệ thống sẽ in nhật ký (log) theo từng epoch và tự động lưu phiên bản có độ chính xác (Accuracy) tốt nhất trên tập Validation vào thư mục `.checkpoints/ckpt_001/`.

---

## 🧠 Chi tiết kỹ thuật (Technical Blueprint)

### Chiến lược xử lý Overfitting (Data Augmentation)

Ảnh CIFAR-100 có kích thước khá nhỏ ($32 \times 32$), việc huấn luyện từ đầu rất dễ dẫn đến hiện tượng học vẹt (overfitting). Dự án áp dụng bộ lọc tăng cường dữ liệu mạnh mẽ tại file `utils/dataset.py`:

* `RandomCrop(32, padding=4)`: Cắt ngẫu nhiên ảnh.
* `RandomHorizontalFlip()`: Lật ảnh ngẫu nhiên theo chiều ngang.
* `Normalize()`: Chuẩn hóa dữ liệu theo phân phối Mean & Std chuẩn của CIFAR-100 giúp ổn định quá trình tính toán đạo hàm.

### Kiến trúc mô hình (Model Block)

Mô hình cấu thành từ khối `Prep block` (3 kênh màu $\rightarrow$ 64 kênh đặc trưng), đi qua 4 tầng `Residual Layer` lớn tăng dần số lượng bộ lọc đặc trưng (64 $\rightarrow$ 128 $\rightarrow$ 256 $\rightarrow$ 512 channels) giúp mạng bắt được các cấu trúc hình học phức tạp của 100 lớp đối tượng khác nhau. Cuối cùng, tầng `Dropout(0.5)` ngắt ngẫu nhiên 50% neuron để ép mạng phải học các đặc trưng tổng quát nhất trước khi đưa qua bộ phân loại tuyến tính `nn.Linear`.

---

## 📈 Kết quả (Evaluation)

Sau khi hoàn tất quá trình huấn luyện, bạn có thể thực hiện load model đã lưu để đánh giá chất lượng trên tập `test` độc lập:

```python
from model import CIFAR100Net
import torch

model = CIFAR100Net(num_classes=100)
# Tải trọng số tốt nhất đã huấn luyện thành công
model.load_state_dict(torch.load(".checkpoints/ckpt_001/best_cifar100_model.pth"))
model.eval()

```