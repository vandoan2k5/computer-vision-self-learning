import os
import cv2
import numpy as np
from torch.utils.data import Dataset
from utils.dataset import PascalVoc
VOC_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", 
    "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", 
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]


def extract_candidates(img):
    """
    Sử dụng thuật toán Selective Search của OpenCV để tìm các vùng đề xuất (ROIs)
    Lưu ý: Yêu cầu gói opencv-contrib-python
    """
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(img)
    ss.switchToSelectiveSearchFast()
    rects = ss.process() # Trả về định dạng: [x, y, w, h]
    # Giới hạn lấy tối đa 500 vùng đề xuất hàng đầu để tránh quá tải bộ nhớ
    return rects[:500] 

def extract_iou(boxA, boxB):
    """
    Tính chỉ số IoU (Intersection over Union) giữa 2 hộp tọa độ dạng [xmin, ymin, xmax, ymax]
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


if __name__ == '__main__':
    # Đường dẫn tới thư mục chứa dữ liệu chuẩn yolo của bạn
    DATA_DIR = "data/data" 
    
    # Khởi tạo đối tượng Dataset
    ds = PascalVoc(dir=DATA_DIR)
    
    # Khởi tạo các danh sách chứa kết quả đầu ra
    FPATHS, GTBBS, CLSS, DELTAS, ROIS, IOUS = [], [], [], [], [], []
    N = 500  # Giới hạn số lượng ảnh chạy thử mẫu ban đầu

    print(f"Bắt đầu tiền xử lý dữ liệu cho {N} ảnh...")

    for ix, (im, bbs, labels, fpath) in enumerate(ds):
        if ix == N: 
            break
            
        # Nếu ảnh không có vật thể nào (Ground truth trống), bỏ qua để tránh lỗi argmax
        if len(bbs) == 0:
            continue
            
        H, W, _ = im.shape
        
        # 1. Trích xuất các ứng viên vùng đề xuất bằng thuật toán Selective Search
        candidates = extract_candidates(im)
        candidates = np.array([(x, y, x+w, y+h) for x, y, w, h in candidates])
        
        # Phòng trường hợp thuật toán không tìm được ứng viên nào
        if len(candidates) == 0:
            continue

        # 2. Tính toán ma trận IoU giữa tất cả ứng viên với tất cả hộp chuẩn trong ảnh
        ious = np.array([[extract_iou(candidate, _bb_) for candidate in candidates] for _bb_ in bbs]).T
        
        img_rois, img_clss, img_deltas = [], [], []
        
        # 3. Duyệt qua từng ứng viên đề xuất để phân loại mục tiêu
        for jx, candidate in enumerate(candidates):
            cx, cy, cX, cY = candidate
            candidate_ious = ious[jx]
            
            best_iou_at = np.argmax(candidate_ious)
            best_iou = candidate_ious[best_iou_at]
            best_bb = _x, _y, _X, _Y = bbs[best_iou_at]
            
            # Thiết lập nhãn lớp dựa trên ngưỡng IoU = 0.3
            if best_iou > 0.3:
                # Lấy class_id từ nhãn gốc (ví dụ: số 0 đến 19)
                img_clss.append(labels[best_iou_at])
            else:
                # Gán nhãn là 'background' (hoặc bạn có thể dùng một ID số cố định như 20)
                img_clss.append('background')
                
            # Tính toán tỷ lệ sai lệch vị trí (Delta Offset) chuẩn hóa theo kích thước ảnh
            delta = np.array([_x - cx, _y - cy, _X - cX, _Y - cY]) / np.array([W, H, W, H])
            
            img_deltas.append(delta)
            img_rois.append(candidate / np.array([W, H, W, H]))
            
        # Lưu kết quả của ảnh này vào danh sách tổng
        FPATHS.append(fpath)
        IOUS.append(ious)
        ROIS.append(img_rois)
        CLSS.append(img_clss)
        DELTAS.append(img_deltas)
        GTBBS.append(bbs)
        
        if (ix + 1) % 50 == 0:
            print(f" -> Đã xử lý xong: {ix + 1}/{N} ảnh")

    print("\n[THÀNH CÔNG] Đã hoàn thành xử lý tập dữ liệu mẫu!")
    print(f"Tổng số ảnh đã trích xuất thành công: {len(FPATHS)}")