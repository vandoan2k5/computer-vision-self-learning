from torch.utils.data import Dataset
import os
import cv2

VOC_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", 
    "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", 
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

class PascalVoc(Dataset):
    def __init__(self, dir):
        self.images_dir = os.path.join(dir, "images")
        self.labels_dir = os.path.join(dir, "labels")
        self.image_files = os.listdir(self.images_dir)
        self.image_files.sort()
    
    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.images_dir, self.image_files[idx])
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        height, width, _ = image.shape

        with open(os.path.join(self.labels_dir, self.image_files[idx][:-4]+'.txt'), 'r') as f:
            labels = f.read().strip().split('\n')
        
        bboxes = []
        classes = []
        for label in labels:
            class_id, xc, yc, w, h = map(float, label.split())
            xmin = int((xc-w/2)*width)
            xmax = int((xc+w/2)*width)
            ymin = int((yc-h/2)*height)
            ymax = int((yc+h/2)*height)

            bboxes.append([xmin, ymin, xmax, ymax])
            classes.append(class_id)
            

        return image, bboxes, classes, image_path


