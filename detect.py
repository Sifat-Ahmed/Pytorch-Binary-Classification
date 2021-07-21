import warnings
warnings.filterwarnings('ignore')

import cv2
import os
from tqdm import tqdm
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader
import torchvision.models as models
from helper import TqdmUpTo, MetricMonitor, calculate_accuracy
from dataset.utils import get_train_test
from dataset.dataset import SmokeDataset
from visualize import display_image_grid, visualize_augmentations
from config import Config


class Detect:
    def __init__(self):
        self._cfg = Config()       
        
        self._model = getattr(models,self._cfg.model_name) (pretrained=False, num_classes = self._cfg.num_classes)
        
        if os.path.isfile(self._cfg.model_path):
            self._model.load_state_dict(torch.load(self._cfg.model_path))
        else:
            raise("Model path not found")
        
        self._model.to(self._cfg.device)
        self._model.eval()
        
        
    def _preprocess_image(self, image):
        if self._cfg.resize:
            image = cv2.resize(image, self._cfg.image_size)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self._cfg.test_transform(image = image)["image"]
        
        return image
    
    
    def is_smoke(self, image):
        output = None
        
        image = self._preprocess_image(image)
        
        with torch.no_grad():
            image = image.unsqueeze(0)
            image = image.to(self._cfg.device)
        
            output = self._model(image)
            output = True if torch.sigmoid(output) >= self._cfg.classification_threshold else False
        
        return output
    
    
    def run(self, source):
        self._read_source(source)
    
    
    def _read_source(self, video_path):
        
        if not os.path.isfile(video_path):
            raise("Video file not found")
        
        cap = cv2.VideoCapture(video_path)
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

        pbar = tqdm(total = frames)
        i, count = 1, 1
        color = (255, 255, 255)
        text = ""

        ret = True

        while ret:
            ret, frame = cap.read()
            if frame is None: continue
            
            image = frame.copy()
            
            output = self.is_smoke(image)
            
            if output: 
                color = (0, 0, 255)
                text = "smoke"
            else: 
                color = (0, 255, 0)
                text = "no smoke"
                
            image = cv2.putText(frame.copy(),
                                text, 
                                (10, 600), 
                                cv2.FONT_HERSHEY_SIMPLEX, 
                                5, 
                                color, 
                                1, 
                                cv2.LINE_AA)
            
            cv2.imshow('output', image)
            count += 1
            pbar.update(i)
            cv2.waitKey(1)
        #     plt.figure(figsize=(25, 15))
        #     plt.imshow(image)
        #     plt.show()
        
        
        

if __name__ == '__main__':
    det = Detect()
    det.run(r"smoke_data/videos/WIN_20210619_16_09_03_Pro.mp4")