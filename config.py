import os
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2


class Config:

    def __init__(self) -> None:

        self.num_classes = 1
        self.epochs = 20
        self.batch_size = 64
        self.dataset_dir = r'smoke_data/dataset'

        self.resize = True
        self.image_size = (224, 224)

        self.num_workers = 4
        self.pin_memory = True
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        
        self.model_name = 'resnet50'
        self.model_path = r'saved/'+ self.model_name +'_'+str(self.image_size[0])+'x'+str(self.image_size[1])+'.pth'
        
        self.learning_rate = 0.001

        self.classification_threshold = 0.75

        self.train_transform = A.Compose(
            [
                A.CLAHE(p=1.0),
                A.Cutout(num_holes=16, max_h_size=13, max_w_size=13, fill_value=[
                         225, 225, 225], always_apply=False, p=0.8),
                A.Flip(p=0.8),
                A.RGBShift(r_shift_limit=5, g_shift_limit=5,
                           b_shift_limit=5, p=0.5),
                A.RandomFog(fog_coef_lower=0.3, fog_coef_upper=0.7,
                            alpha_coef=0.2, always_apply=False, p=0.8),
                A.RandomBrightnessContrast(p=0.5),
                A.Normalize(mean=(0.485, 0.456, 0.406),
                            std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )

        self.val_transform = A.Compose(
            [
                A.Normalize(mean=(0.485, 0.456, 0.406),
                            std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )

        self.test_transform = A.Compose(
            [
                A.Normalize(mean=(0.485, 0.456, 0.406),
                            std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )
