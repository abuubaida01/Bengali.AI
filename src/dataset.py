import albumentations
import joblib
from PIL import Image 
import pandas as pd
import numpy as np
import torch

class BengaliDatasetTrain:
    def __init__(self, folds, img_height, img_width, mean, std):
        df = pd.read_csv(r'C:\Users\Abu Ubaida\Desktop\Projects\DL_Projects\Bengali.AI\input\train_folds.csv')
        df = df[['image_id', 'grapheme_root', 'vowel_diacritic', 'consonant_diacritic', 'kfold']]
        # bring those fold which have been asked
        df = df[df.kfold.isin(folds)].reset_index(drop=True) 
        self.image_id = df.image_id.values
        self.grapheme_root = df.grapheme_root.values
        self.vowel_diacritic = df.vowel_diacritic.values
        self.consonant_diacritic = df.consonant_diacritic.values
        
        if len(folds)==1: # means it is an validation dataset
            self.aug = albumentations.Compose([
                albumentations.Resize(img_height, img_width, always_apply=True),
                albumentations.Normalize(mean, std, always_apply=True)
            ])
        else:             # Training dataset
            self.aug = albumentations.Compose([
                albumentations.Resize(img_height, img_width, always_apply=True),
                albumentations.ShiftScaleRotate(shift_limit=0.06,
                                                scale_limit=0.1,                              rotate_limit=5,
                                                p=0.9),
                albumentations.Normalize(mean, std, always_apply=True)
            ])
    def __len__(self):
        return len(self.image_id)
    
    def __getitem__(self, item):
        image = joblib.load(f'C:/Users/Abu Ubaida/Desktop/Projects/DL_Projects/Bengali.AI/input/image_pickles/{self.image_id[item]}.pkl')
        image = image.reshape(137, 236).astype(float) # from 1 d to 137, 236
        image = Image.fromarray(image).convert('RGB') # convert to PIL image
        image = self.aug(image=np.array(image))['image'] # Augmentations
        image = np.transpose(image, (2,0,1)).astype(np.float32)

        return {
            'image': torch.tensor(image, dtype=torch.float32),
            'vowel_diacritic':torch.tensor(self.vowel_diacritic[item], dtype=torch.long),
            'consonant_diacritic': torch.tensor(self.consonant_diacritic[item], dtype=torch.long),
            'grapheme_root':torch.tensor(self.grapheme_root[item], dtype=torch.long)
            }
            