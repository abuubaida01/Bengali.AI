# if i import images from file, as it is in other formate, will take a lot of time, so let's change the formate of file
import pandas as pd 
import numpy as np 
from tqdm import tqdm 
import glob
import joblib


if __name__ == '__main__':
    files = glob.glob(r'C:/Users/Abu Ubaida/Desktop/Projects/DL_Projects/Bengali.AI/input/t rain_*.parquet')
    for f in files:
        df = pd.read_parquet(f)
        image_ids = df.image_id.values
        
        # this df contains pixel values, each row is one image
        df = df.drop('image_id', axis=1)
        image_array = df.values

        for j, img_id in tqdm(enumerate(image_ids), total=len(image_ids)):
            joblib.dump(image_array[j, :], f'C:/Users/Abu Ubaida/Desktop/Projects/DL_Projects/Bengali.AI/input/image_pickles/{img_id}.pkl') # jth row and all cols
