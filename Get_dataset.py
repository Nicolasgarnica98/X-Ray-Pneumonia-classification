import os
import gdown
import glob
import numpy as np
from tqdm import tqdm
from zipfile import ZipFile
from skimage.io import imread
from skimage.color import rgb2gray

class get_dataset:

    def download(url):
        if os.path.exists('./dataset') == False:
            os.mkdir('./dataset')
            file_id=url.split('/')[-2]
            prefix = 'https://drive.google.com/uc?/export=download&id='
            downloaded_file = gdown.download(prefix+file_id,'./dataset/chest_xray.zip')
            return downloaded_file


    def unzip_dataset(dataset_comp):
        if dataset_comp != None:
            with ZipFile(dataset_comp,'r') as zip_object:
                zip_object.extractall(path='./dataset/')
            os.remove(dataset_comp)


    def get_labels(df):
        labels = []
        txt_labels = []
        for i in range(0,len(df)):
            if df[i].find('virus')!=-1:
                labels.append(1)
                txt_labels.append('Virus')
            elif df[i].find('bacteria')!=-1:
                labels.append(2)
                txt_labels.append('Bacteria')
            else:
                labels.append(0)
                txt_labels.append('Normal')
        labels = np.array(labels)

        return labels, txt_labels


    def load_images(df_img):
        img_array = []
        for i in tqdm(range(0,len(df_img)),desc='Loading images...'):
            img_array.append(imread(df_img[i]))
            
        return img_array
