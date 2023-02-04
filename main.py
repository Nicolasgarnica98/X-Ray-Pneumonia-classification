import glob
import os
import numpy as np 
from Get_dataset import get_dataset
from PreProcessing import pre_processing

def main():
    compressed_dataset = get_dataset.download('https://drive.google.com/file/d/1OimFddxkp3Y9hYEdtFaFUst0QVLl5zXX/view?usp=sharing')
    get_dataset.unzip_dataset(compressed_dataset)

    df_test_normal = glob.glob(os.path.join('dataset/test/NORMAL','*.jpeg'))
    df_test_pneumonia = glob.glob(os.path.join('dataset/test/PNEUMONIA','*.jpeg'))
    df_train_normal = glob.glob(os.path.join('dataset/train/NORMAL','*.jpeg'))
    df_train_pneumonia = glob.glob(os.path.join('dataset/train/PNEUMONIA','*.jpeg'))
    df_train = df_train_normal + df_train_pneumonia
    df_test = df_test_normal + df_test_pneumonia

    train_fullLabels = get_dataset.get_labels(df_train)
    train_lbl = train_fullLabels[0]
    train_lbl_txt = train_fullLabels[1]
    train_IMG = get_dataset.load_images(df_train)
    # get_dataset.data_exploration(train_IMG,train_lbl_txt)

    #Pre-processing
    train_IMG = pre_processing.rgb_to_gray(train_IMG)
    train_IMG = pre_processing.resize_images(500,500,train_IMG)
    train_IMG = pre_processing.image_normalization(train_IMG)

    train_IMG = np.array(train_IMG)
    train_lbl = np.array(train_lbl)

    print(train_IMG.shape)
    # get_dataset.data_exploration(train_IMG,train_lbl_txt)

if __name__ =='__main__':
    main()
