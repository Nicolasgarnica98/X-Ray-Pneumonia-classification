import glob
import os
from Get_dataset import get_dataset
import matplotlib.pyplot as plt

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
    get_dataset.data_exploration(train_IMG,train_lbl_txt)


if __name__ =='__main__':
    main()
