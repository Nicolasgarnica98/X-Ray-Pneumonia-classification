import os
import wget
import base64
import shutil
import numpy as np
from tqdm import tqdm
from zipfile import ZipFile
from skimage.io import imread
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

class get_dataset:

    def download(url):

        def create_onedrive_directdownload (onedrive_link):
            data_bytes64 = base64.b64encode(bytes(onedrive_link, 'utf-8'))
            data_bytes64_String = data_bytes64.decode('utf-8').replace('/','_').replace('+','-').rstrip("=")
            resultUrl = f"https://api.onedrive.com/v1.0/shares/u!{data_bytes64_String}/root/content"
            return resultUrl

        if os.path.exists('./dataset') == False:
            os.mkdir('./dataset')
            onedrive_url = url
            # Generate Direct Download URL from above Script
            direct_download_url = create_onedrive_directdownload(onedrive_url)

            print('Downloading dataset')
            r = wget.download(url=direct_download_url, out='./dataset/')
            print('\n')
            return r



    def unzip_dataset(dataset_comp):
        if dataset_comp != None:
            with ZipFile(dataset_comp,'r') as zip_object:
                zip_object.extractall(path='./dataset/')
            os.remove(dataset_comp)


    def get_labels(df):
        labels = []
        txt_labels = []
        num_normal = 0
        num_bacteria = 0
        num_virus = 0
        for i in range(0,len(df)):
            if df[i].find('virus')!=-1:
                labels.append(1)
                txt_labels.append('Virus')
                num_virus += 1
            elif df[i].find('bacteria')!=-1:
                labels.append(2)
                txt_labels.append('Bacteria')
                num_bacteria += 1
            else:
                labels.append(0)
                txt_labels.append('Normal')
                num_normal += 1
        labels = np.array(labels)
        num_samples_per_class = {'virus':num_virus, 'bacteria':num_bacteria, 'Normal':num_normal}
        print(num_samples_per_class)
        
        return labels, txt_labels


    def load_images(df_img):
        img_array = []
        for i in tqdm(range(0,len(df_img)),desc='Loading images...'):
            img_array.append(imread(df_img[i]))
            
        return img_array
    
    def data_class_balance(df_img):
        len_df_virus = df_img.count('virus')
        len_df_bacteria = df_img.count('bacteria')
        len_df_normal = len(df_img)-(len_df_virus+len_df_bacteria)
        print(len_df_virus, len_df_bacteria, len_df_normal)

    def divide_dataset_in_folders(df_img):
        os.mkdir('./dataset/train')
        os.mkdir('./dataset/test')
        os.mkdir('./dataset/val')
        df_train, df_test = train_test_split(df_img,test_size=0.15)
        df_train, df_val = train_test_split(df_train,test_size=0.1)

        folder_array = ['./dataset/train', './dataset/test', './dataset/val']
        df_array = [df_train,df_test,df_val]

        for i in tqdm(range(0,len(folder_array)),'Moving files: '):
            for file in df_array[i]:
                source = file
                shutil.move(source, folder_array[i])

    


    def data_exploration(img_array, labels_txt):
        fig1, ax1 = plt.subplots(2,3)
        plot_img = []
        plot_lbl = []
        for i in range(0,6):
            img_indx = np.random.randint(0,len(img_array)-1)
            img = img_array[img_indx]
            img_label = labels_txt[img_indx]
            plot_img.append(img)
            plot_lbl.append(img_label)
        
        fig1, ax1 = plt.subplots(2,3)
        fig1.suptitle('\nDataset exploration\n')
        ax1[0][0].set_title(f'Img size: {plot_img[0].shape}\n Label: {plot_lbl[0]}')
        ax1[0][0].imshow(plot_img[0], cmap='gray')
        ax1[0][0].axis('off')
        ax1[0][1].set_title(f'Img size: {plot_img[1].shape}\n Label: {plot_lbl[1]}')
        ax1[0][1].imshow(plot_img[1], cmap='gray')
        ax1[0][1].axis('off')
        ax1[0][2].set_title(f'Img size: {plot_img[2].shape}\n Label: {plot_lbl[2]}')
        ax1[0][2].imshow(plot_img[2], cmap='gray')
        ax1[0][2].axis('off')
        ax1[1][0].set_title(f'Img size: {plot_img[3].shape}\n Label: {plot_lbl[3]}')
        ax1[1][0].imshow(plot_img[3], cmap='gray')
        ax1[1][0].axis('off')
        ax1[1][1].set_title(f'Img size: {plot_img[4].shape}\n Label: {plot_lbl[4]}')
        ax1[1][1].imshow(plot_img[4], cmap='gray')
        ax1[1][1].axis('off')
        ax1[1][2].set_title(f'Img size: {plot_img[5].shape}\n Label: {plot_lbl[5]}')
        ax1[1][2].imshow(plot_img[5], cmap='gray')
        ax1[1][2].axis('off')
        fig1.tight_layout()
        plt.show()

