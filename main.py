import glob
import os
import tensorflow as tf 
from Get_dataset import get_dataset
from PreProcessing import pre_processing
from model_processing import CNN_Model


def main():
    model_name = str(input('Insert a name for the model to load/train: '))
    compressed_dataset = get_dataset.download('https://drive.google.com/file/d/1OimFddxkp3Y9hYEdtFaFUst0QVLl5zXX/view?usp=sharing')
    get_dataset.unzip_dataset(compressed_dataset)

    Actual_CNN_Model = CNN_Model(model_name=model_name)

    def train_pipeline():
        df_train_normal = glob.glob(os.path.join('dataset/train/NORMAL','*.jpeg'))
        df_train_pneumonia = glob.glob(os.path.join('dataset/train/PNEUMONIA','*.jpeg'))
        df_train = df_train_normal + df_train_pneumonia


        train_fullLabels = get_dataset.get_labels(df_train)
        train_lbl = train_fullLabels[0]
        train_lbl_txt = train_fullLabels[1]
        train_IMG = get_dataset.load_images(df_train)
        # get_dataset.data_exploration(train_IMG,train_lbl_txt)

        #Pre-processing
        train_IMG = pre_processing.rgb_to_gray(train_IMG)
        train_IMG = pre_processing.resize_images(50,50,train_IMG)
        train_IMG = pre_processing.image_normalization(train_IMG)

        train_IMG = pre_processing.get_input_shape(train_IMG,'image array input')
        train_lbl = pre_processing.get_input_shape(train_lbl,'labels')
        print(train_IMG[0].shape)
        # get_dataset.data_exploration(train_IMG,train_lbl_txt)
        print(train_lbl.shape)

        Actual_CNN_Model.train_model(train_IMG,train_lbl)

    if os.path.exists(f'./saved models/{model_name}_SavedModel.h5') == False:
        if os.path.exists('./saved models')==False:
            os.mkdir('./saved models')
            os.mkdir('./saved train-history')
            train_pipeline()
        else:
            train_pipeline()
    elif os.path.exists(f'./saved models/{model_name}_SavedModel.h5') == True:
        want_to_train = input(f'If you want to re-train the model "{model_name}", write True: ')
        if want_to_train == 'True':
            train_pipeline()
        
    Actual_CNN_Model.get_train_performance_metrics()


    #Test the model
    #Predictions
    df_test_normal = glob.glob(os.path.join('dataset/test/NORMAL','*.jpeg'))
    df_test_pneumonia = glob.glob(os.path.join('dataset/test/PNEUMONIA','*.jpeg'))
    df_test = df_test_normal + df_test_pneumonia
    print(f'Testing model "{model_name}"...')
    test_IMG = get_dataset.load_images(df_test)
    test_lbl = get_dataset.get_labels(df_test)[0]
    test_IMG = pre_processing.rgb_to_gray(test_IMG)
    test_IMG = pre_processing.image_normalization(test_IMG)
    test_IMG = pre_processing.resize_images(50,50,test_IMG)

    test_IMG = pre_processing.get_input_shape(test_IMG,'image array input')
    test_lbl = pre_processing.get_input_shape(test_lbl,'labels')

    with tf.device('/CPU:0'):
        Actual_CNN_Model.model_prediction(test_IMG, test_lbl)

if __name__ =='__main__':
    main()
