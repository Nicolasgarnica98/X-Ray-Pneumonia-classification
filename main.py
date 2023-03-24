#Author: Nicolas Garnica
#Classification of X-Ray images taken from different pacients with Pneumonia (Bacteria, Virus).

#Library import
import glob
import os
import tensorflow as tf 
from Get_dataset import get_dataset
from PreProcessing import pre_processing
from model_processing import CNN_Model

#Main method: it will call all the necessary functions from the other scripts in order to train/validate or test
#a model. in case of training, it will create an instance of the CNN_Model class and train the data.
def main():
    #Define a model name to be trained/loaded 
    model_name = str(input('Insert a name for the model to load/train: '))
    #if there is no dataset, it will automatically download the dataset from OneDrive into the main folder
    #and unzip it
    compressed_dataset = get_dataset.download('https://1drv.ms/u/s!Aocxj1Hi_hVIldsmWIMj9AcU2MH7hw?e=Z4FxiE')
    get_dataset.unzip_dataset(compressed_dataset)

    #It creates an instance of the CNN_Model with the name provided.
    Actual_CNN_Model = CNN_Model(model_name=model_name)

    #If training is true
    def train_pipeline():
        
        #Load data names from the train and val folders
        df_train = glob.glob(os.path.join('dataset/train/','*.jpeg'))
        print('Train data')
        print(len(df_train))
        #Get labels (targets) for each image. it will return integer encoded labels and txt labels.
        train_lbl, train_lbl_txt = get_dataset.get_labels(df_train)

        df_val = glob.glob(os.path.join('dataset/val/','*.jpeg'))
        print('Validation data')
        print(len(df_val))
        val_lbl, val_lbl_txt = get_dataset.get_labels(df_val)

        #Load images from the path names
        train_IMG = get_dataset.load_images(df_train)
        val_IMG = get_dataset.load_images(df_val)
        #Explore the train dataset
        get_dataset.data_exploration(train_IMG,train_lbl_txt)

        #Pre-processing: RGB2GRAY transformations to images in non-graysacale color space.
        #Regularization of images that have values outside of [0,1]
        #Reshaping the image for size uniformity
        train_IMG = pre_processing.rgb_to_gray(train_IMG)
        train_IMG = pre_processing.resize_images(32,32,train_IMG)
        train_IMG = pre_processing.image_normalization(train_IMG)

        #Change the input array shape for preparing it for the CNN input
        train_IMG = pre_processing.get_input_shape(train_IMG,'image array input')
        train_lbl = pre_processing.get_input_shape(train_lbl,'labels')

        val_IMG = pre_processing.rgb_to_gray(val_IMG)
        val_IMG = pre_processing.resize_images(32,32,val_IMG)
        val_IMG = pre_processing.image_normalization(val_IMG)

        #Change the input array shape for preparing it for the CNN input
        val_IMG = pre_processing.get_input_shape(val_IMG,'image array input')
        val_lbl = pre_processing.get_input_shape(val_lbl,'labels')

        #Train the CNN model with the instance of the model created before.
        Actual_CNN_Model.train_model(train_IMG,train_lbl,val_IMG,val_lbl)

    #Check for the name of the model. If it exist it will ask if train again or not, if not, it will
    #train the model and saved with the provided name. If no folders, it will create two folders, one for the
    #saved model and other for the model's training history.
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
    
    #Load the model's history and visualization of loss and accuracy.
    Actual_CNN_Model.get_train_performance_metrics()


    #Test the model
    #Load the test dataset and load images
    df_test = glob.glob(os.path.join('dataset/test/','*.jpeg'))
    print(f'Testing model "{model_name}"...')
    #Load test images
    test_IMG = get_dataset.load_images(df_test)
    #Get labels for the images
    test_lbl = get_dataset.get_labels(df_test)[0]
    #Preprocessing the images form the test data
    test_IMG = pre_processing.rgb_to_gray(test_IMG)
    test_IMG = pre_processing.image_normalization(test_IMG)
    test_IMG = pre_processing.resize_images(32,32,test_IMG)
    #Change the input array shape for preparing it for model prediction input
    test_IMG = pre_processing.get_input_shape(test_IMG,'image array input')
    test_lbl = pre_processing.get_input_shape(test_lbl,'labels')

    #Using the CPU, do predictions with the loaded model.
    with tf.device('/CPU:0'):
        Actual_CNN_Model.model_prediction(test_IMG, test_lbl)

if __name__ =='__main__':
    main()
