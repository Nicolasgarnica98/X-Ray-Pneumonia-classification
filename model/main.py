#Author: Nicolas Garnica
#Classification of X-Ray images taken from different pacients with Pneumonia (Bacteria, Virus).

#Library import
import glob
import os
import tensorflow as tf 
from get_dataset import get_dataset
from pre_processing import pre_processing
from model_processing import CNN_Model

#Main method: it will call all the necessary functions from the other scripts in order to train/validate or test
#a model. in case of training, it will create an instance of the CNN_Model class and train the data.
def main():
    
    #if there is no dataset, it will automatically download the dataset from OneDrive into the main folder
    #and unzip it
    if os.path.exists('./dataset') == False:
        compressed_dataset = get_dataset.download('https://1drv.ms/u/s!Aocxj1Hi_hVIlu97MFww798zGMnF0g?e=f3jRsO')
        get_dataset.unzip_dataset(compressed_dataset)
        df_img = glob.glob(os.path.join('./dataset/chest_xray/','*.jpeg'))
        get_dataset.data_class_balance(df_img)
        df_img = glob.glob(os.path.join('./dataset/chest_xray/','*.jpeg'))
        get_dataset.divide_dataset_in_folders(df_img)

    #Define a model name to be trained/loaded
    #Model parameters
    model_name = str(input('Insert a name for the model to load/train: '))
    # epochs = int(input('Insert a number of epochs: '))
    # image_resize = int(input('Insert size value for the image reshape (N x N): '))
    # batch_size = int(input('Insert batch size: '))
    Actual_Model = None
    epochs = 30
    image_resize = 64

    #If training is true
    def train_pipeline(model):

        #Model is a CNN model
        if isinstance(model,CNN_Model):
            #Load data names from the train and val folders
            df_train = glob.glob(os.path.join('./dataset/train/','*.jpeg'))
            print('Train data')
            print(len(df_train))
            #Get labels (targets) for each image. it will return integer encoded labels and txt labels.
            train_lbl, train_lbl_txt = get_dataset.get_labels(df_train)

            df_val = glob.glob(os.path.join('./dataset/val/','*.jpeg'))
            print('Validation data')
            print(len(df_val))
            val_lbl, val_lbl_txt = get_dataset.get_labels(df_val)

            #Load images from the path names
            train_IMG = get_dataset.load_images(df_train)
            val_IMG = get_dataset.load_images(df_val)
            # Explore the train dataset
            get_dataset.data_exploration(train_IMG,train_lbl_txt)

            #Pre-processing: RGB2GRAY transformations to images in non-graysacale color space.
            #Regularization of images that have values outside of [0,1]
            #Reshaping the image for size uniformity
            train_IMG = pre_processing.rgb_to_gray(train_IMG)
            train_IMG = pre_processing.resize_images(image_resize,image_resize,train_IMG)
            train_IMG = pre_processing.image_normalization(train_IMG)

            #Change the input array shape for preparing it for the CNN input
            train_IMG = pre_processing.get_input_shape(train_IMG,'image array input')
            train_lbl = pre_processing.get_input_shape(train_lbl,'labels')

            val_IMG = pre_processing.rgb_to_gray(val_IMG)
            val_IMG = pre_processing.resize_images(image_resize,image_resize,val_IMG)
            val_IMG = pre_processing.image_normalization(val_IMG)

            #Change the input array shape for preparing it for the CNN input
            val_IMG = pre_processing.get_input_shape(val_IMG,'image array input')
            val_lbl = pre_processing.get_input_shape(val_lbl,'labels')

            #Train the model
            model.train_model(input_shape=(image_resize,image_resize,1), train_lbl=train_lbl, train_img=train_IMG, val_img=val_IMG, val_lbl=val_lbl)


    choose_model = str(input('Choose model from -> CNN_Model: '))
    if choose_model == 'CNN_Model':
        Actual_Model = CNN_Model(model_name=model_name, epochs=epochs)

    #Check for the name of the model. If it exist it will ask if train again or not, if not, it will
    #train the model and saved with the provided name. If no folders, it will create two folders, one for the
    #saved model and other for the model's training history.
    if os.path.exists(f'./model/saved models/{model_name}_SavedModel.h5') == False:
        if os.path.exists('./model/saved models')==False:
            os.mkdir('./model/saved models')
            os.mkdir('./model/saved train-history')
            train_pipeline(model=Actual_Model)
        else:
            train_pipeline(model=Actual_Model)
    elif os.path.exists(f'./model/saved models/{model_name}_SavedModel.h5') == True:
        want_to_train = input(f'If you want to re-train the model "{model_name}", write True: ')
        if want_to_train == 'True':
            train_pipeline(model=Actual_Model)
    
    #Load the model's history and visualization of loss and accuracy.
    Actual_Model.get_train_performance_metrics()
    Actual_Model.get_model_summary()


    #Test the model
    #Load the test dataset and load images
    df_test = glob.glob(os.path.join('./dataset/test/','*.jpeg'))
    print(f'Testing model "{model_name}"...')
    #Load test images
    test_IMG = get_dataset.load_images(df_test)
    #Get labels for the images
    test_lbl = get_dataset.get_labels(df_test)[0]
    #Preprocessing the images form the test data
    if choose_model == 'CNN_Model':
        test_IMG = pre_processing.rgb_to_gray(test_IMG)
        test_IMG = pre_processing.image_normalization(test_IMG)
        test_IMG = pre_processing.resize_images(image_resize,image_resize,test_IMG)
        #Change the input array shape for preparing it for model prediction input
        test_IMG = pre_processing.get_input_shape(test_IMG,'image array input')
        test_lbl = pre_processing.get_input_shape(test_lbl,'labels')

    #Using the CPU, do predictions with the loaded model.
    with tf.device('/CPU:0'):
        Actual_Model.model_prediction(test_IMG, test_lbl)

if __name__ =='__main__':
    main()
