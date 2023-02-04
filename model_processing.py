import os
import numpy as np
import seaborn as sn
import tensorflow as tf
from tensorflow import keras
from keras.models import Model
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, recall_score
from keras.layers import Input, Dense, Conv2D, BatchNormalization, Dropout, MaxPooling2D, Flatten

class CNN_Model:
        
    def train_model(train_img_array, train_lbls):

        def model(train_array,train_labels):
            num_classes = len(np.unique(train_labels))

            i = Input(shape=train_array[0].shape)
            x = Conv2D(filters=32, kernel_size=(3,3), activation='relu',padding='same')(i)
            x = BatchNormalization()(x)
            x = Conv2D(filters=32, kernel_size=(3,3), activation='relu',padding='same')(x)
            x = BatchNormalization()(x)
            x = MaxPooling2D(pool_size=(2,2))(x)

            x = Conv2D(filters=64, kernel_size=(3,3), activation='relu',padding='same')(x)
            x = BatchNormalization()(x)
            x = Conv2D(filters=64, kernel_size=(3,3), activation='relu',padding='same')(x)
            x = BatchNormalization()(x)
            x = Conv2D(filters=64, kernel_size=(3,3), activation='relu',padding='same')(x)
            x = BatchNormalization()(x)
            x = MaxPooling2D(pool_size=(2,2))(x)

            x = Conv2D(filters=128, kernel_size=(3,3), activation='relu',padding='same')(x)
            x = BatchNormalization()(x)
            x = MaxPooling2D(pool_size=(2,2))(x)

            x = Flatten()(x)
            x = Dense(units=1000, activation='relu')(x)
            x = Dropout(0.2)(x)
            x = Dense(units=500, activation='relu')(x)
            x = Dropout(0.2)(x)
            x = Dense(units=20,activation='relu')(x)
            # x = Dropout(0.1)(x)
            x = Dense(num_classes, activation='softmax')(x)

            model = Model(i,x)
            model.compile(optimizer='adam',loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            result = model.fit(x=train_array,y=train_labels,epochs=30)
            model.save('./saved models/CNN_SavedModel.h5')
            np.save('./saved train-history/CNN_SavedTrainHistory.npy',result.history)

        if os.path.exists('./saved models/CNN_SavedModel.h5')==False:
            if os.path.exists('./saved models') == False:
                os.mkdir('./saved models')
                os.mkdir('./saved train-history')
                model(train_img_array,train_lbls)
            else:
                # WantToTrain = str(input('\nIf you want to train the model, write "True": '))
                # if WantToTrain=='True':
                model(train_img_array,train_lbls)
        elif os.path.exists('./saved models/CNN_SavedModel.h5')==True:
            WantToTrain = str(input('\nIf you want to train the model again, write "True": '))
            if WantToTrain=='True':
                print('\nTraining the model...')
                model(train_img_array,train_lbls)
            else:
                print('\nThe prediction model used is the one saved before.\n')


    def get_train_performance_metrics():
        
        ModelHistory =np.load('./saved train-history/CNN_SavedTrainHistory.npy',allow_pickle='TRUE').item()
        fig1, ax1=plt.subplots(1,2)
        fig1.suptitle('Model evaluation')
        ax1[0].set_title('Accuracy per epoch')
        ax1[0].plot(ModelHistory['accuracy'],label='Accuracy')
        ax1[0].set_xlabel('Epoch')
        ax1[0].set_ylabel('Accuracy')
        ax1[0].legend()
        ax1[0].grid(True)
        ax1[1].plot(ModelHistory['loss'],label='Loss',color='orange')
        ax1[1].set_title('Loss per epoch')
        ax1[1].set_xlabel('Epoch')
        ax1[1].set_ylabel('Loss')
        ax1[1].legend()
        ax1[1].grid(True)
        fig1.tight_layout()
        plt.show()


    def model_prediction():
        return