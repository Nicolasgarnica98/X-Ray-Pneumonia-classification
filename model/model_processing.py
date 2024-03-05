import os
import numpy as np
import seaborn as sn
import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from keras import regularizers
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from keras.layers import Input, Dense, Conv2D, BatchNormalization, Dropout, MaxPooling2D, Flatten


class CNN_Model():

    def __init__(self, model_name, epochs,):
        self.model_name = model_name
        self.epochs = epochs

    def train_model(self, input_shape, train_img, val_img, train_lbl, val_lbl):

        num_classes = len(np.unique(train_lbl))

        i = Input(shape=input_shape)
        x = Conv2D(filters=32, kernel_size=(3,3), activation='relu',padding='same', kernel_regularizer=regularizers.l2(0.01))(i)
        x = BatchNormalization()(x)
        x = MaxPooling2D(pool_size=(2,2))(x)

        x = Conv2D(filters=64, kernel_size=(3,3), activation='relu',padding='same', kernel_regularizer=regularizers.l2(0.02))(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D(pool_size=(2,2))(x)

        x = Conv2D(filters=128, kernel_size=(3,3), activation='relu',padding='same', kernel_regularizer=regularizers.l2(0.04))(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D(pool_size=(2,2))(x)

        x = Flatten()(x)
        x = Dense(units=1024,activation='relu')(x)
        x = Dropout(0.2)(x)
        x = Dense(units=256,activation='relu')(x)
        x = Dropout(0.1)(x)
        x = Dense(num_classes, activation='softmax')(x)

        model = Model(i,x)
        model.compile(optimizer='adam',loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        result = model.fit(train_img,train_lbl, epochs=self.epochs, validation_data=(val_img,val_lbl), batch_size=32)
        model.save(f'./model/saved models/{self.model_name}_SavedModel.h5')
        np.save(f'./model/saved train-history/{self.model_name}_SavedTrainHistory.npy',result.history)


    def get_train_performance_metrics(self):
        
        ModelHistory =np.load(f'./model/saved train-history/{self.model_name}_SavedTrainHistory.npy',allow_pickle='TRUE').item()
        fig1, ax1=plt.subplots(2,1, sharex=True)
        fig1.suptitle(f'{self.model_name} evaluation')
        fig1.subplots_adjust(hspace=0)
        ax1[0].plot(ModelHistory['accuracy'],label='Accuracy',color='#3740a6',linewidth=2)
        ax1[0].plot(ModelHistory['val_accuracy'],label='Val Accuracy', color='#66e3be',linewidth=2)
        ax1[0].set_ylabel('Accuracy')
        ax1[0].legend()
        ax1[0].grid(True)
        ax1[1].plot(ModelHistory['loss'],label='Loss',color='#3740a6',linewidth=2)
        ax1[1].plot(ModelHistory['val_loss'],label='Val Loss', color='#66e3be',linewidth=2)
        ax1[1].set_xlabel('Epoch')
        ax1[1].set_ylabel('Loss')
        ax1[1].legend()
        ax1[1].grid(True)
        fig1.tight_layout()
        plt.show()


    def model_prediction(self, img_array, lbl):
        model = tf.keras.models.load_model(f'./model/saved models/{self.model_name}_SavedModel.h5')
        prob_predictions = model.predict(img_array)
        predictions = prob_predictions.argmax(axis=1)
        predictions.astype(int)

        cm = confusion_matrix(lbl,predictions)
        plt.title('Confusion matrix')
        sn.set(font_scale=1.4)
        x_axis_labels = ['Normal', 'Virus', 'Bacteria']
        y_axis_labels = ['Normal', 'Virus', 'Bacteria']
        conf = sn.heatmap(cm, annot=True, annot_kws={'size':8}, cmap='Blues',xticklabels=x_axis_labels, yticklabels=y_axis_labels)
        conf.set(xlabel='Predicted class', ylabel='True class')
        conf.tick_params(left=True, bottom=True)
        plt.tight_layout()
        plt.show()
        metrics = classification_report(lbl,predictions, target_names= x_axis_labels)
        print(metrics)
        print("Loss of the model is - " , model.evaluate(img_array,lbl)[0])
        print("Accuracy of the model is - " , np.round(model.evaluate(img_array,lbl)[1]*100,3) , "%")

 
 
    def get_model_summary(self):
           model = tf.keras.models.load_model(f'./model/saved models/{self.model_name}_SavedModel.h5')
           print(model.summary())



