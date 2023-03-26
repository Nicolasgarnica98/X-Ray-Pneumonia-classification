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

    def __init__(self, model_name):
        self.model_name = model_name

    def train_model(self, train_array, train_labels,df_val_img, lbl_val):

        num_classes = len(np.unique(train_labels))

        i = Input(shape=train_array[0].shape)
        x = Conv2D(filters=128, kernel_size=(3,3), activation='relu',padding='same',kernel_regularizer=regularizers.l2(0.0001))(i)
        x = BatchNormalization()(x)
        # x = Conv2D(filters=32, kernel_size=(3,3), activation='relu',padding='same',kernel_regularizer=regularizers.l2(0.0001))(x)
        # x = BatchNormalization()(x)
        # x = MaxPooling2D(pool_size=(2,2))(x)

        x = Conv2D(filters=256, kernel_size=(3,3), activation='relu',padding='same',kernel_regularizer=regularizers.l2(0.0001))(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D(pool_size=(2,2))(x)

        # x = Conv2D(filters=128, kernel_size=(3,3), activation='relu',padding='same',kernel_regularizer=regularizers.l2(0.0001))(x)
        # x = BatchNormalization()(x)
        # x = MaxPooling2D(pool_size=(2,2))(x)

        x = Flatten()(x)
        x = Dense(units=1024, activation='relu')(x)
        x = Dropout(0.3)(x)
        # x = Dense(units=500,activation='relu')(x)
        # x = Dropout(0.3)(x)
        x = Dense(units=200,activation='relu')(x)
        x = Dropout(0.2)(x)
        x = Dense(units=50,activation='relu')(x)
        x = Dropout(0.1)(x)
        x = Dense(num_classes, activation='softmax')(x)

        model = Model(i,x)
        model.compile(optimizer='adam',loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        result = model.fit(x=train_array,y=train_labels,epochs=100,batch_size=32,validation_data=(df_val_img,lbl_val))
        model.save(f'./saved models/{self.model_name}_SavedModel.h5')
        np.save(f'./saved train-history/{self.model_name}_SavedTrainHistory.npy',result.history)


    def get_train_performance_metrics(self):
        
        ModelHistory =np.load(f'./saved train-history/{self.model_name}_SavedTrainHistory.npy',allow_pickle='TRUE').item()
        fig1, ax1=plt.subplots(1,2)
        fig1.suptitle(f'{self.model_name} evaluation')
        ax1[0].set_title('Accuracy per epoch')
        ax1[0].plot(ModelHistory['accuracy'],label='Accuracy')
        ax1[0].plot(ModelHistory['val_accuracy'],label='Val Accuracy')
        ax1[0].set_xlabel('Epoch')
        ax1[0].set_ylabel('Accuracy')
        ax1[0].legend()
        ax1[0].grid(True)
        ax1[1].plot(ModelHistory['loss'],label='Loss')
        ax1[1].plot(ModelHistory['val_loss'],label='Val Loss')
        ax1[1].set_title('Loss per epoch')
        ax1[1].set_xlabel('Epoch')
        ax1[1].set_ylabel('Loss')
        ax1[1].legend()
        ax1[1].grid(True)
        fig1.tight_layout()
        plt.show()


    def model_prediction(self, img_array, lbl):
        model = tf.keras.models.load_model(f'./saved models/{self.model_name}_SavedModel.h5')
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
        metrics = classification_report(lbl,predictions, target_names= x_axis_labels)
        print(metrics)
        plt.tight_layout()
        plt.show()
 
 
    def get_model_summary(self):
           model = tf.keras.models.load_model(f'./saved models/{self.model_name}_SavedModel.h5')
           print(model.summary())



#Transfer Learning from VGG16
class TransferLearning_VGG16():

    def __init__(self, model_name,epochs, input_shape):
        self.model_name = model_name
        self.epochs = epochs
        self.input_shape = input_shape #(hight, width, channels)

    def train_model(self, train_array, train_labels,df_val_img, lbl_val):
        from keras.applications import VGG16
        # Load the pre-trained VGG16 model
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=self.input_shape)

        # Freeze the layers of the pre-trained model
        for layer in base_model.layers:
            layer.trainable = False

        # Add a new classification layer on top of the pre-trained model
        model = tf.keras.models.Sequential([
            base_model,
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax')
        ])

        # Compile the model
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        # Train the model on a new dataset
        model.fit(train_array, train_labels, epochs=self.epochs, validation_data=(df_val_img, lbl_val))

    def get_train_performance_metrics(self):        
        ModelHistory =np.load(f'./saved train-history/{self.model_name}_SavedTrainHistory.npy',allow_pickle='TRUE').item()
        fig1, ax1=plt.subplots(1,2)
        fig1.suptitle(f'{self.model_name} evaluation')
        ax1[0].set_title('Accuracy per epoch')
        ax1[0].plot(ModelHistory['accuracy'],label='Accuracy')
        ax1[0].plot(ModelHistory['val_accuracy'],label='Val Accuracy')
        ax1[0].set_xlabel('Epoch')
        ax1[0].set_ylabel('Accuracy')
        ax1[0].legend()
        ax1[0].grid(True)
        ax1[1].plot(ModelHistory['loss'],label='Loss')
        ax1[1].plot(ModelHistory['val_loss'],label='Val Loss')
        ax1[1].set_title('Loss per epoch')
        ax1[1].set_xlabel('Epoch')
        ax1[1].set_ylabel('Loss')
        ax1[1].legend()
        ax1[1].grid(True)
        fig1.tight_layout()
        plt.show()


    def model_prediction(self, img_array, lbl):
        model = tf.keras.models.load_model(f'./saved models/{self.model_name}_SavedModel.h5')
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
        metrics = classification_report(lbl,predictions, target_names= x_axis_labels)
        print(metrics)
        plt.tight_layout()
        plt.show()
 
 
    def get_model_summary(self):
           model = tf.keras.models.load_model(f'./saved models/{self.model_name}_SavedModel.h5')
           print(model.summary())