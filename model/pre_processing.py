import numpy as np
from tqdm import tqdm
from skimage.color import rgb2gray
from skimage.transform import resize
from keras.preprocessing.image import ImageDataGenerator


class pre_processing:
    
    #Rgb to gray scaled images
    def rgb_to_gray(img_array):
        gray_img_array = []
        gray_img = None
        for i in tqdm(range(0,len(img_array)),'Converting RGB images to gray-scale'):
            gray_img = img_array[i]
            #Check if image is not grayscaled.
            if len(img_array[i].shape) > 2:
                gray_img = rgb2gray(gray_img)
            gray_img_array.append(gray_img)
        return gray_img_array
    
    #Resize images to the desired size
    def resize_images(x_size, y_size, img_array):
        rs_img_array = []
        rs_img = None
        for i in tqdm(range(0,len(img_array)),f'Resizing images to {x_size}px x {y_size}px'):
            rs_img = resize(img_array[i], (x_size,y_size))
            rs_img_array.append(rs_img)
        return rs_img_array

    #Image normalization between [0,1]
    def image_normalization(img_array):
        norm_img_array = []
        act_img = None
        for i in tqdm(range(0,len(img_array)),'Normalizing images'):
            act_img = img_array[i]
            #Check if image is not already normalized
            if np.max(act_img)>1:
                act_img = act_img/255
            norm_img_array.append(act_img)
        return(norm_img_array)

    #Change image array dimension in order to fit the Tnesorflow standarized input shape
    def get_input_shape(array, type_data):
        if type_data == 'image array input':
            gs_array = np.array(array, dtype=np.float32)
            gs_array = np.expand_dims(gs_array,-1)
        elif type_data == 'labels':
            gs_array = np.array(array)
        return gs_array

    #Image normalization between [0,1]
    def image_preprocessing_single(img):
        #Check if image is not already normalized
        if np.max(img)>1:
            img = img/255
        return img
        
class batch_loader():

    def __init__(self):
        pass

    def image_data_generator():
        train_datagen = ImageDataGenerator(preprocessing_function=pre_processing.image_preprocessing_single)
        generator = train_datagen.flow_from_directory()

#Save the preprocessing methods
import pickle
pickle.dump(pre_processing,open('./model/pre_processing.pkl','wb'))