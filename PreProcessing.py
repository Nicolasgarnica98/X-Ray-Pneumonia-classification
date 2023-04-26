import keras
import numpy as np
from tqdm import tqdm
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.transform import resize



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
    
    def image_normalization_single(img):
        act_img = img
        #Check if image is not already normalized
        if np.max(act_img)>1:
            act_img = act_img/255
        norm_img = act_img
        return norm_img

    #Change image array dimension in order to fit the Tnesorflow standarized input shape
    def get_input_shape(array, type_data):
        
        print(f'Getting the correct input shape for {type_data}...')
        if type_data == 'image array input':
            gs_array = np.array(array)
            gs_array = np.expand_dims(gs_array,-1)
        elif type_data == 'labels':
            gs_array = np.array(array)
        print(gs_array.shape)
        return gs_array


class My_Custom_Generator(keras.utils.Sequence) :
  
  def __init__(self, image_filenames, labels, batch_size, x_size, y_size) :
    self.image_filenames = image_filenames
    self.labels = labels
    self.batch_size = batch_size
    self.x_size = x_size
    self.y_size = y_size
    
    
  def __len__(self) :
    return (np.ceil(len(self.image_filenames) / float(self.batch_size))).astype(int)

  
  def __getitem__(self, idx) :
    batch_x = self.image_filenames[idx * self.batch_size : (idx+1) * self.batch_size]
    batch_y = self.labels[idx * self.batch_size : (idx+1) * self.batch_size]
    
    return np.array([pre_processing.image_normalization_single(resize(rgb2gray(imread(str(file_name))), (self.x_size, self.y_size, 3))) for file_name in batch_x]), np.array(batch_y)

        

