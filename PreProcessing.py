import numpy as np
from tqdm import tqdm
from skimage.color import rgb2gray
from skimage.transform import resize
from keras.utils import to_categorical

class pre_processing:
    
    def rgb_to_gray(img_array):
        gray_img_array = []
        gray_img = None
        for i in tqdm(range(0,len(img_array)),'Converting RGB images to gray-scale'):
            gray_img = img_array[i]
            if len(img_array[i].shape) > 2:
                gray_img = rgb2gray(gray_img)
            gray_img_array.append(gray_img)
        return gray_img_array

    def resize_images(x_size, y_size, img_array):
        rs_img_array = []
        rs_img = None
        for i in tqdm(range(0,len(img_array)),f'Resizing images to {x_size}px x {y_size}px'):
            rs_img = resize(img_array[i], (x_size,y_size), anti_aliasing=True)
            rs_img_array.append(rs_img)
        return rs_img_array

    def image_normalization(img_array):
        norm_img_array = []
        act_img = None
        for i in tqdm(range(0,len(img_array)),'Normalizing images'):
            act_img = img_array[i]
            if np.max(act_img)>1:
                act_img = act_img/255
            norm_img_array.append(act_img)
        return(norm_img_array)

    def get_input_shape(array, type_data):
        
        print(f'Getting the correct input shape for {type_data}...')
        if type_data == 'image array input':
            gs_array = np.array(array)
            gs_array = np.expand_dims(gs_array,-1)
        elif type_data == 'labels':
            gs_array = np.array(array)
        print(gs_array.shape)

        return gs_array

