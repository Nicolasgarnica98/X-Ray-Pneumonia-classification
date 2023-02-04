import numpy as np
from tqdm import tqdm
from skimage.color import rgb2gray
from skimage.transform import resize

class pre_processing:
    
    def rgb_to_gray(img_array):
        gray_img_array = []
        gray_img = None
        for i in tqdm(range(0,len(img_array)),'Converting RGB images to gray-scaled'):
            gray_img = img_array[i]
            if len(img_array[i].shape) > 2:
                gray_img = rgb2gray(gray_img)
            gray_img_array.append(gray_img)
        return gray_img_array

    def resize_images(x_size, y_size, img_array):
        rs_img_array = []
        rs_img = None
        for i in tqdm(range(0,len(img_array)),f'Resizing images to {x_size}px x {y_size}px'):
            rs_img = resize(img_array[i], (x_size,y_size))
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

    # def get_CNN_input_shape(img_array):

