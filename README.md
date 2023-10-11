# Chest X-Ray Image Pneumonia classification

***
### Introduction

Pneumonia is a common disease that affects the lungs of the patient, generating inflammation and obstruction of the intake airways, which can lead to suffocation and death. The principal reason behind this disease is the contact of external microorganisms with the lung parenchyma, and the action of the immune system to fight the infection is what causes the inflammation. The most common microorganisms found responsible for the pneumonia infection are either bacterial or viral in origin; these microorganisms are present in the air, which makes them easy to inhale while breathing.

Evaluation and diagnosis of the disease are always important parts of the treatment pipeline. Pneumonia is commonly diagnosed with respect to the patient's symptoms, such as cough, fever, general weakness, and difficulty breathing (choking), and x-ray images taken from the patient. As shown in the picture below, the congestion of the bronchi and alveoli will cause a higher intensity zone when taking an x-ray image from an infected patient, which helps to diagnose the pneumonia.

<br>
<p align="center">
  <img width="600" height="350" src="Document_resources/pneumonia_example.jpg">
</p>
<br>
The treatment for pneumonia will depend on the origin of the pathogen, and to determine the origin, the doctor will request a tissue sample or mucus sample from the lungs in order to make a bacteria culture or a viral detection. This process takes some time and is uncomfortable for the patient. Determining the kind of pathogen is essential because if it is bacteria, it could be treated with antibiotics, which would not work with virus infections.

In this project, I will create a CNN model for classifying x-ray images from people with pneumonia in order to improve the diagnosis time and patient´s comfort.
<br>
### Dataset exploration
The original dataset can be found in: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia. It contains 3 classes divided by folders (train, test, val):
- Virus pneumonia
- Bacteria pneumonia
- Normal (No pneumonia)

I downloaded the dataset and merged all the folders into one unique folder in order to split the dataset in a random way into train, val and test folders. Then I uploaded the folder into my personal Onedrive for later image analysis.

````

def divide_dataset_in_folders(df_img):
        os.mkdir('./dataset/train')
        os.mkdir('./dataset/test')
        os.mkdir('./dataset/val')
        df_train, df_test = train_test_split(df_img,test_size=0.15)
        df_train, df_val = train_test_split(df_train,test_size=0.1)

        folder_array = ['./dataset/train', './dataset/test', './dataset/val']
        df_array = [df_train,df_test,df_val]
        for i in tqdm(range(0,len(folder_array)),'Moving files: '):
            for file in df_array[i]:
                source = file
                shutil.move(source, folder_array[i])

````

For better understanfing of the data I am dealing with it was necessary to visualize some random images of my dataset:

 ````

def data_exploration(img_array, labels_txt):
        fig1, ax1 = plt.subplots(2,3)
        plot_img = []
        plot_lbl = []
        for i in range(0,6):
            img_indx = np.random.randint(0,len(img_array)-1)
            img = img_array[img_indx]
            img_label = labels_txt[img_indx]
            plot_img.append(img)
            plot_lbl.append(img_label)
        
        fig1, ax1 = plt.subplots(2,3)
        fig1.suptitle('\nDataset exploration\n')
        ax1[0][0].set_title(f'Img size: {plot_img[0].shape}\n Label: {plot_lbl[0]}')
        ax1[0][0].imshow(plot_img[0], cmap='gray')
        ax1[0][0].axis('off')
        ax1[0][1].set_title(f'Img size: {plot_img[1].shape}\n Label: {plot_lbl[1]}')
        ax1[0][1].imshow(plot_img[1], cmap='gray')
        ax1[0][1].axis('off')
        ax1[0][2].set_title(f'Img size: {plot_img[2].shape}\n Label: {plot_lbl[2]}')
        ax1[0][2].imshow(plot_img[2], cmap='gray')
        ax1[0][2].axis('off')
        ax1[1][0].set_title(f'Img size: {plot_img[3].shape}\n Label: {plot_lbl[3]}')
        ax1[1][0].imshow(plot_img[3], cmap='gray')
        ax1[1][0].axis('off')
        ax1[1][1].set_title(f'Img size: {plot_img[4].shape}\n Label: {plot_lbl[4]}')
        ax1[1][1].imshow(plot_img[4], cmap='gray')
        ax1[1][1].axis('off')
        ax1[1][2].set_title(f'Img size: {plot_img[5].shape}\n Label: {plot_lbl[5]}')
        ax1[1][2].imshow(plot_img[5], cmap='gray')
        ax1[1][2].axis('off')
        fig1.tight_layout()
        plt.show()y   
 ````
<p align="center">
  <img width="600" height="450" src="Document_resources/Dataset_exploration.png">
</p>
<br>

### Image pre-processing
From the dataset exploration we can see that the dataset contains images in grayscale and RGB images, all of them with different image sizes (width x lenght). As the RGB images do not show any relevant or additional information to the analysis, I decided to transform all images to grayscale color space.

````

from skimage.color import rgb2gray 

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

````

From the rgb2gray transformation, it was necessary then to normalize the images by dividing by 255, in order to get values between [0,1]. After the rgb2gray transformation I noted that some images were already normalized in the desired range, so i used the code below to normalize the rest of the images:

````

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

````

As a final pre-processing transformation, it was necessary to resize all images to an unique squared size in order to fit the input shape of the CNN model. The size of the image will be decided mostly by the compute capacity of the machine, as bigger image sizes require much more compute power, specially when training a CNN. the image size choosen for my forst two experiments were 32px and 128px, and it was performed with the code below:

````

#Resize images to the desired size
    def resize_images(x_size, y_size, img_array):
        rs_img_array = []
        rs_img = None
        for i in tqdm(range(0,len(img_array)),f'Resizing images to {x_size}px x {y_size}px'):
            rs_img = resize(img_array[i], (x_size,y_size))
            rs_img_array.append(rs_img)
        return rs_img_array

````

It was necessary to get the image ground truth labels in order to perform any kind of model. As the labels of the images were the filenames of each of them, I created a function to get the labels that will return an integer-coded label for each class {Virus = 1, Bacteria = 2, Normal = 0} and the string labels which i will use later for the presentation of the test results. It will print too, the samples per class.


````

def get_labels(df):
        labels = []
        txt_labels = []
        num_normal = 0
        num_bacteria = 0
        num_virus = 0
        for i in range(0,len(df)):
            if df[i].find('virus')!=-1:
                labels.append(1)
                txt_labels.append('Virus')
                num_virus += 1
            elif df[i].find('bacteria')!=-1:
                labels.append(2)
                txt_labels.append('Bacteria')
                num_bacteria += 1
            else:
                labels.append(0)
                txt_labels.append('Normal')
                num_normal += 1
        labels = np.array(labels)
        num_samples_per_class = {'virus':num_virus, 'bacteria':num_bacteria, 'Normal':num_normal}
        print(num_samples_per_class)
        
        return labels, txt_labels

````

<br>
### CNN Model and Image analysis

As the Tensorflow documentation states, it is important to get the correct shape for the input array, thus I used the code below to get the correct input shape for the labels and images:

````

def get_input_shape(array, type_data):
        
        print(f'Getting the correct input shape for {type_data}...')
        if type_data == 'image array input':
            gs_array = np.array(array)
            gs_array = np.expand_dims(gs_array,-1)
        elif type_data == 'labels':
            gs_array = np.array(array)
        print(gs_array.shape)
        return gs_array

````

#### Image size 32px model
The model used for the image size 32px was:

<p align="center">
  <img width="450" height="350" src="Document_resources/CNN_32px_test1_ModelSummary.jpg">
  <img width="470" height="350" src="Document_resources/CNN_32px_test1_trainPerformance.png">
</p>


After testing with the trained model and with the test images already pre-processed:

<p align="center">
  <img width="650" height="550" src="Document_resources/CNN_32px_test1.png">
  <img width="470" height="200" src="Document_resources/CNN_32px_Test1_metrics.jpg">
</p>
<br>

#### Image size 128px model

The model used for the image size 128px was:

<p align="center">
  <img width="600" height="900" src="Document_resources/CNN_128px_test1_ModelSummary.jpg">
  <img width="600" height="450" src="Document_resources/CNN_128px_test1_trainPerformance.png">
</p>


After testing with the trained model and with the test images already pre-processed:

<p align="center">
  <img width="650" height="550" src="Document_resources/CNN_128px_test1.png">
  <img width="470" height="180" src="Document_resources/CNN_128px_Test1_metrics.jpg">
</p>
<br>


### Discusion

After experimenting with both models, i got to the conclusion that the image size is an important parameter to consider as it shows that the bigger the image the better accuracy in general. It could be because of the small infected alveoli will vanish or combine with other regions of the image such as bones, heart or other sorrunding tissue when the image shrinks to an smaller size, the same way when the image is forced to reshape to a squared format, loosing information when shrinking from the original rectangular shape.

From the confussion matrix I deduce that the binnary classification between pneumonia vs normal is very good, with the best accuracy score on the test dataset from the 128px model (97% acc.). It is possible that bacteria and virus pneumonia infection are not distinguishable just by watching x-ray images. But the accuracy values for virus and bacteria suggest that they are separable, at least to a degree.

It seems that both models are suffering from over-fitting when hitting the 80% accuracy in the training session. Even when adding dropout layers for adding some randomness to the dense layers and L2 regularization to the CNN filters to penalize the weights. It seems that after hitting 80% on the val dataset the randomness of the dropout layer will influence the learning of the model.

 I consider now three estrategies to improve my results and performance of the model:
 - Add a batch generator to train batches of images, thus it could benefit my system compute power, allowing it to analyze bigger image sizes.
 - Check class invariance. While checking my dataset class distribution I noted that there are big differences in the number of samples per class:
 
<p align="center">
  <img width="450" height="130" src="Document_resources/dataset_size.jpg">
</p>

- More pre-processing options: I have consider to pad the images into a square frame, thus the image will not have to be forced into a squared shape while allowing the CNN to analyze the images with a "virtual same shape". With this method I expect to avoid information loss, leaving the infected alveoli structures the most intact possible.

