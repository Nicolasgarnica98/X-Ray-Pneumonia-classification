# Chest X-Ray Image Pneumonia classification

***
###1. Introduction
The goal of this project is to build a CNN capable of detecting pneumonia in chest x-ray images and classify them by virus/bacteria pneumonia and normal chest x-ray. The project is built in different python scripts that will handle the image pre-processing, database download and sorting, model experimentation and app setup.

This project is built with the Tensorflow 2.X framework in Python 3.9, and it is being deployed as a web app by using a simple API interface with the popular Flask framework, in which I decided to add a visual interface made in HTML.

###2. Try it out! (Run the web app)
There are two ways in which you can try my app. In both of them the app will be deployed on the port 5000 and will show the following console output:
```console
 * Serving Flask app 'app'
 * Debug mode: on
WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.
 * Running on all addresses (0.0.0.0)
 * Running on http://<your IP adress #1>:5000
 * Running on http://<your IP adress #2>:5000
Press CTRL+C to quit
 * Restarting with stat
```  
You will have to click on the first IP adress+port:5000 that appears under ```Running on all addresses (0.0.0.0) ``` , this will open a new tab on your predetermined web browser in which you will be facing the following interface:

<div style="text-align: center;">
<img src="static\UI.jpg" alt="ui_image"/>
</div>

You can try the app by manually downloading any chest x-ray image (normal or pneumonia-infected) from google on your computer and then load it by clicking on the "Choose File" button. Once the image is loaded, you will be able to see the name of the file and then it is possible to make a prediction by clicking on the "Predict Image" button. After a couple of seconds the prediction will be shown as a label: _Normal_ , _Virus Pneumonia_ or _Bacteria Pneumonia_.

####2.1. Local run
To locally run the app, you can pull this repository and then install the required libraries:
```console
pip install -r requirements.txt
```
Now you only have to run the ```app.py``` script and follow the instructions of section 2.

####2.2. Docker image
Alternatively, I made a docker image in order to avoid any inconveninece during the enviroment set-up. To run the image you can pull it from docker hub by running the command:
```console
docker pull nicolasgarncia/pneumonia_classification_app:V1.0
```
Make sure to have docker installed and running. To run the image:
```console
docker run -p <your port>:5000 nicolasgarncia/pneumonia_classification_app:V1.0
```
Make sure to have ```your port``` available for the app to deploy. This port is the one that will be mapped with the exposed port 5000 of the image. When the image is up and running you only have to follow the steps on section 2.

###3. How to explore and tune the model

###4. View more
