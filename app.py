from flask import Flask, render_template, request, app, jsonify
import tensorflow as tf
from model.pre_processing import pre_processing
from model.get_dataset import get_dataset
import os

app = Flask(__name__)
model = tf.keras.models.load_model('./model/saved models/CNN_64px_NN512_SavedModel.h5')

@app.route('/', methods=['GET'])
def hello_word():
    return render_template('main.html')

@app.route('/', methods=['POST'])
def predict():
    
    imagefile= request.files['imagefile']
    print()
    image_path = "./" + imagefile.filename
    print(image_path)
    imagefile.save(image_path)
    
    if os.path.exists(image_path):
        image = get_dataset.load_images([image_path])
        image = pre_processing.rgb_to_gray(image)
        image = pre_processing.image_normalization(image)
        image = pre_processing.resize_images(x_size=64,y_size=64,img_array=image)
        image = pre_processing.get_input_shape(image, 'image array input')

        yhat = model.predict(image)
        predictions = yhat.argmax(axis=1)
        predictions.astype(int)
        if predictions[0] == 0:
            predictions = 'Normal'
        elif predictions[0] == 1:
            predictions = 'Virus pneumonia'
        else:
            predictions = 'Bacteria pneumonia'

        classification = f'Predicted class: {str(predictions)}'
        print(classification)
        os.remove(os.path.join(image_path))


    return render_template('main.html', prediction=classification)

if __name__=='__main__':
    port = os.environ.get("PORT",5000)
    app.run(debug=True, host="0.0.0.0", port=port)
