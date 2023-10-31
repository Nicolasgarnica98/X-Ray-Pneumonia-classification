from flask import Flask, render_template
import subprocess
import tensorflow as tf


app = Flask(__name__)
model = tf.keras.models.load_model(f'./saved models/CNN_64px_NN512_SavedModel.h5')

@app.route('/')
def home():
    return render_template('templates/home.html')


if __name__=='__app__':
    app.run(debug=True)