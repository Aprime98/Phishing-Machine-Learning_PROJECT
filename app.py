from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np


#create flask app
app = Flask(__name__)


#load Model
model = pickle.load(open('model.pkl', 'rb'))

#render default webpage
@app.route('/')
def index():
    return render_template('index.html')

#Use model to predict
@app.route('/predict', methods=['POST'])
def predict():
    #get data from form
    data = request.form.get('data')
    #convert data to numpy array
    data = np.array([data])
    #predict
    prediction = model.predict(data)
    #return prediction
    return render_template('index.html', prediction_text='Prediction: {}'.format(prediction))


if __name__ == '__main__':
    app.run(debug=True)