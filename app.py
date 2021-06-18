from flask import Flask,render_template,request
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = int(prediction[0])
    if output>100:
        output = 100
    return render_template('index.html', prediction_text='Predicted Student Mark :  {}'.format(output))


if __name__ == "__main__":
    app.run(debug=False)# -*- coding: utf-8 -*-