import pickle
from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd


app = Flask(__name__) ##__name__ is the starting point from which the code will run.

#Load the model
regmodel =pickle.load(open('regmodel.pkl','rb')) #rb -> read bite mode
scalar = pickle.load(open('scaling.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict_api',methods=['POST'])

def predict_api():
    data = request.json['data'] #the input is stored in json format in 'data',when we hit predict_api the input is called and predicted
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))
    #as we saw in the Linear_regression.ipynb, single data should be sent for prediction.
    #we need to reshape the data for the correct format. We can only reshape array, so first we are changing it to array and then reshaping.

    new_data = scalar.transform(np.array(list(data.values())).reshape(1,-1))
    output = regmodel.predict(new_data)
    print(output[0]) #the output will be a 2 dimensional array, we are taking just the first value.
    return jsonify(output[0])

@app.route('/predict',methods=['POST'])
def predict():
    data=[float(x) for x in request.form.values()]
    final_input = scalar.transform(np.array(data).reshape(1,-1))
    print(final_input)
    output=regmodel.predict(final_input)[0]
    return render_template("home.html",prediction_text="The House price is {}".format(output))


if __name__ == "__main__":
    app.run(debug=True)