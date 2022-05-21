from flask import Flask,render_template,request
import pickle
import numpy as np
import pandas as pd

#file = open("./gradient_boosting_regressor_model.pkl", 'rb')
model1 = pickle.load(open('insurance_premium_prediction_model.pkl','rb'))

data = pd.read_csv('insurance.csv')
data.head()
app = Flask(__name__)
@app.route('/')
def index():
    sex = sorted(data['sex'].unique())
    smoker = sorted(data['smoker'].unique())
    region = sorted(data['region'].unique())
    return render_template('index.html', sex=sex, smoker=smoker, region=region)


@app.route('/predict', methods=['POST'])
def predict():
    age = int(request.form.get('age'))
    bmi = float(request.form.get('BMI'))
    children = int(request.form.get('children'))
    sex = request.form.get('sex')
    smoker = request.form.get('smoker')
    region = request.form.get('region')

    prediction = model1.predict(pd.DataFrame([[age,bmi,children,sex,smoker,region]],
                                            columns=['age',  'bmi', 'children','sex', 'smoker', 'region']))

    return str(prediction[0])


if __name__ == '__main__':
    app.run(debug=True)