import numpy as np
import pandas as pd
import pickle

from flask import Flask, request, jsonify, render_template

from sklearn.preprocessing import StandardScaler


model = pickle.load(open('model.pkl', 'rb'))

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():

    

    fixed_acidity = request.form.get("Fixed_Acidity")
    residual_sugar = request.form.get("Residual_Sugar")
    free_sulfur_dioxide = request.form.get("Free_Sulphur_Dioxide")
    total_sulfur_dioxide = request.form.get("Total_Sulphur_Dioxide")
    alcohol = request.form.get("Alcohol")
   

    data1 = [fixed_acidity,residual_sugar,free_sulfur_dioxide,total_sulfur_dioxide,alcohol]
    
    df_test = pd.DataFrame([data1],columns = ['fixed_acidity','residual_sugar','free_sulfur_dioxide','total_sulfur_dioxide','alcohol'])
   
    print(df_test.dtypes)

    df_test['fixed_acidity'] = pd.to_numeric(df_test['fixed_acidity'], downcast="float")
    df_test['residual_sugar'] = pd.to_numeric(df_test['residual_sugar'], downcast="float")
    df_test['free_sulfur_dioxide'] = pd.to_numeric(df_test['free_sulfur_dioxide'], downcast="float")
    df_test['total_sulfur_dioxide'] = pd.to_numeric(df_test['total_sulfur_dioxide'], downcast="float")
    df_test['alcohol'] = pd.to_numeric(df_test['alcohol'], downcast="float")
    # predict with new data.
    prediction = model.predict(df_test)
    output = prediction

    if output == 4:
        return render_template('index.html', result='The quality of wine is 4!')
    elif output == 5:
        return render_template('index.html', result='The quality of wine is 5!')
    elif output == 6:
        return render_template('index.html', result='The quality of wine is 6!')
    elif output == 7:
        return render_template('index.html', result='The quality of wine is 7!')
    else:
        return render_template('index.html', result='The quality of wine is 8!')

if __name__ == "__main__":
    app.run(debug=True)
