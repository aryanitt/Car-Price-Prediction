from flask import Flask, request, render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        data = CustomData(
            carlength = float(request.form.get('carlength', 0)),
            carwidth = float(request.form.get('carwidth', 0)),
            curbweight = float(request.form.get('curbweight', 0)),
            enginesize = float(request.form.get('enginesize', 0)),
            horsepower = float(request.form.get('horsepower', 0)),
            citympg = float(request.form.get('citympg', 0)),
            highwaympg = float(request.form.get('highwaympg', 0)),
            CarName = request.form.get('CarName'),
            drivewheel = request.form.get('drivewheel'),
            enginelocation = request.form.get('enginelocation'),
            fuelsystem = request.form.get('fuelsystem')
        )

        pred_df = data.get_data_as_dataframe()
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)

        return render_template('home.html', results=round(results[0], 2))


if __name__ == "__main__":
    app.run(debug=True)
