from flask import Flask,request,render_template
import numpy as np
import pandas as pd
from src.pipeline.predict_pipeline import CustomData,predict_pipeline
application = Flask(__name__)
app=application

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictData',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data=CustomData(request.form.get('potential_issue'),
                   request.form.get('deck_risk'),
                   request.form.get('oe_constraint'),
                   request.form.get('ppap_risk'),
                   request.form.get('stop_auto_buy'),
                   request.form.get('rev_stop'),
                   request.form.get('national_inv'),
                   request.form.get('lead_time'),
                   request.form.get('in_transit_qty'),
                   request.form.get('forecast_3_month'),
                   request.form.get('sales_1_month'),
                   request.form.get('pieces_past_due'),
                   request.form.get('perf_12_month_avg'),
                   request.form.get('local_bo_qty'))
        df=data.get_data_as_dataframe()
        obj=predict_pipeline()
        prediction=obj.predict(df)
        return  render_template('home.html',results=prediction)
        
        
    
if __name__ == '__main__':
    app.run(host="0.0.0.0",port=8080)