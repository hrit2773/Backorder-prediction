import sys
import os
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler,OrdinalEncoder
from sklearn.compose import ColumnTransformer
from src.utils import load_object
from src.exception import CustomException
class predict_pipeline:
    def __init__(self):
        self.lr_path='//home//ubuntu//artifacts//LogisticRegression.pkl'
        self.rf_path='//home//ubuntu//artifacts//RandomForest.pkl'
        self.xgb_Path='//home//ubuntu//artifacts//XgBoost.pkl'
        self.Scale_path='//home//ubuntu//artifacts//Scaler.pkl'
        self.thres_path='//home//ubuntu//artifacts//threshold.pkl'
    def predict(self,features):
        try:
            features.iloc[:,0]=np.where(features.iloc[:,0]=='Yes',1,0)
            features.iloc[:,1]=np.where(features.iloc[:,1]=='Yes',1,0)
            features.iloc[:,2]=np.where(features.iloc[:,2]=='Yes',1,0)
            features.iloc[:,3]=np.where(features.iloc[:,3]=='Yes',1,0)
            features.iloc[:,4]=np.where(features.iloc[:,4]=='Yes',1,0)
            features.iloc[:,5]=np.where(features.iloc[:,5]=='Yes',1,0)
            s=load_object(self.Scale_path)
            scaled=pd.DataFrame(s.transform(features.iloc[:,6:]),columns=['national_inv','lead_time','in_transit_qty','forecast_3_month','sales_1_month','pieces_past_due','perf_12_month_avg','local_bo_qty'])
            df_obtained=pd.concat([features.iloc[:,0:6],scaled],axis=1)
            df_obtained=df_obtained.astype(float)
            lr=load_object(self.lr_path)
            rf=load_object(self.rf_path)
            xgb=load_object(self.xgb_Path)
            df_obtained.fillna(0, inplace=True)
            Y_proba_lr=lr.predict_proba(df_obtained)
            Y_proba_rf=rf.predict_proba(df_obtained)
            Y_proba_xgb=xgb.predict_proba(df_obtained)
            thres=load_object(self.thres_path)
            final=pd.DataFrame(pd.concat([pd.DataFrame(Y_proba_lr[:,1]),pd.DataFrame(Y_proba_rf[:,1]),pd.DataFrame(Y_proba_xgb[:,1])],axis=1).mean(axis=1))
            y_pred_final=np.where(final>float(thres),"Yes","No")
            return y_pred_final[0]
        except Exception as e:
            raise CustomException(e,sys)
class CustomData:
    def __init__(self,potential_issue,deck_risk,oe_constraint,ppap_risk,stop_auto_buy,rev_stop,national_inv,lead_time,in_transit_qty,forecast_3_month,sales_1_month,pieces_past_due,perf_12_month_avg,local_bo_qty):
        self.potential_issue=potential_issue
        self.deck_risk=deck_risk
        self.oe_constraint=oe_constraint
        self.ppap_risk=ppap_risk
        self.stop_auto_buy=stop_auto_buy
        self.rev_stop=rev_stop
        self.national_inv=national_inv
        self.lead_time=lead_time
        self.in_transit_qty=in_transit_qty
        self.forecast_3_month=forecast_3_month
        self.sales_1_month=sales_1_month
        self.pieces_past_due=pieces_past_due
        self.perf_12_month_avg=perf_12_month_avg
        self.local_bo_qty=local_bo_qty
    def get_data_as_dataframe(self):
        try:
            df=pd.DataFrame({"potential_issue":[self.potential_issue],
                        "deck_risk":[self.deck_risk],
                        "oe_constraint":[self.oe_constraint],
                        "ppap_risk":[self.ppap_risk],
                        "stop_auto_buy":[self.stop_auto_buy],
                        "rev_stop":[self.rev_stop],
                        "national_inv":[self.national_inv],
                        "lead_time":[self.lead_time],
                        "in_transit_qty":[self.in_transit_qty],
                        "forecast_3_month":[self.forecast_3_month],
                        "sales_1_month":[self.sales_1_month],
                        "pieces_past_due":[self.pieces_past_due],
                        "perf_12_month_avg":[self.perf_12_month_avg],
                        "local_bo_qty":[self.local_bo_qty]})
            return df
        except Exception as e:
            raise CustomException(e,sys)
        
        
        
        
        
        
        
        
        
    
        
