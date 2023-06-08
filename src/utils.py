import os
import sys
import numpy as np
import pandas as pd
import dill
from src.exception import CustomException
from sklearn.metrics import roc_curve
from sklearn.metrics import f1_score

def save_object(file_path,obj):
    try:
        dir_name=os.path.dirname(file_path)
        os.makedirs(dir_name,exist_ok=True)
        with open(file_path,'wb') as file_obj:
            dill.dump(obj,file_obj)
    except Exception as e:
        raise CustomException(e,sys)

def Threshold(model1,model2,model3,test_input,test_output):
    try:
        test_input=test_input.astype(float)
        test_input.columns=test_input.columns.astype(str)
        Y_proba_model1=model1.predict_proba(test_input)
        Y_proba_model2=model2.predict_proba(test_input)
        Y_proba_model3=model3.predict_proba(test_input)
        Y_proba_model1=pd.DataFrame(Y_proba_model1[:,1])
        Y_proba_model2=pd.DataFrame(Y_proba_model2[:,1])
        Y_proba_model3=pd.DataFrame(Y_proba_model3[:,1])
        final=pd.DataFrame(pd.concat([Y_proba_model1,Y_proba_model2,Y_proba_model3],axis=1).mean(axis=1))
        fpr,tpr,thresholds=roc_curve(test_output,final)
        f1_scores=[]
        for i in thresholds:
            y_pred=np.where(final>i,1,0)
            f1_scores.append(f1_score(test_output,y_pred,average='macro'))
        y_pred_final=np.where(final>thresholds[f1_scores.index(max(f1_scores))],1,0)
        return (y_pred_final,thresholds[f1_scores.index(max(f1_scores))])
    except Exception as e:
        raise CustomException(e,sys)
def load_object(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e,sys)
       
        
       
        
    