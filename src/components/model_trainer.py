import sys
import os
from src.exception import CustomException
from src.logger import logging
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from dataclasses import dataclass
from src.utils import save_object,Threshold
from sklearn.metrics import roc_curve
from sklearn.metrics import f1_score
@dataclass
class ModelTrainingConfig:
    LR_model_path=os.path.join("artifacts","LogisticRegression.pkl")
    RF_model_path=os.path.join("artifacts","RandomForest.pkl")
    xgb_model_path=os.path.join("artifacts","XgBoost.pkl")

class model_training:
    def __init__(self):
        self.model_training_config=ModelTrainingConfig()
        self.lr=LogisticRegression(C=50)
        self.rf=RandomForestClassifier(criterion='entropy',max_depth= 26,max_features= 0.7909048016705856,max_samples= 0.6,min_samples_leaf= 0.021346409070201855,min_samples_split= 0.020701228043779718,n_estimators= 482,n_jobs=-1)
        self.xgb=XGBClassifier(learning_rate=0.1,n_estimators=100,colsample_bytree=0.7,verbosity=1,reg_lambda=15,max_depth=5,n_jobs=-1)
    def Initiate_model_training(self,train_data_input,train_data_output,test_data_input,test_data_output):
        try:
            logging.info("Model training is started")
            self.lr.fit(train_data_input,train_data_output)
            save_object(self.model_training_config.LR_model_path,self.lr)
            logging.info("Logistic Regression is trained and saved successfully")
            self.rf.fit(train_data_input,train_data_output)
            save_object(self.model_training_config.RF_model_path,self.rf)
            logging.info("Random Forest trained and saved successfully")
            self.xgb.fit(train_data_input,train_data_output)
            save_object(self.model_training_config.xgb_model_path,self.xgb)
            logging.info("XgBoost trained and saved successfully")
            logging.info("Model testing initiated")
            final_prediction,optimal_threshold=Threshold(self.lr,self.rf,self.xgb,test_data_input,test_data_output)
            f1Score=f1_score(test_data_output,final_prediction,average='macro')
            save_object(os.path.join('artifacts','threshold.pkl'),optimal_threshold)
            logging.info("Model testing is completed")
            return (f1Score,optimal_threshold)
        except Exception as e:
            raise CustomException(e,sys)
        


        

        