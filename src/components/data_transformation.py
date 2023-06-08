from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
import sys
import os
import pandas as pd
import numpy as np
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')
    scaler_file_path=os.path.join('artifacts','Scaler.pkl')
    
class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
    def get_data_pipeline_object(self):
        try:
            trf1=ColumnTransformer([('mean_imputation',SimpleImputer(strategy='mean'),[0,1,2,3,4,5,6,7,8,9,10,12,13,14,15]),
                       ('mode imputation',SimpleImputer(strategy='most_frequent'),[11,16,17,18,19,20])],remainder='passthrough')
            trf2=ColumnTransformer([('oe',OrdinalEncoder(categories=[['No','Yes'],['No','Yes'],['No','Yes'],['No','Yes'],['No','Yes'],['No','Yes']]),[15,16,17,18,19,20])],remainder='passthrough')
            pipe=make_pipeline(trf1,trf2)
            
        except Exception as e:
            raise CustomException(e,sys)
        return pipe
    def initiate_data_transformation(self,train_path,test_path):
        try:
            training_data=pd.read_csv(train_path)
            testing_data=pd.read_csv(test_path)
            input_columns=['national_inv', 'lead_time', 'in_transit_qty', 'forecast_3_month',
            'forecast_6_month', 'forecast_9_month', 'sales_1_month',
            'sales_3_month', 'sales_6_month', 'sales_9_month', 'min_bank',
            'potential_issue', 'pieces_past_due', 'perf_6_month_avg',
            'perf_12_month_avg', 'local_bo_qty', 'deck_risk', 'oe_constraint',
            'ppap_risk', 'stop_auto_buy', 'rev_stop']
            target_column=['went_on_backorder']
            X_train=training_data[input_columns]
            Y_train=training_data[target_column]
            X_test=testing_data[input_columns]
            Y_test=testing_data[target_column]
            training_data=training_data[training_data['went_on_backorder'].isnull()==False]
            logging.info("Train and test data reading is completed")
            preprocessing_obj=self.get_data_pipeline_object()
            logging.info("Obtained the pipeline object")
            X_train_trans=preprocessing_obj.fit_transform(X_train)
            X_test_trans=preprocessing_obj.transform(X_test)
            save_object(self.data_transformation_config.preprocessor_obj_file_path,preprocessing_obj)
            logging.info("Preprocessor file saved in artifacts folder")
            logging.info("Encoding and missing values imputation process is done")
            X_train_trans=pd.DataFrame(X_train_trans,columns=[['potential_issue','deck_risk', 'oe_constraint','ppap_risk', 'stop_auto_buy', 'rev_stop','national_inv', 'lead_time', 'in_transit_qty', 'forecast_3_month','forecast_6_month', 'forecast_9_month', 'sales_1_month','sales_3_month', 'sales_6_month', 'sales_9_month','min_bank','pieces_past_due','perf_6_month_avg','perf_12_month_avg','local_bo_qty']])
            X_test_trans=pd.DataFrame(X_test_trans,columns=[['potential_issue','deck_risk', 'oe_constraint','ppap_risk', 'stop_auto_buy', 'rev_stop','national_inv', 'lead_time', 'in_transit_qty', 'forecast_3_month','forecast_6_month', 'forecast_9_month', 'sales_1_month','sales_3_month', 'sales_6_month', 'sales_9_month','min_bank','pieces_past_due','perf_6_month_avg','perf_12_month_avg','local_bo_qty']])
        
            logging.info("Converted the numpy arrays into Dataframes")
            X_train_trans=X_train_trans.astype(float)
            Y_train=np.where(Y_train['went_on_backorder']=='Yes',1,0)
            sm=SMOTE(sampling_strategy=1.0,random_state=42,n_jobs=-1)
            X_res,Y_res=sm.fit_resample(X_train_trans,Y_train)   
            X_res['potential_issue']=np.where(X_res['potential_issue']>=0.5,1,0)
            X_res['deck_risk']=np.where(X_res['deck_risk']>=0.5,1,0)
            X_res['oe_constraint']=np.where(X_res['oe_constraint']>=0.5,1,0)
            X_res['ppap_risk']=np.where(X_res['ppap_risk']>=0.5,1,0)
            X_res['stop_auto_buy']=np.where(X_res['stop_auto_buy']>=0.5,1,0)
            X_res['rev_stop']=np.where(X_res['rev_stop']>=0.5,1,0)
            logging.info("SMOTE oversampling is applied")
            test=pd.concat([X_test_trans,Y_test],axis=1)
            test_trans=pd.concat([test[test['went_on_backorder']=='No'].sample(2688,random_state=150),test[test['went_on_backorder']=='Yes']],axis=0)
            X_test_new=test_trans.drop(['went_on_backorder'],axis=1)
            Y_test_new=test_trans['went_on_backorder']
            logging.info("A sample of test data is taken with equal class ratio")
            X_res=X_res.astype(float)
            logging.info("All the columns are converted into float type")
            X_res=X_res.drop(['forecast_6_month','sales_3_month','forecast_9_month','sales_6_month','perf_6_month_avg','sales_9_month','min_bank'],axis=1)
            X_test_trans=X_test_trans.drop(['forecast_6_month','sales_3_month','forecast_9_month','sales_6_month','perf_6_month_avg','sales_9_month','min_bank'],axis=1)
            logging.info("Feature selection using VIF is done")
            s=StandardScaler()
            scaled=pd.DataFrame(s.fit_transform(X_res.iloc[:,6:]),columns=['national_inv','lead_time','in_transit_qty','forecast_3_month','sales_1_month','pieces_past_due','perf_12_month_avg','local_bo_qty'])
            trans=pd.concat([X_res.iloc[:,0:6],scaled],axis=1)
            test_scaled=pd.DataFrame(s.transform(X_test_trans.iloc[:,6:]),columns=['national_inv','lead_time','in_transit_qty','forecast_3_month','sales_1_month','pieces_past_due','perf_12_month_avg','local_bo_qty'])
            test_trans=pd.concat([X_test_trans.iloc[:,0:6],test_scaled],axis=1)
            save_object(self.data_transformation_config.scaler_file_path,s)
            logging.info("Standard scaling is done")
            trans.columns=['potential_issue','deck_risk','oe_constraint','ppap_risk','stop_auto_buy','rev_stop','national_inv','lead_time','in_transit_qty','forecast_3_month','sales_1_month','pieces_past_due','perf_12_month_avg','local_bo_qty']
            test=pd.concat([test_trans,Y_test],axis=1)
            test_trans_new=pd.concat([test[test['went_on_backorder']=='No'].sample(2688,random_state=150),test[test['went_on_backorder']=='Yes']],axis=0)
            X_test_new=test_trans_new.drop(['went_on_backorder'],axis=1)
            Y_test_new=test_trans_new['went_on_backorder']
            X_test_new.columns=trans.columns
            Y_test_new=np.where(Y_test_new=='Yes',1,0)
            logging.info("Data transformation is completed")
            
            return (
                trans,Y_res,X_test_new,Y_test_new,self.data_transformation_config.preprocessor_obj_file_path
            )
            
        except Exception as e:
            raise CustomException(e,sys)
        
        