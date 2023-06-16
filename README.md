<h1>End to end implementation of Backorder Prediction machine learning project</h1><hr>
<p>requirements.txt---Contains all the packages that are needed to be installed during the project development.</p><hr>
<p>setup.py---Can build the whole application as a package itself and maintains all the packages.</p><hr>
<p>logger.py---Logs all the information in a log file so that the developers can always be aware of the workflow of their own project."from logger.py import logging".</p><hr>
<p>exception.py---Contains the CustomException class which can be imported in any file and raise CustomException. This enables a developer to review the errors during project development."from exception.py import CustomException"</p><hr>
<p>utils.py---Contains all the handy user-defined functions which can be useful in many files of the project.</p><hr>
<p>app.py---Contains the code for the web app(Flask)</p><hr>
<p>Templates folder---Contains the html files.</p><hr>
<p><p>src folder---consists of several sub folders like 'components', 'pipeline'</p><br>
  <p>components folder---Consists of 'data ingestion.py' in which the train and test datasets are loaded, consists of 'data_transformation.py' which contains the whole code for feature engineering, consists of 'model_trainer.py' which contains the code for training and testing the machine learning models such as Logistic regression,RandomForest and Xgboost.</p><br>
  <p>pipeline folder---Consists of 'predict_pipeline.py' which contains the code that takes the data given by the web app user and predicts the output(went_on_backorder:Yes or No).</p>
</p><hr>
<p>artifacts folder--- contains all the pickle files loaded during the data ingestion, data transformation and model training process which are useful for prediction pipeline</p><hr>
<h1>The End</h1>

