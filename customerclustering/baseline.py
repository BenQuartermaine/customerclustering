# This returns the baseline model and the pre-processed data

import pandas as pd
from sklearn.preprocessing import RobustScaler, LabelEncoder, OneHotEncoder
from sklearn.cluster import MiniBatchKMeans
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.compose import ColumnTransformer,make_column_transformer
from sklearn import set_config; set_config(display='diagram')
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from customerclustering.get_training_data import *
from customerclustering.db_connection import Db



import joblib
from termcolor import colored
#import mlflow
from memoized_property import memoized_property
#from mlflow.tracking import MlflowClient


#MLFLOW_URI = "https://mlflow.lewagon.ai/"
#EXPERIMENT_NAME = "first_experiment"

class Baseline(object):
    def __init__(self, df0, n_cluster):
        """
        df0 is the training data
        """
        self.pipeline = None
        # reset_index
        self.df = df0
        self.n_cluster=n_cluster


        # drop 'pProfileID' and 'stripeCustID' and Date columns
        self.df.drop(columns=['pProfileID','stripeCustID','startDate', 'endDate', 'createDate'],inplace=True)



        # for MLFlow
        #self.experiment_name = EXPERIMENT_NAME

    # def set_experiment_name(self, experiment_name):
    #     '''defines the experiment name for MLFlow'''
    #     self.experiment_name = experiment_name

    def preprocessing(self):
       # select columns
        # use all numerical columns
        num_col=df.describe().columns
        #do not include meta_title
        cat_col=[col for col in df.columns if (col not in num_col)&(col!='metaGoalTitle')]

        # Robustscaler all numerical columns
        num_transformer=make_pipeline(SimpleImputer(strategy='median'),RobustScaler())

        #LabelEncoder all categorical columns
        cat_transformer=make_pipeline(SimpleImputer(strategy='most_frequent'),OneHotEncoder())


        preproc=make_column_transformer((num_transformer,num_col),(cat_transformer,cat_col),remainder='drop')
        return preproc


    def set_pipeline(self):
        """defines the pipeline as a class attribute
        Apply SimpleImputer(median), RobustScaler to all numerical features
        SimpleImputer(),OneHotEncoder to all categporical features
        """
        preproc=self.preprocessing()
        pipe=make_pipeline(preproc,MiniBatchKMeans(n_clusters=self.n_cluster))
        return pipe




        self.pipeline
        return self.pipeline

    def run(self):
        self.pipe=self.set_pipeline()
        self.pipe.fit(self.df)
        joblib.dump(self.pipeline, 'model.joblib')
        print(colored("model.joblib saved locally", "green"))

    def save_model(self):
        """Save the model into a .joblib format"""
        joblib.dump(self.pipeline, 'model.joblib')
        print(colored("model.joblib saved locally", "green"))




    # # MLFlow methods
    # @memoized_property
    # def mlflow_client(self):
    #     mlflow.set_tracking_uri(MLFLOW_URI)
    #     return MlflowClient()

    # @memoized_property
    # def mlflow_experiment_id(self):
    #     try:
    #         return self.mlflow_client.create_experiment(self.experiment_name)
    #     except BaseException:
    #         return self.mlflow_client.get_experiment_by_name(
    #             self.experiment_name).experiment_id

    # @memoized_property
    # def mlflow_run(self):
    #     return self.mlflow_client.create_run(self.mlflow_experiment_id)

    # def mlflow_log_param(self, key, value):
    #     self.mlflow_client.log_param(self.mlflow_run.info.run_id, key, value)

    # def mlflow_log_metric(self, key, value):
    #     self.mlflow_client.log_metric(self.mlflow_run.info.run_id, key, value)


if __name__ == "__main__":
    df=GetTrainingData(conn=Db.db_conn(),rows=200000).get_training_data()
    baseline=Baseline(df,n_cluster=10).run()
