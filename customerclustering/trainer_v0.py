from inspect import trace
import joblib
from termcolor import colored
from sklearn.preprocessing import RobustScaler, OrdinalEncoder, OneHotEncoder
from sklearn.cluster import MiniBatchKMeans
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.compose import ColumnTransformer,make_column_transformer
from sklearn import set_config; set_config(display='diagram')
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from customerclustering.get_training_data import *
from customerclustering.db_connection import Db


 # This model is PCA







# functions to clean the data
def clean_data(df,threshold=1-0.007):
    # clean numerical data
    # first get rid of outliers
    df.quantile(1-threshold)
    #num_otl=['docPerYear','docOnAusmedPerYear','numQueued','minQueued','minCompleted','GoalsPerYear','minPerYear']

    thr_dpy=df.docPerYear.quantile(threshold)
    thr_dapy=df.docOnAusmedPerYear.quantile(threshold)
    thr_nq=df.numQueued.quantile(threshold)
    thr_mq=df.minQueued.quantile(threshold)
    thr_mc=df.minCompleted.quantile(threshold)
    thr_gpy=df.GoalsPerYear.quantile(threshold)
    thr_mpy=df.minPerYear.quantile(threshold)
    thr_mapy=df.minOnAusmedPerYear.quantile(threshold)

    df_cleaned=df[(df.docPerYear<thr_dpy) & (df.docOnAusmedPerYear<thr_dapy) &
                (df.numQueued<thr_nq)& (df.minQueued<thr_mq)& (df.minCompleted<thr_nq)&
                (df.GoalsPerYear<thr_gpy) & (df.minPerYear<thr_mpy) & (df.minOnAusmedPerYear<thr_mapy)]

    # clean some categorical data

    # complex

    # au


    return df_cleaned


class Trainer(object):
    def __init__(self, df, MODEL_NAME):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipe = None
        self.pca = None
        self.df = clean_data(df)
        # a dataframe to save principal component
        self.pc=None
        # a dataframe to save projected data on principal component
        self.df_proj=None
        # for job
        self.model_name = MODEL_NAME



    def preprocessing(self):
       # select columns
        # use all numerical columns
        num_col=self.df.describe().columns
        #do not include meta_title
        cat_col=[col for col in self.df.columns if (col not in num_col)&(col!='metaGoalTitle')]

        # Robustscaler all numerical columns
        num_transformer=make_pipeline(SimpleImputer(strategy='median'),RobustScaler())

        ## preprocess categorical data
        ## OrdinalEncoder
        cat_col=[col for col in self.df.columns if col not in num_col]


        # select the ones with only a few unique values
        cat_ord=['located','Status', 'access', 'plan_type']


        # prepare OrdinalEncoder

        # order columns
        feature_1_sorted_values = ['remote area','other rural area','small rural centre',
                                'large rural centre','other metropolitan centre','metropolitan centre','capital city'] # 'located', need to check this
        feature_2_sorted_values = ['canceled','incomplete_expired','past_due', 'trialing','active',]
        feature_3_sorted_values = ['never','sometimes' ,'usually','always']
        feature_4_sorted_values = ['monthly','quarterly',  'annually']


        # create categories iteratively: the shape of categories has to be (n_feature,)


        categories_base=[

                feature_1_sorted_values,
                feature_2_sorted_values,
                feature_3_sorted_values,
                feature_4_sorted_values
            ]

        categories=categories_base #luckily, we don't need to


        print(categories)

        ord_enc = OrdinalEncoder(
            categories=categories,
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            encoded_missing_value=-1
        )

        cat_transformer=make_pipeline(ord_enc, RobustScaler())
        preproc=make_column_transformer((num_transformer,num_col),(cat_transformer,cat_ord),remainder='drop')
        return preproc




    def set_pipeline(self,n_cluster):
        """defines the pipeline as a class attribute
        Apply SimpleImputer(median), RobustScaler to all numerical features
        SimpleImputer(),OneHotEncoder to all categporical features
        """
        preproc=self.preprocessing()
        pipe=make_pipeline(preproc,PCA(), MiniBatchKMeans(n_clusters=n_cluster))
        return pipe






    def run(self,n_cluster):
        df_processed=self.preprocessing(self.df)
        self.pca=PCA()
        self.pca.fit(df_processed)
        # Access our PCs
        W = self.pca.components_

        # Print PCs as COLUMNS
        self.pc = pd.DataFrame(W.T,
                        index=self.df.columns,
                        columns=[f'PC{i}' for i in range(1, len(self.df.columns)+1)])
        # Let data project on the PCs
        df_proj=self.pca.transform(df_processed)
        self.df_proj=pd.DataFrame(df_proj,columns=[f'PC{i}' for i in range(1, len(num_col)+len(cat_ord)+1)])



        self.pipe=self.set_pipeline(n_cluster=n_cluster)
        self.pipe.fit(self.df)
        joblib.dump(self.pca, f'../models/{self.model_name}_pca.joblib')
        joblib.dump(self.pipe, f'../models/{self.model_name}.joblib')
        print(colored("model_v0: pca+kmeans,model_v0.joblib and model_v0_pca.joblib saved locally", "green"))


    def save_model(self):
        """Save the model into a .joblib format"""
        joblib.dump(self.pca, 'model_v0_pca.joblib')
        joblib.dump(self.pipe, f'../models/{self.model_name}.joblib')
        print(colored("model_v0.joblib and model_v0_pca.joblib saved locally", "green"))

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
    baseline=Trainer(df,'model_v0').run(n_cluster=3)
