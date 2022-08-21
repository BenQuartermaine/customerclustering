import joblib
from termcolor import colored
import mlflow
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

    #


    return df_cleaned


class Trainer(object):
    def __init__(self, df):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.pca = None
        self.df = clean_data(df)
        self.pc=None
        self.df_proj=None
        # # for MLFlow
        # self.experiment_name = EXPERIMENT_NAME



    def preprocessing(self):
       # select columns
        # use all numerical columns
        num_col=df.describe().columns
        #do not include meta_title
        cat_col=[col for col in df.columns if (col not in num_col)&(col!='metaGoalTitle')]

        # Robustscaler all numerical columns
        num_transformer=make_pipeline(SimpleImputer(strategy='median'),RobustScaler())

        # Or all categorical columns
        cat_transformer_ord=make_pipeline(SimpleImputer(strategy='most_frequent'),OneHotEncoder())


        preproc=make_column_transformer((num_transformer,num_col),(cat_transformer,cat_col),remainder='drop')
        return preproc




    def set_pipeline(self,n_cluster):
        """defines the pipeline as a class attribute
        Apply SimpleImputer(median), RobustScaler to all numerical features
        SimpleImputer(),OneHotEncoder to all categporical features
        """
        preproc=self.preprocessing()
        pipe=make_pipeline(preproc,MiniBatchKMeans(n_clusters=n_cluster))
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
        df_num_proj = self.pca.transform(df_processed)
        self.df_proj = pd.DataFrame(df_num_proj, columns=[f'PC{i}' for i in range(1, len(self.df.columns)+1)])



        self.pipe=self.set_pipeline(n_cluster=n_cluster)
        self.pipe.fit(self.df)
        joblib.dump(self.pca, 'model_v0_pca.joblib')
        joblib.dump(self.pipeline, 'model_v0.joblib')
        print(colored("model_v0.joblib and model_v0_pca.joblib saved locally", "green"))


    def save_model(self):
        """Save the model into a .joblib format"""
        joblib.dump(self.pca, 'model_v0_pca.joblib')
        joblib.dump(self.pipeline, 'model_v0.joblib')
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
    # Get and clean data
    N = 100
    df = get_data_from_gcp(nrows=N)
    df = clean_data(df)
    y = df["fare_amount"]
    X = df.drop("fare_amount", axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    print(X_test.head())
    # Train and save model, locally and
    trainer = Trainer(X=X_train, y=y_train)
    trainer.set_experiment_name('xp2')
    trainer.run()
    rmse = trainer.evaluate(X_test, y_test)
    print(f"rmse: {rmse}")
    trainer.save_model_locally()
    storage_upload()


    #################
    params={
        "pickup_datetime": "2013-07-06 17:18:00",
        "pickup_longitude": "-73.950655",
        "pickup_latitude": "40.783282",
        "dropoff_longitude": "-73.984365",
        "dropoff_latitude": "40.769802",
        "passenger_count": "1"
        }
    key='2013-07-06 17:18:00.000000119'

    pickup_datetime = datetime.strptime("2013-07-06 17:18:00", "%Y-%m-%d %H:%M:%S")


    # localize the user datetime with NYC timezone
    eastern = pytz.timezone("US/Eastern")
    localized_pickup_datetime = eastern.localize(pickup_datetime, is_dst=None)

    # localize the datetime to UTC
    utc_pickup_datetime = localized_pickup_datetime.astimezone(pytz.utc)

    # convert convert a datetime to the format expected by the pipeline
    formatted_pickup_datetime = utc_pickup_datetime.strftime("%Y-%m-%d %H:%M:%S UTC")

    X_new=pd.DataFrame.from_dict({
        'key':[key],
        'pickup_datetime': [utc_pickup_datetime],
        'pickup_longitude': [-73.950655],
        'pickup_latitude': [40.783282],
        'dropoff_longitude': [-73.984365],
        'dropoff_latitude': [40.769802],
        'passenger_count': [1]
        })
    model=joblib.load('model.joblib')
    y_pred=model.predict(X_new)
    print(y_pred)
    print(f'the predicted fare is {y_pred[0]}')
