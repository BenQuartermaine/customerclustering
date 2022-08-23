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
    def __init__(self, df, MODEL_NAME, run_pca=True):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipe = None
        self.pca = None
        self.df = clean_data(df)
        # a dataframe to save principal component
        self.W=None
        # a dataframe to save projected data on principal component, or the processed data if pca is not in use
        # cluster lables will be added to this table
        self.df_proj=None
        # for job
        self.model_name = MODEL_NAME
        self.run_pca=run_pca

        # save numerical and categorical column names
        self.num_col=self.df.describe().columns
        self.cat_ord=['located','Status', 'access', 'plan_type']
        #select columns for MinMax
        self.num_minmax=['activated','ratioOfAchivedGoals','learnFromAusmedRatio_num',
                         'RatioOfCompletion_min','hasPracticeRecord',
                         'RatioOfCompletion_num','RatioOfCompletion_min',]


        #select columns for Robustscaler
        self.num_robust=[col for col in self.df.describe().columns if col not in self.num_minmax]
        #print(num_robust)

    def preprocessing(self):



        # categorical columns!
        cat_col=[col for col in self.df.columns if col not in self.df.describe().columns]


        # select the ones with only a few unique values
        cat_ord=self.cat_ord




        # order columns
        feature_1_sorted_values = ['remote area','other rural area','small rural centre',
                                'large rural centre','other metropolitan centre','metropolitan centre','capital city'] # 'located', need to check this
        feature_2_sorted_values = ['canceled','incomplete_expired','past_due', 'trialing','active',]
        feature_3_sorted_values = ['never','sometimes' ,'usually','always']
        feature_4_sorted_values = ['monthly','quarterly',  'annually']


        # create categories iteratively: the shape of categories has to be (n_feature,)
        categories=[]

        categories_base=[

                feature_1_sorted_values,
                feature_2_sorted_values,
                feature_3_sorted_values,
                feature_4_sorted_values
            ]

        categories=categories_base


        #print(categories)

        ord_enc = OrdinalEncoder(
            categories=categories,
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            encoded_missing_value=-1
        )



        # Robustscaler all numerical columns
        num_transformer0=make_pipeline(SimpleImputer(strategy='median'),RobustScaler())
        num_transformer1=make_pipeline(SimpleImputer(strategy='median'),MinMaxScaler())

        #LabelEncoder all categorical columns
        cat_transformer=make_pipeline(ord_enc,RobustScaler())




        preproc1=make_column_transformer((num_transformer0,self.num_robust),
                                         (num_transformer1,self.num_minmax),
                                         (cat_transformer,cat_ord),remainder='drop')
        return preproc1


    def preprocessing0(self):
        #original preprocessing steps, MinMax not involved
       # select columns
        # use all numerical columns
        num_col=self.num_col
        #do not include meta_title
        #cat_col=[col for col in self.df.columns if (col not in num_col)&(col!='metaGoalTitle')]

        # Robustscaler all numerical columns
        num_transformer=make_pipeline(SimpleImputer(strategy='median'),RobustScaler())

        ## preprocess categorical data
        ## OrdinalEncoder



        # select the ones with only a few unique values
        cat_ord=self.cat_ord


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


        #print(categories)

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

        if self.run_pca==True:
            print('PCA is included')
            # run PCA and save the principal component
            df_processed=self.preprocessing().fit_transform(self.df)
            self.pca=PCA()
            self.pca.fit(df_processed)
            print('pca fitted')
            # Access our PCs
            W = self.pca.components_

            # Print PCs as COLUMNS
            self.W = pd.DataFrame(W.T,
                 index=self.num_robust+self.num_minmax+self.cat_ord,
                 columns=[f'PC{i}' for i in range(1, len(self.num_robust)+len(self.num_minmax)+len(self.cat_ord)+1)])

            # self.W = pd.DataFrame(W.T,
            #                 index=self.num_col.to_list()+self.cat_ord,
            #                 columns=[f'PC{i}' for i in range(1, len(self.num_col)+len(self.cat_ord)+1)])


            # Let the data project on PCs
            df_proj=self.pca.transform(df_processed)
            #self.df_proj=pd.DataFrame(df_proj,columns=[f'PC{i}' for i in range(1, len(self.num_col)+len(self.cat_ord)+1)])
            self.df_proj=pd.DataFrame(df_proj,columns=[f'PC{i}' for i in range(1, len(self.num_robust)+len(self.num_minmax)+len(self.cat_ord)+1)])

            self.pipe=self.set_pipeline(n_cluster=n_cluster)
            # add label
            self.df_proj['label']=self.pipe.fit(self.df).predict(self.df)
        else:
            print('PCA is not included')
            self.df_proj=self.df.drop(columns=['RatioOfCompletion_num','RatioOfCompletion_min','num_subs'])
            self.pipe=make_pipeline(self.preprocessing(),MiniBatchKMeans(n_clusters=n_cluster))
            self.df_proj['label']=self.pipe.fit(self.df).predict(self.df)
            print('MiniKmeans fitted')




    def save_model(self):
        """Save the model into a .joblib format"""
        if self.run_pca==True:
            joblib.dump(self.pca, f'../models/{self.model_name}_withpca_pca.joblib')
            joblib.dump(self.pipe, f'../models/{self.model_name}_withpca.joblib')
            print(colored(f"pca+kmeans model,{self.model_name}_withpca_pca.joblib\
                          and {self.model_name}_withpca.joblib saved locally", "green"))
        else:
            joblib.dump(self.pipe, f'../models/{self.model_name}_withoutpca.joblib')
            print(colored(f"Just kmeans,{self.model_name}_nopca.joblib\
                          saved locally", "green"))


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
    train=Trainer(df,'model_v0',run_pca=True)
    train.run(n_cluster=4)
    train1=Trainer(df,'model_v0',run_pca=False).run(n_cluster=8)
    train.save_model()
