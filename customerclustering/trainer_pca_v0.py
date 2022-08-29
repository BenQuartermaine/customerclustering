from inspect import trace
from xml.etree.ElementTree import PI
import joblib
from termcolor import colored
from sklearn.preprocessing import RobustScaler,MinMaxScaler, OrdinalEncoder, OneHotEncoder
from sklearn.cluster import MiniBatchKMeans, DBSCAN
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.compose import ColumnTransformer,make_column_transformer
from sklearn import set_config; set_config(display='diagram')
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from customerclustering.get_training_data import *
from customerclustering.db_connection import Db


 # This model is PCA

#col_drop=['RatioOfCompletion_num','RatioOfCompletion_min','num_subs']


#col_drop=['numCompletedFromQueuePerYear','minCompletedFromQueuePerYear']

col_select=['account_age','docPerYear','docOnAusmedPerYear',
            'minPerYear','minOnAusmedPerYear',
            'numQueuedPerYear','minQueuedPerYear',
            'numCompletedOnelinePerYear','minCompletedOnelinePerYear',
            'event_cpd_day_diff','doc_in_activation','GoalsPerYear','activated',
            'learnFromAusmedRatio_num','hasPracticeRecord','located','access','autonomy','complex']

def intersection(lst1, lst2):
    return list(set(lst1) & set(lst2))



# functions to clean the data
def clean_data(df,col_select=col_select,threshold=1-0.007):
    # clean numerical data
    # first get rid of outliers
    # col_drop=['numCompletedFromQueuePerYear','minCompletedFromQueuePerYear',
    #       'numCompletedFromQueuePerYear','minCompletedFromQueuePerYear',]

    threshold=1-0.007
    df.quantile(1-threshold)
    #num_otl=['docPerYear','docOnAusmedPerYear','numQueued','minQueued','minCompleted','GoalsPerYear','minPerYear']

    thr_dpy=df.docPerYear.quantile(threshold)

    # set threshold
    thr_gpy=25
    thr_mpy=4000
    thr_mapy=500



    df_cleaned=df[(df.minPerYear<thr_mpy)& (df.GoalsPerYear<thr_gpy) & (df.minOnAusmedPerYear<thr_mapy)]
    df_cleaned=df[col_select]

    # clean some categorical data

    # complex
    df_cleaned.complex.replace('have a high complexity','high complexity',inplace=True)
    df_cleaned.complex.replace('be generally complex','generally complex',inplace=True)
    df_cleaned.complex.replace('have high complexity','high complexity',inplace=True)
    df_cleaned.complex.replace('be very complex','very complex',inplace=True)

    # autonomy
    df_cleaned.autonomy.replace('a moderate amount of professional autonomy','moderate',inplace=True)
    df_cleaned.autonomy.replace('a high level of professional autonomy','high',inplace=True)
    df_cleaned.autonomy.replace('a minimal professional autonomy','minimal',inplace=True)




    return df_cleaned


class Trainer(object):
    def __init__(self, df, MODEL_NAME, clustering_model='kmeans'):
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
        self.clustering_model=clustering_model



        # save numerical and categorical column names
        self.num_col=self.df.describe().columns
        self.cat_ord=intersection(['located','Status', 'access', 'plan_type','autonomy','complex'],
                                  self.df.columns)
        #select columns for MinMax
        self.num_minmax=intersection(['activated','ratioOfAchivedGoals','learnFromAusmedRatio_num',
                        'hasPracticeRecord','learnFromAusmedRatio_min'
                        ],self.df.columns)


        #select columns for Robustscaler
        self.num_robust=[col for col in self.df.describe().columns if col not in self.num_minmax]
        #print(num_robust)

        # to save the centers
        self.centres=None
        self.labels=None

    def preprocessing(self):



        # categorical columns!
        cat_col=[col for col in self.df.columns if col not in self.df.describe().columns]


        # select the ones with only a few unique values
        cat_ord=self.cat_ord




        # order columns
        feature_1_sorted_values = ['remote area','other rural area','small rural centre',
                                'large rural centre','other metropolitan centre','metropolitan centre','capital city'] # 'located', need to check this
        #feature_2_sorted_values = ['canceled','incomplete_expired','past_due', 'trialing','active',]
        feature_3_sorted_values = ['never','sometimes' ,'usually','always']
        #feature_4_sorted_values = ['monthly','quarterly',  'annually']
        feature_5_sorted_values = [ 'minimal','moderate','high']
        feature_6_sorted_values = [ 'low complexity','generally complex', 'very complex','high complexity']



        # create categories iteratively: the shape of categories has to be (n_feature,)
        categories=[]

        categories_base=[

                feature_1_sorted_values,
                #feature_2_sorted_values,
                feature_3_sorted_values,
                #feature_4_sorted_values,
                feature_5_sorted_values,
                feature_6_sorted_values
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








    def set_pipeline(self,n_cluster=6):
        """defines the pipeline as a class attribute
        Apply SimpleImputer(median), RobustScaler to all numerical features
        SimpleImputer(),OneHotEncoder to all categporical features
        """
        preproc=self.preprocessing()
        #pipe=make_Pipipeline(preproc,PCA(), MiniBatchKMeans(n_clusters=n_cluster))
        if self.clustering_model=='kmeans':
            pipe=Pipeline([('preproc',preproc),('pca',PCA()),('kmeans',MiniBatchKMeans(n_clusters=n_cluster))])
        elif self.clustering_model=='DBSCAN':
            pipe=Pipeline([('preproc',preproc),('pca',PCA()),('kmeans',DBSCAN())])

        else:
            pipe=Pipeline([('preproc',preproc),('kmeans',MiniBatchKMeans(n_clusters=n_cluster))])

        return pipe






    def run(self,n_cluster):

        if self.clustering_model=='kmeans':
            print(f'Kmeans is use for clustering with {n_cluster} clusters')
            # run PCA and save the principal component
            self.pipe=self.set_pipeline(n_cluster=15)
            self.labels=self.pipe.fit_predict(self.df)
            self.pca=self.pipe['pca']
            print('the model fitted')
            # Access our PCs and save it
            W = self.pca.components_

            # Print PCs as COLUMNS
            self.W = pd.DataFrame(W.T,
                 index=self.df.columns,
                 columns=[f'PC{i}' for i in range(1, len(self.df.columns)+1)])

            # self.W = pd.DataFrame(W.T,
            #                 index=self.num_col.to_list()+self.cat_ord,
            #                 columns=[f'PC{i}' for i in range(1, len(self.num_col)+len(self.cat_ord)+1)])


            # Let the data project on PCs
            df_proj=self.pca.transform(self.pipe['preproc'].transform(self.df))
            #self.df_proj=pd.DataFrame(df_proj,columns=[f'PC{i}' for i in range(1, len(self.num_col)+len(self.cat_ord)+1)])
            self.df_proj=pd.DataFrame(df_proj,columns=[f'PC{i}' for i in range(1, len(self.df.columns)+1)])

            # add labels to the
            self.df['label']=self.labels
            self.df_proj['label']=self.labels


        else:
            print('PCA is not included')
            self.df_proj=self.df.drop(columns=['RatioOfCompletion_num','RatioOfCompletion_min','num_subs'])
            #self.pipe=make_pipeline(self.preprocessing(),MiniBatchKMeans(n_clusters=n_cluster))
            self.pipe=self.set_pipeline(n_cluster=n_cluster)

            self.df_proj['label']=self.pipe.fit(self.df).predict(self.df)
            self.centres=pd.DataFrame(self.pipe['kmeans'].cluster_centers_,)
            print('MiniKmeans fitted')




    def save_model(self):
        """Save the model into a .joblib format"""
        self.df.to_csv(f'../raw_data/{self.model_name}_labeled_data.csv')
        if self.clustering_model=='kmeans':
            #joblib.dump(self.pca, f'../models/{self.model_name}_withpca_pca.joblib')
            self.df_proj.to_csv(f'../raw_data/{self.model_name}_leabel_processed_data.csv')
            joblib.dump(self.pipe, f'../models/{self.model_name}_withpca.joblib')
            print(colored(f"pca+kmeans model,{self.model_name}_withpca.joblib and data with labels saved locally", "green"))
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
    #train.save_model()
