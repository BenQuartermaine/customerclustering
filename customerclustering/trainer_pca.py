from inspect import trace
import re
from xml.etree.ElementTree import PI
import joblib
from termcolor import colored
from sklearn.preprocessing import RobustScaler,MinMaxScaler, OrdinalEncoder, OneHotEncoder, FunctionTransformer
from sklearn.cluster import MiniBatchKMeans, DBSCAN
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.compose import ColumnTransformer,make_column_transformer
from sklearn import set_config; set_config(display='diagram')
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from customerclustering.get_training_data import *
from customerclustering.db_connection import Db
from scipy import stats
from customerclustering.encoder import get_ord_encoder


 # This model is PCA

#col_drop=['RatioOfCompletion_num','RatioOfCompletion_min','num_subs']


col_drop=['numCompletedFromQueuePerYear','minCompletedFromQueuePerYear']

col_select=['account_age',
            'docPerYear',
            'docOnAusmedPerYear',
            'minPerYear',
            'minOnAusmedPerYear',
            'numQueuedPerYear',
            'minQueuedPerYear',
            'numCompletedOnelinePerYear',
            'minCompletedOnelinePerYear',
            'event_cpd_day_diff',
            'doc_in_activation',
            'GoalsPerYear','activated',
            'learnFromAusmedRatio_num',
            'hasPracticeRecord',
            'located','access','autonomy','complex']

cols_w_outliers = ['num_subs',
                   'docPerYear',
                   'docOnAusmedPerYear',
                   'numQueuedPerYear',
                   'minQueuedPerYear',
                   'GoalsPerYear',
                   'minPerYear',
                   'minOnAusmedPerYear',
                   'minCompletedOnelinePerYear',
                   'doc_in_activation',
                   'subscribe_days']


def remove_outliers(df, pctile_remove = 0.7, columns_list = cols_w_outliers):

    df_cleaned = df.copy()

    threshold = 1 - (pctile_remove / 100)

    for col in columns_list:
        df_cleaned = df_cleaned[(df_cleaned[col] < df_cleaned[col].quantile(threshold))]

    print(f'the percentage of outliers disgarded is {1-len(df_cleaned)/len(df)}')

    return df_cleaned


def intersection(lst1, lst2):
    return list(set(lst1) & set(lst2))

def log_transform(x):
    return np.log(x + 1)

# functions to clean the data
def clean_data(df,pctile_remove=0.8,col_drop=col_drop):
    # clean numerical data
    # first get rid of outliers
    # col_drop=['numCompletedFromQueuePerYear','minCompletedFromQueuePerYear',
    #       'numCompletedFromQueuePerYear','minCompletedFromQueuePerYear',]
    df_cleaned=remove_outliers(df,pctile_remove = pctile_remove, columns_list = cols_w_outliers)

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

    # drop some columns
    df_cleaned.drop(columns=col_drop,inplace=True)


    return df_cleaned


class Trainer(object):
    def __init__(self, df, MODEL_NAME, clustering_model='kmeans'):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipe = None
        self.pca = None
        self.df = clean_data(df,pctile_remove=0.8)

        # a dataframe for the preprocessed data
        self.df_processed=None

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
        self.minmax_cols = ['num_subs',
                        'activated',
                        'hasPracticeRecord',
                        'learnFromAusmedRatio_num',
                        'learnFromAusmedRatio_min',
                        'ratioOfAchivedGoals']
        # self.num_minmax=intersection(['activated','ratioOfAchivedGoals','learnFromAusmedRatio_num',
        #                 'hasPracticeRecord','learnFromAusmedRatio_min'
        #                 ],self.df.columns)

        # log because they are skewed
        self.log_cols = ['minPerYear',
                        'minOnAusmedPerYear',
                        'numQueuedPerYear',
                        'numCompletedOnelinePerYear',
                        'minCompletedOnelinePerYear',
                        'minQueuedPerYear',
                        'doc_in_activation']
        #select columns for Robustscaler
        self.rscaler_cols=[col for col in self.df.describe().columns if col not in self.minmax_cols+self.log_cols]
        #print(num_robust)

        # to save the centers
        self.kmeans=None
        self.labels=None
        self.BDSCAN=None

    def preprocessor(self):
        # categorical columns!
        cat_col=[col for col in self.df.columns if col not in self.df.describe().columns]


        # select the ones with only a few unique values
        cat_ord=self.cat_ord

        # standard scaler
        rscaler_transformer = make_pipeline(SimpleImputer(strategy = 'median'),
                                        RobustScaler())

        # minmax scaler
        minmax_transformer = make_pipeline(SimpleImputer(strategy = 'most_frequent'),
                                        MinMaxScaler())

        # log transform & scaler
        log_transformer = make_pipeline(SimpleImputer(strategy = 'median'),
                                        FunctionTransformer(log_transform),
                                        RobustScaler())

        # use the above ordinal encoder & minmax. all exist so no need for simple imputer?
        ord_transformer = make_pipeline(get_ord_encoder(),
                                        MinMaxScaler())

        # combine all pipelines together, drop the unassigned columns
        preproc = make_column_transformer((rscaler_transformer, self.rscaler_cols),
                                        (minmax_transformer, self.minmax_cols),
                                        (log_transformer, self.log_cols),
                                        (ord_transformer, self.cat_ord),
                                        remainder='drop')

        # print out all columns to use


        return preproc










    def get_processed_data(self):
        for feature_list, feature_type in zip((self.rscaler_cols,
                                               self.minmax_cols,
                                               self.log_cols,
                                               self.cat_ord), ('robust', 'minmax', 'log', 'ordinal')):
            print(f' \n--- {feature_type} ---')
            for feat in feature_list:
                print(feat)
        preproc=self.preprocessor()
        # add all columns to one list to make it easier
        all_columns = self.rscaler_cols + self.minmax_cols + self.log_cols + self.cat_ord # + ['missing_indicator']
        # process the dataframe and save back into dataframe
        df_processed = pd.DataFrame(preproc.fit_transform(self.df),columns = all_columns)
        return df_processed



    # def set_pipeline(self,best_num_pcs=12,n_cluster=6):
    #     """defines the pipeline as a class attribute
    #     Apply SimpleImputer(median), RobustScaler to all numerical features
    #     SimpleImputer(),OneHotEncoder to all categporical features
    #     """

    #     #pipe=make_Pipipeline(preproc,PCA(), MiniBatchKMeans(n_clusters=n_cluster))
    #     if self.clustering_model=='kmeans':
    #         pipe=Pipeline([('preproc',preproc),('pca',PCA(12)),('kmeans',MiniBatchKMeans(n_clusters=n_cluster))])
    #     elif self.clustering_model=='DBSCAN':
    #         pipe=Pipeline([('preproc',preproc),('pca',PCA(12)),('kmeans',DBSCAN())])

    #     else:
    #         pipe=Pipeline([('preproc',preproc),('kmeans',MiniBatchKMeans(n_clusters=n_cluster))])

    #     return pipe

    def run(self,best_num_pcs=12,best_num_clusters=6):


        preproc=self.preprocessor()
        self.df_processed=self.get_processed_data()
        print('\n --- got preprocessed data ')
        # create list of PC names
        pc_columns = [f'PC{i + 1}' for i in range(best_num_pcs)]

        # use best number of pcs to create a final pca pipe
        best_pca_pipe = make_pipeline(preproc,
                                PCA(best_num_pcs))

        # process the dataframe and save back into dataframe
        pcs_df = pd.DataFrame(best_pca_pipe.fit_transform(self.df),
                                columns = pc_columns)

        print(' \n --- pca fitted ')

        # get fitted pca
        fitted_pca = best_pca_pipe[-1]

        # save the pca-ed data and weights before clustering
        self.pca=fitted_pca
        self.df_proj=pcs_df
        W = self.pca.components_
        # Print PCs as COLUMNS
        self.W = pd.DataFrame(W.T,
                              index=self.df_processed.columns,
                              columns=[f'PC{i}' for i in range(1, best_num_pcs+1)])

        print('\n --- got weights ')

        if self.clustering_model=='kmeans':
            print(f'\n --- Kmeans is use for clustering with {best_num_clusters} clusters')
            # use best number of clusters to create a final kmeans pipe
            self.pipe = make_pipeline(best_pca_pipe,
                                    MiniBatchKMeans(n_clusters=best_num_clusters))

            # fit pipeline
            self.pipe.fit(self.df)
            print('\n --- Kmean model fitted ')
            # get fitted kmeans
            fitted_kmeans = self.pipe[-1]
            self.kmeans=fitted_kmeans

            self.labels=self.kmeans.labels_
            # add labels to the
            self.df['cluster_id']=self.kmeans.labels_
            self.df_proj['cluster_id']=self.kmeans.labels_


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
            self.df_proj.to_csv(f'../raw_data/{self.model_name}_labeled_processed_data.csv')
            joblib.dump(self.pipe, f'../models/{self.model_name}_pca_Kmn.joblib')
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
