# This returns the baseline model and the pre-processed data
from sklearn.preprocessing import RobustScaler, LabelEncoder, OneHotEncoder
from sklearn.cluster import MiniBatchKMeans
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.compose import ColumnTransformer,make_column_transformer
from sklearn import set_config; set_config(display='diagram')
from customerclustering.get_training_data import *
from customerclustering.db_connection import Db


# access to the data
df0=GetTrainingData(conn=Db.db_conn(),rows=200000).get_training_data()


def baseline(df=df0):
    result={}
    # select columns
    # use all numerical columns
    num_col=df.describe().columns
    # Do not use favActivityType
    cat_col=['Product', 'Status','typeOfPractice', 'located',
             'specialities', 'population', 'focus', 'complex', 'autonomy', 'access','country']


    # Robustscaler all numerical columns
    num_transformer=RobustScaler()

    #LabelEncoder all categorical columns
    cat_transformer=LabelEncoder()


    # Our first base pipeline-doesn;t work
    preproc=make_column_transformer((num_transformer,num_col),(cat_transformer,cat_col))



    # manually transform the numerical and categorical columns
    ## numerical
    df_num=pd.DataFrame(num_transformer.fit_transform(df[num_col]))
    df_num.columns=num_col
    df_num.head()

    ## categorical
    df_cat=df[cat_col].apply(cat_transformer.fit_transform)

    df_processed=df_num
    df_processed[cat_col]=df_cat
    df_processed.head()

    # reindex df by userID
    df_processed.set_index(df['userID'],inplace=True)
    df_processed
    result['preprocessed_data']=df_processed
    #base pipe
    #basepipe=make_pipeline(preproc,MiniBatchKMeans()) #doesn't work:( shame on you!

    n_cluster=14
    base_model=MiniBatchKMeans(n_clusters=14)
    X_pred=base_model.fit(df_processed).predict(df_processed)

    result['baseline_model']=base_model

    return result


conn = Db.db_conn()

df=GetTrainingData(conn=Db.db_conn(),rows=1000).get_training_data()
df.head()
result=baseline(df)
centers=pd.DataFrame(result['baseline_model'].cluster_centers_)
centers.columns=result['preprocessed_data'].columns
centers.head()
