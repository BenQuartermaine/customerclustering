import pandas as pd
import numpy as np
from db_connection import Db
import datetime as dt


#define a function to get the nth/2nd most frequent activityType
def fav_activityType(series,n=2):
    ls=series.value_counts().index.tolist()[:n-1]
    return ls

class Learning:
    def __init__(self, conn, df_act1):

        self.conn=conn
        self.df_act = df_act1
        self.df_usr=pd.read_sql_query("SELECT * FROM user;", conn).drop_duplicates()

    def get_activity_data(self):
        """
        Returns a data frame with 'userID'
        """
        # clean data here
        # create a data frame with 'userID', 'activityType', 'providerName',
        df_act=self.df_act
        df_act['providerName'].unique()
        #df_act['status'].unique()

        # drop duplicated events
        df_act=df_act.drop_duplicates(subset=['activityID','owner','resource','min'],keep='last')

        # create minPerYear
        ## step1: get total min
        selected_columns=['owner','providerName','min','updateDate','completeDate','activityType']
        df_act=df_act[selected_columns]
        #replace missing values of 'updateDate'
        df_act['completeDate']=df_act['completeDate'].fillna(df_act['updateDate'])

        # replace 'N/A' by None
        df_act['providerName'].replace('N/A',None,inplace=True)
        # see if can get missing values from tracking event!!!!!!!!!!!!!!!!!!!! DO THIS!!!!!!!!!!!!!!!!!!!!!!!!!!






        # fill in missing value with 'Unknown'
        df_act['providerName']=df_act['providerName'].fillna('Unknown')



        # rename 'owner' as 'userID'
        df_act.rename(columns={'owner':'userID'},inplace=True)
        return df_act


    def get_Ausmed_year(self):
        """
        Returns a data frame including 'userID','yearsOnAusmed'
        """
        # get yearsOnAusmed
        df_act=self.get_activity_data()
        df_usr=self.df_usr[['userID','createDate']]
        df_usr=df_usr.merge(df_act, on='userID', how='inner')
        df_usr['yearsOnAusmed']=df_usr['completeDate']-df_usr['createDate']
        df_usr=df_usr.groupby('userID').max().reset_index()
        return df_usr




    def get_activity_features(self):
        """
        Returns a data frame with 'userID','favoriteActivityType',
        '2ndfavoriteActivityType','minPerYear','percentageLearningFromAusmed'
        """
        df_act=self.get_activity_data()
        # create a column "isAusmed": returns 1 if the provider is Ausmend
        df_act['isAusmed']=df_act['providerName'].apply(lambda x: 1 if ('Ausmed' in x) else 0 )
        df_act['minOnAusmed']=df_act['min']*df_act['isAusmed']


        # Get user's favorite and 2nd favorite activityType
        # first create a copy of acticityType to get 2nd favorite activityType
        df_act['2ndFavActivityType']=df_act['activityType']
        df_act=df_act.groupby('userID').agg({'min': sum, 'minOnAusmed': sum, 'activityType': pd.Series.mode, '2ndFavActivityType': fav_activityType}).reset_index()


        # merge with df_usr
        df_usr=self.get_Ausmed_year()
        df_act=df_usr[['userID','yearsOnAusmed']].merge(df_act, on='userID', how='inner')


        # rename the columns
        df_act.rename(columns={'activityType': 'favActivityType'},inplace=True)





        # get Min learnt from Ausmed
        df_act['yearsOnAusmed']=df_act['yearsOnAusmed']/ np.timedelta64(1, 'Y')
        df_act['minOnAusmedPerYear']=df_act['minOnAusmed']/df_act['yearsOnAusmed']

        # get percentageOfLearningFromAusmed
        df_act['minPerYear']=df_act['min']/df_act['yearsOnAusmed']
        df_act['percentageOfLearningFromAusmed']=df_act['minOnAusmedPerYear']/df_act['minPerYear']

        # drop uneeded columns
        df_act=df_act.drop(columns=['yearsOnAusmed','min','minOnAusmedPerYear'])

        # drop na
        df_act=df_act.dropna()
        return df_act

if __name__ == '__main__':
    conn = Db.db_conn()
    df_act1=pd.read_sql_query("SELECT * FROM activity_20220808 LIMIT 200;", conn).drop_duplicates()
    learning=Learning(conn,df_act1)
    df=learning.get_activity_features()
    print(df.head())
