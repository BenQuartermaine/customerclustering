import pandas as pd
import numpy as np
from customerclustering.db_connection import Db
import datetime as dt
import random

#define a function to get the nth/2nd most frequent activityType
def fav_activityType(series,n=2):
    ls=series.value_counts().index.tolist()
    if len(ls)>1:
        return ls[1]
    else:
        return ls[0]

# to handle multiple-mode situation
# If a user has multiple favActivityType, randomly select one
def fav_fav(series):
    mode=series.mode()
    if len(mode)==1:
        return mode
    else:
        #print(mode)
        ind=random.randint(0,len(mode)-1)
        return mode[ind]
        #return random.shuffle(mode)[0]



class Learning:
    def __init__(self, conn, df_act1):

        self.conn=conn
        self.df_act = df_act1
        #self.df_usr=pd.read_sql_query("SELECT * FROM user;", conn).drop_duplicates()

    def get_activity_data(self):
        """
        Returns a data frame with 'userID','activityType', 'providerName'
        """
        # clean data here
        # create a data frame with 'userID', 'activityType', 'providerName',
        df_act=self.df_act
        df_act['providerName'].unique()
        #df_act['status'].unique()

        #sort by 'userID' the 'min'
        df_act=df_act.sort_values(by=['owner','min'],ascending=[True,True])
        # drop duplicated events

        df_act=df_act.drop_duplicates(subset=['activityID','owner','resource','min'],keep='last')

        # create minPerYear
        ## step1: get total min
        selected_columns=['activityID','owner','providerName','min','createDate','activityType']
        df_act=df_act[selected_columns]

        # get Years of the activities
        df_act['createDate']=pd.to_datetime(df_act['createDate'])
        df_act['createDate']=pd.DatetimeIndex(df_act['createDate']).year




        df_act['providerName'].replace('N/A','Unknown',inplace=True)
        df_act['providerName'].replace('','Unknown',inplace=True)
        # see if can get missing values from tracking event!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! DO THIS!!!!!!!!!!!!!!!!!!!!!!!!!!

        # fill in missing value with 'Unknown'
        df_act['providerName']=df_act['providerName'].fillna('Unknown')




        # rename 'owner' as 'userID'
        df_act.rename(columns={'owner':'userID'},inplace=True)
        #df_act.head(20)
        return df_act

    def get_activity_per_year(self):
        """
        returns a dataframe with 'userID','createYear','numOfDoc','numOnAusmed','min','minOnAusmed'.
        this dataframe is for reference only, with yearly num and min information.
        """
        # create a column "isAusmed": returns 1 if the provider is Ausmend
        df_act1=self.get_activity_data()
        df_act1['isAusmed']=df_act1['providerName'].apply(lambda x: 1 if ('Ausmed' in x) else 0 )
        df_act1['minOnAusmed']=df_act1['min']*df_act1['isAusmed']

        df_act1.head(20)

        # create a column for the number of documetation
        df_act1['numOfDoc']=1

        df_act1=df_act1.groupby(['userID','createDate']).agg({'numOfDoc': sum,'isAusmed':sum, 'min': sum, 'minOnAusmed': sum}).reset_index()




        # rename the columns
        df_act1.rename(columns={ 'createDate': 'createYear','isAusmed':'numOnAusmed'},inplace=True)
        df_act1.sort_values('userID',inplace=True)

        return df_act1

    def get_doc_per_year(self):
        """
        returns a dataframe with 'userID','docPerYear','docOnAusmedPerYear',
        'minPerYear','minOnAusmedPerYear',
        'learnFromAusmedRatio_num','learnFromAusmedRatio_min'
        """

        # get the docPerYear and minPerYear, learnFromAusmedRatio_min, learnFromAusmedRatio_num
        df_act2=self.get_activity_per_year()
        df_act2=df_act2.groupby('userID').mean().reset_index()
        # rename the data frame
        df_act2.rename(columns={'numOfDoc':'docPerYear','numOnAusmed':'docOnAusmedPerYear','min':'minPerYear','minOnAusmed':'minOnAusmedPerYear'},inplace=True)

        # drop 'createYear'
        df_act2.drop(columns=['createYear'],inplace=True)

        # get ratios
        df_act2['learnFromAusmedRatio_num']=df_act2['docOnAusmedPerYear']/df_act2['docPerYear']
        df_act2['learnFromAusmedRatio_min']=df_act2['minOnAusmedPerYear']/df_act2['minPerYear']
        return df_act2

    def get_fav_activity(self):
        """
        returns a dataframe with 'userID','favActivityType' and 'secondFavActivityType'
        """
        df_act3=self.get_activity_data()

        # create a replica of the activityType to get the secondFavActivityType
        df_act3['secondFavActivityType']=df_act3['activityType']
        df_act3=df_act3.groupby(['userID']).agg({'activityType': fav_fav, 'secondFavActivityType': fav_activityType}).reset_index()
        df_act3

        return df_act3




    def get_activity_features(self):
        print('Getting activity features')
        """
        Returns a data frame with 'userID','favoriteActivityType',
        '2ndfavoriteActivityType','minPerYear','min','percentageLearningFromAusmed'
        """
        df_act=self.get_doc_per_year().merge(
            self.get_fav_activity(),on='userID', how='inner'
        )
        # drop na
        df_act=df_act.dropna()
        return df_act




if __name__ == '__main__':
    conn = Db.db_conn()
    df_act1=pd.read_sql_query("SELECT * FROM activity_20220808 LIMIT 200;", conn).drop_duplicates()
    learning=Learning(conn,df_act1)
    df=learning.get_activity_features()
    #print(df['favActivityType'].unique())
    print(df.describe())
