import os
from pickle import TRUE
import pymysql
import pandas as pd
from dotenv import load_dotenv, find_dotenv
import datetime as dt
import numpy as np

conn = pymysql.connect(
    host=os.getenv('HOST'),
    port=int(3306),
    user=os.getenv('USER_DB'),
    passwd=os.getenv('PASSWORD'),
    db=os.getenv('DB'),
    charset='utf8mb4')

class Queue:
    def __init__(self, df):
        # Import data only once
        self.df_trk = pd.read_sql_query("SELECT * FROM tracking_event;", conn).drop_duplicates()
        self.df_res = pd.read_sql_query("SELECT * FROM resource;", conn).drop_duplicates()

    def get_event_data(self):
        """
        return a dataframe with 'userID', 'resourceID',
        'eventType',source','resourceType','action'

        """
        #drop the last two columns as they have >80% missing value
        df_trk=self.df_trk.iloc[1:,:-2]

        # reorder
        df_trk=df_trk.sort_values(by=['userID','eventDate'],ascending=[True,True]).reset_index(drop=True)

        #-------------------------------------------------------------------------------------------
        # Get queue status (returns 1 if is queued)
        df_trk['resourceType']=df_trk['eventType'].apply(lambda x: x.split('_')[0])
        df_trk['action']=df_trk['eventType'].apply(lambda x: x.split('_')[1])

        return df_trk




        # return a dataframe with userID, numOfResourcesToQueue, numOfCompletionFromQueue, RatioOfCompletion_num, minOfResourcesToQueue, minOfCompletedFromQueue, RatioOfCompletion_min










    def get_queue_features(self,specify_source=False):

        """
        return a dataframe with 'userID',
        'numOfResourcesToQueue', 'numOfCompletionFromQueue',
        'RatioOfCompletion_num', 'minOfResourcesToQueue',
        'minOfCompletedFromQueue', 'RatioOfCompletion_min'
        """

        df_trk=self.get_event_data()

        #Get resourceID and source
        df_src=df_trk_copy[['resourceID','source']].drop_duplicates()
        # create a dataframe with userID, numOfResourcesToQueue, numOfCompletionFromQueue, RatioOfCompletion_num, minOfResourcesToQueue, minOfCompletedFromQueue, RatioOfCompletion_min
        # create "isQueued" columns, returns 1 if Queued
        df_trk['isQueued']=df_trk['action'].map({'queued':1}).fillna(0).astype(int)
        # create "isCompleted" columns, returns 1 if Completed
        df_trk['isCompleted']=df_trk['action'].map({'completed':1, 'documented':1}).fillna(0).astype(int)

        df_trk=df_trk.groupby(['userID','resourceID']).sum().reset_index()
        # Get CompletionFromQueue (returns 1 if completed from the queue)
        df_trk['CompletedFromQueue']=((df_trk['isQueued']==1)&(df_trk['isCompleted']==1)).astype(int)


        df_trk=df_trk.merge(df_src,on='resourceID')




        # add mins from the resourceID
        df_res_select=df_res[['resourceID','min']].drop_duplicates()
        df_trk=df_trk.merge(df_res_select,on='resourceID')
        df_trk['minQueued']=df_trk['isQueued']*df_trk['min']
        df_trk['minCompleted']=df_trk['CompletedFromQueue']*df_trk['min']


        # if 'specified_source=True'
        df_trk1=df_trk.groupby(['userID','source']).sum().reset_index()
        df_trk1.rename(columns={'isQueued':'numQueued','CompletedFromQueue':'numCompletedFromQueue'},inplace=True)




        # if 'specified_source=False'
        if specify_source==False :
            df_trk2=df_trk1.groupby(['userID']).sum().reset_index()
            df_trk2['numQueued']=df_trk2['numQueued'].replace(0,-1)
            df_trk2['minQueued']=df_trk2['minQueued'].replace(0,-1)
            df_trk2['RatioOfCompletion_num']=np.abs(df_trk2['numCompletedFromQueue']/df_trk2['numQueued'])
            df_trk2['RatioOfCompletion_min']=np.abs(df_trk2['minCompleted']/df_trk2['minQueued'])
            # drop 'min' and 'isCompleted' as unnecessary
            df_trk2=df_trk2.drop(['min','isCompleted'],axis=1)
            #replace -1 back to 0
            df_trk2['numQueued']=df_trk2['numQueued'].replace(-1,0)
            df_trk2['minQueued']=df_trk2['minQueued'].replace(-1,0)
            return df_trk2
        #------------------------------------------------------------------

        # if 'specified_source=True'
        # this is used after clustering
        df_trk1['numQueued']=df_trk1['numQueued'].replace(0,-1)
        df_trk1['minQueued']=df_trk1['minQueued'].replace(0,-1)
        df_trk1['RatioOfCompletion_num']=np.abs(df_trk1['numCompletedFromQueue']/df_trk1['numQueued'])
        df_trk1['RatioOfCompletion_min']=np.abs(df_trk1['minCompleted']/df_trk1['minQueued'])
        #replace -1 back to 0
        df_trk1['numQueued']=df_trk1['numQueued'].replace(-1,0)
        df_trk1['minQueued']=df_trk1['minQueued'].replace(-1,0)
        # drop min and 'isCompleted'
        df_trk1=df_trk1.drop(['min','isCompleted'],axis=1)

        return df_trk1
