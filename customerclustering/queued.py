import pandas as pd
import numpy as np
from customerclustering.db_connection import Db




class Queue:
    def __init__(self, conn):
        # Import data only once
        self.conn=conn
        self.df_trk = pd.read_sql_query("SELECT * FROM tracking_event;", self.conn).drop_duplicates()
        self.df_res = pd.read_sql_query("SELECT * FROM resource;", self.conn).drop_duplicates()

    def get_event_data(self):
        """return a dataframe with userID, eventType,
        source,eventYear, resourceType and action, min,
        minQueued, minCompletedFromQueue
        """

        # clean track_event
        # reorder the dataframe according to the userID and eventDate
        df_trk=self.df_trk
        #df_trk=df_trk_copy
        df_trk.isna().sum()/len(df_trk) #content & skipMigration have over 80% missing values, delete these columns. And drop eventID
        df_trk=df_trk.iloc[1:,:-2]
        #df_trk['eventType'].apply(lambda)


        # reorder
        df_trk=df_trk.sort_values(by=['userID','eventDate'],ascending=[True,True]).reset_index(drop=True)
        df_trk['eventDate']=pd.to_datetime(df_trk['eventDate'])
        df_trk['eventDate']=pd.DatetimeIndex(df_trk['eventDate']).year
        df_trk.rename(columns={'eventDate': 'eventYear'},inplace=True)


        # Get queueing status (returns 1 if completed from queue)
        df_trk['resourceType']=df_trk['eventType'].apply(lambda x: x.split('_')[0])
        df_trk['action']=df_trk['eventType'].apply(lambda x: x.split('_')[1])
        df_trk.head(20)
        df_trk.eventYear.unique()

        # Get CompletionFromQueue (returns 1 if completed from the queue)


        # Get mins learned per year(make this in a different py)



        #Get resourceID and source
        df_src=df_trk[['resourceID','source']].drop_duplicates()
        # create a dataframe with userID, ResourcesToQueuePerYear_num, CompletionFromQueuePerYear_num, RatioOfCompletion_num, ResourcesToQueuePerYear_min, CompletedFromQueuePerYear_min, RatioOfCompletion_min
        # create "isQueued" columns, returns 1 if Queued
        df_trk['isQueued']=df_trk['action'].map({'queued':1}).fillna(0).astype(int)
        # create "isCompleted" columns, returns 1 if Completed
        df_trk['isCompleted']=df_trk['action'].map({'completed':1, 'documented':1}).fillna(0).astype(int)


        # Get CompletionFromQueue (returns 1 if completed from the queue)
        df_trk['CompletedFromQueue']=((df_trk['isQueued']==1)&(df_trk['isCompleted']==1)).astype(int)



        df_trk=df_trk.drop_duplicates()

        #df_trk.head(20)

        # add mins from the resourceID
        df_res_select=self.df_res[['resourceID','min']].drop_duplicates()
        df_trk=df_trk.merge(df_res_select,on='resourceID')
        df_trk['minQueued']=df_trk['isQueued']*df_trk['min']
        df_trk['minCompletedFromQueue']=df_trk['CompletedFromQueue']*df_trk['min']
        #df_trk['minCompleted']=df_trk['isCompleted']*df_trk['min']


        return df_trk


        # return a dataframe with userID, numOfResourcesToQueuePerYear, numOfCompletionFromQueuePerYear, RatioOfCompletion_num, minOfResourcesToQueue, minOfCompletedFromQueuePerYear, RatioOfCompletion_min
    def get_queue_features(self,specify_source=False):
        print('Getting queue features')
        """
        return a dataframe with 'userID',
        'numOfResourcesToQueue', 'numOfCompletionFromQueue',
        'minOfResourcesToQueue','minOfCompletedFromQueue'
        """
        df_que=self.get_event_data()
        df_que_gr0=df_que.groupby(['userID','eventYear']).sum().reset_index()
        df_que_gr1=df_que_gr0.groupby('userID').mean().reset_index()
        df_que_gr1.drop(columns=['eventYear'],inplace=True)
        df_que_gr1.rename(columns={'isQueued':'numQueuedPerYear','isCompleted':'numCompletedOnelinePerYear',
                                'CompletedFromQueue':'numCompletedFromQueuePerYear',
                                'min':'minCompletedOnelinePerYear',
                                'minQueued':'minQueuedPerYear','minCompletedFromQueue':'minCompletedFromQueuePerYear'},inplace=True)

        return df_que_gr1


if __name__ == '__main__':
    conn = Db.db_conn()

    que=Queue(conn)
    df=que.get_queue_features()
    print(df.head())
