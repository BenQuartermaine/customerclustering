import pandas as pd
from customerclustering.db_connection import Db
from datetime import datetime
import numpy as np

#define a CPD due date as 31/05; and its due date for each year
#there are 73 different event types, The idea is to capture all eventtype as an evidence
#for interactivity vs CPD due time

def cpd_due_date(event_date):
    month = 5
    day = 31
    if datetime.strptime(str(event_date), "%Y-%m-%d %H:%M:%S").month <= 5:
        year = datetime.strptime(str(event_date), "%Y-%m-%d %H:%M:%S").year
    else:
        year = datetime.strptime(str(event_date), "%Y-%m-%d %H:%M:%S").year + 1
    return f"{year}/{month}/{day}"

class CPD:
    def __init__(self, conn):
        # Import data only once

        self.conn=conn

        self.track_event_df = pd.read_sql_query("select * from tracking_event;",self.conn)


    def cpd_event_day_diff(self):

        #calculate the day difference between event date and its corrsponding cpd due date
        track_event_df = self.track_event_df
        track_event_df['cpd_due']= pd.to_datetime(track_event_df['eventDate'].apply(cpd_due_date))
        track_event_df['event_cpd_day_diff'] = (track_event_df['cpd_due'] - track_event_df['eventDate']).dt.days +1
        #calculate the mean of event date and cpd due date difference through pivot table function

        cpd_day_diff_df = pd.pivot_table(track_event_df, values = ['event_cpd_day_diff'],\
                                    index = ['userID'], aggfunc = np.mean).reset_index()
        return cpd_day_diff_df



if __name__ == '__main__':
    conn = Db.db_conn()

    cpd=CPD(conn)
    df=cpd.cpd_event_day_diff()
    print(df.head())
