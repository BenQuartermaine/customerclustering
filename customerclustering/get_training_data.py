import pandas as pd
from customerclustering.db_connection import Db
from customerclustering.documentation import Documenting
from customerclustering.practice import *
from customerclustering.learning import *
from customerclustering.queued import *
from customerclustering.cpd import *
from customerclustering.activation import Activation
from customerclustering.subscriber import *


class GetTrainingData:
    def __init__(self, conn, rows=0):
        self.conn=conn
        self.rows=rows

# Returns a dataframe with the specified number of rows.
# If no row value is passed, all rows in the activity table will be returned
    def activity_table_df(self):
            if self.rows == 0:
                df_acts = pd.read_sql_query(
                    "SELECT * FROM activity_20220808",
                    self.conn)
                return df_acts
            else:
                df_acts = pd.read_sql_query(
                    f"SELECT * FROM activity_20220808 LIMIT {self.rows}",
                    self.conn)
                return df_acts

    def get_training_data(self):
        """A function to include all our dataframes and merge them together"""
        """Let's get tables we need!"""
        df_act = self.activity_table_df()
        df_lrn = Learning(self.conn, df_act).get_activity_features()
        df_que=Queue(self.conn).get_queue_features()
        df_subs_per_user = Documenting(self.conn).get_ratio_subs_per_user()
        df_practice=Practice(self.conn).get_practice_features()
        df_cpd=CPD(self.conn).cpd_event_day_diff()
        df_activation = Activation(self.conn, self.rows, df_act)
        print(df_activation.head())
        df_subscriber=Subscribe(self.conn).subscriber_features()

        df_training=df_subs_per_user \
            .merge(df_practice,on='userID', how='inner') \
            .merge(df_lrn, on='userID', how='inner') \
            .merge(df_que, on='userID', how='inner') \
            .merge(df_cpd, on='userID', how='inner') \
            .merge(df_subscriber,on='userID',how='inner')\
            .merge(df_activation, on='userID',how='inner')\


        return df_training

if __name__ == '__main__':
    conn = Db.db_conn()

    df = GetTrainingData(conn,rows=2000).activity_table_df()

    print(df.columns)
