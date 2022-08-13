import pandas as pd
from db_connection import Db
from documentation import Documenting
from practice import *


class GetTrainingData:
    def __init__(self,conn, rows=0):
        self.conn=conn
        self.rows=rows

# Returns a dataframe with the specified number of rows.
# If no row value is passed, all rows in the activity table will be returned
    def activity_table_df(self, rows = 0):
            if self.rows == 0:
                df_acts = pd.read_sql_query(
                    "SELECT * FROM activity_20220808",
                    self.conn)
                return df_acts
            else:
                df_acts = pd.read_sql_query(
                    f"SELECT * FROM activity_20220808 LIMIT {rows}",
                    self.conn)
                return df_acts

    def get_training_data(self):
        """A function to include all our dataframes and merge them together"""
        """Let's get tables we need!"""
        df_act=self.activity_table_df(self.rows)
        df_subs_per_user = Documenting.get_ratio_subs_per_user(self)
        df_practice=Practice(self.conn).get_practice_features()
        df_training=df_subs_per_user.merge(df_practice,on='userID', how='inner')
        return df_training.head()



if __name__ == '__main__':
    conn = Db.db_conn()

    df = GetTrainingData(conn,rows=2000).get_training_data()

    print(df)
