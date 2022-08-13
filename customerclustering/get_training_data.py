import pandas as pd
from db_connection import Db
from documentation import Documenting
from practice import *
from get_data import *

class GetTrainingData:
    def __init__(self,conn):
        self.conn=conn

# Returns a dataframe with the specified number of rows.
# If no row value is passed, all rows in the activity table will be returned

    def get_training_data(self,rows):
        """A function to include all our dataframes and merge them together"""
        """Let's get tables we need!"""

        df_subs_per_user = Documenting.get_ratio_subs_per_user(self)
        #df_practice=Practice(self.conn).get_practice_features(self)
        df_training=df_subs_per_user#.merge(df_practice,on='userID', how='inner')
        return df_training



if __name__ == '__main__':
    conn = Db.db_conn()

    df = GetTrainingData(conn).get_training_data(2000)

    print(df)
