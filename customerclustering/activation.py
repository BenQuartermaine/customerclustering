import pandas as pd
from customerclustering.db_connection import Db

class Activation:
    def __init__(self, conn, rows, activity_df):
        self.conn = conn
        self.rows = rows
        self.activity_df = activity_df
    
    def get_users(self):
        """Return a dataframe wiht userID and account create date"""
        df_users = pd.read_sql_query(
            f"""
            SELECT userID, createDate
            FROM user
            """, self.conn)    
        return df_users
    
    def get_activity_for_users(self):
        """Return a dataframe with userID, user create date, and all users activities documented"""
        df_activity = self.activity_df
        df_users = self.get_users()
        
        df_user_activity = df_activity.merge(right = df_users, left_on = 'owner', right_on = 'userID', how = 'left')
        
        return df_user_activity
    
    def get_activated_user_df(self):

        df_user_activity = self.get_activity_for_users()
        df_acts = df_user_activity[['userID', 'createDate_x', 'createDate_y']].copy()
        df_acts.rename(columns = {'createDate_x': 'account_creation', 'createDate_y': 'activity_creation'}, inplace = True)
        df_acts['date_diff'] = (df_acts['activity_creation'] - df_acts['account_creation']).dt.days
        df_acts['doc_in_activation'] = df_acts['date_diff'].apply(lambda x: 1 if x < 31 else 0)
        df_activated = df_acts[['userID', 'doc_in_activation']].groupby('userID').count()
        df_activated['activated'] = df_activated['doc_in_activation'].apply(lambda x: 0 if x < 2 else 1)
        df_activated.reset_index(inplace = True)

        return df_activated
    
if __name__ == '__main__':
    conn = Db.db_conn()

    df = Activation(conn, rows = 2000).get_activated_user_df()
    print(df)