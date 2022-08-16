import pandas as pd
from customerclustering.db_connection import Db



class Goal:
    def __init__(self, conn):
        # Import data only once

        self.conn=conn

        self.df_goal=pd.read_sql_query("SELECT * FROM goal;", self.conn).drop_duplicates()

    def get_goals_per_year(self):
        """"
        Returns a dataframe with userID and average number of goals set each year
        """
        df_goal=self.df_goal
        df_goal['createDate']=pd.to_datetime(df_goal['createDate'])
        df_goal['createYear']=pd.DatetimeIndex(df_goal['createDate']).year

        # rename owner to userID
        df_goal.rename(columns={'owner': 'userID'},inplace=True)

        df_goal=df_goal.groupby(['userID','createYear']).count().reset_index()

        # get average goals per year
        df_goal=df_goal[['userID','createYear','goalID']]
        df_goal.rename(columns={'goalID': 'GoalsPerYear'},inplace=True)
        df_goal=df_goal.groupby('userID').mean().reset_index()

        # drop 'createYear'
        df_goal.drop(columns=['createYear'],inplace=True )

        return df_goal




if __name__ == '__main__':
    conn = Db.db_conn()

    goal=Goal(conn)
    df=goal.get_goals_per_year()
    print(df.head())
