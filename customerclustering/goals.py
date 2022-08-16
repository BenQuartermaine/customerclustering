import pandas as pd
from customerclustering.db_connection import Db



class Goal:
    def __init__(self, conn):
        # Import data only once

        self.conn=conn

        self.df_goal=pd.read_sql_query("SELECT * FROM goal;", self.conn).drop_duplicates()
        # rename 'owner' to 'userID'
        self.df_goal.rename(columns={'owner': 'userID'},inplace=True)

    def get_goals_per_year(self):
        """"
        Returns a dataframe with userID and average number of goals set each year
        """
        df_goal=self.df_goal
        df_goal['createDate']=pd.to_datetime(df_goal['createDate'])
        df_goal['createYear']=pd.DatetimeIndex(df_goal['createDate']).year



        df_goal=df_goal.groupby(['userID','createYear']).count().reset_index()

        # get average goals per year
        df_goal=df_goal[['userID','createYear','goalID']]
        df_goal.rename(columns={'goalID': 'GoalsPerYear'},inplace=True)
        df_goal=df_goal.groupby('userID').mean().reset_index()

        # drop 'createYear'
        df_goal.drop(columns=['createYear'],inplace=True )

        return df_goal

    def get_ratio_achieved(self):
        """
        Retures a dataframe with all 'userID', 'ratioOfAchivedGoal'
        """

        df_goal=self.df_goal


        df_goal=df_goal.groupby('userID').agg({'goalID':'count', 'isAchieved': 'sum'}).reset_index()

        df_goal['ratioOfAchivedGoals']=df_goal['isAchieved']/df_goal['goalID']
        df_goal.rename(columns={'goalID': 'numOfGoals'},inplace=True)
        # drop 'isAchieved' and 'numOfGoals' columns

        df_goal.drop(columns=['numOfGoals','isAchieved'],inplace=True)

        return df_goal

    # join all goal titles together
    def get_meta_title(self):
        """
        Returns a dataframe with 'userID', 'metaGoalTitle'
        """

        df_goal=self.df_goal
        df_goal['metaGoalTitle']=df_goal.groupby('userID')[['title']].transform(lambda x: ' '.join(x))
        df_goal=df_goal[['userID','metaGoalTitle']].drop_duplicates()
        return df_goal


    def get_goal_features(self):
        df_goal=self.get_goals_per_year()\
            .merge(self.get_ratio_achieved(),on='userID',how='inner')\
            .merge(self.get_meta_title(),on='userID',how='inner')
        return df_goal





if __name__ == '__main__':
    conn = Db.db_conn()

    goal=Goal(conn)
    df=goal.get_goal_features()
    print(df.head())
