import pandas as pd
from db_connection import Db
#from customerclustering.db_connection import Db


class Practice:
    def __init__(self, conn):
        # Import data only once

        self.conn=conn

        self.df_cop = pd.read_sql_query("SELECT * FROM contextOfPractice;", self.conn).drop_duplicates()
        self.df_pp=pd.read_sql_query("SELECT * FROM professionalprofile;", self.conn).drop_duplicates()
        #self.df_cop=

    def get_practice_data(self):

        """
        Returns a DataFrame with:
        'pProfile', 'typeOfPractice', 'located', 'specialities',
       'population', 'focus', 'complex', 'autonomy', 'access',
       'startDate', 'endDate','createDate'
        """
        df_cop=self.df_cop[['pProfile', 'typeOfPractice', 'located', 'specialities',
       'population', 'focus', 'complex', 'autonomy', 'access', 'startDate',
       'endDate', 'createDate']]

        # df_cop.iloc[[3575]]['startDate'] is '-2019-12-09' must have been typo, manually update that
        df_cop.at[3575,'startDate']='2019-12-09'
        return df_cop





    def get_practice_features(self):
        """
        Returns a DataFrame with:
        'userID','pProfileID', 'typeOfPractice', 'located', 'specialities',
       'population', 'focus', 'complex', 'autonomy',
       'access','country'

        """
        df_cop=self.get_practice_data()
        # link with userID
        df_pp_select=self.df_pp[['pProfileID', 'owner','country']].drop_duplicates()
        df_pp_select.head()
        merged_df=df_cop.merge(df_pp_select,left_on='pProfile',right_on='pProfileID')\
        .drop_duplicates().rename(columns={'owner':'userID'})# rename 'owner' as 'userID'

        # drop 'pProfile' as redundant, move usersID and pProfileID to the front
        merged_df=merged_df.drop(['pProfile'],axis=1)
        cols = list(merged_df)
        # move the column to head of list using index, pop and insert
        cols.insert(0, cols.pop(cols.index('pProfileID')))
        cols.insert(0, cols.pop(cols.index('userID')))


        merged_df = merged_df.loc[:, cols].drop_duplicates().sort_values(by=['userID','startDate'],ascending=[True,False])

        # only keep the lastest
        merged_df.drop_duplicates(subset=['userID'],keep='first',ignore_index=True,inplace=True)
        #merged_df.head(10)

        # drop "startDate, createDate, endDate" as they are no longer needed
        merged_df.drop(['startDate', 'createDate', 'endDate'],axis=1)

        return merged_df

if __name__ == '__main__':
    conn = Db.db_conn()

    Prac=Practice(conn)
    df=Prac.get_practice_features()
    print(df.head())
