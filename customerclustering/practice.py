from re import I
import pandas as pd
from customerclustering.db_connection import Db



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
        print('Getting CoP features')
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
        merged_df=df_cop.merge(df_pp_select,left_on='pProfile',right_on='pProfileID',how='right')\
        .drop_duplicates().rename(columns={'owner':'userID'})# rename 'owner' as 'userID'


        #merged_df[merged_df.duplicated(subset=['userID'],keep=False)]



        # create a column if the user has record in contextOfPractice
        merged_df['pProfile'].fillna(0,inplace=True)
        merged_df['hasPracticeRecord']=merged_df['pProfile'].apply(lambda x: 1 if x!=0 else 0)



        # some users have multiple profileID use the lastest pro profile, preferably with record
        merged_df.sort_values(by=['userID','hasPracticeRecord','startDate'],ascending=[True,False,False],inplace=True)
        merged_df=merged_df.drop_duplicates(subset=['userID'],keep='first')


        # drop 'pProfile' as redundant, move usersID and pProfileID to the front
        merged_df=merged_df.drop(['pProfile'],axis=1)
        cols = list(merged_df)
        # move the column to head of list using index, pop and insert
        cols.insert(0, cols.pop(cols.index('pProfileID')))
        cols.insert(0, cols.pop(cols.index('userID')))
        merged_df=merged_df[cols]

        # drop "startDate, createDate, endDate" as they are no longer needed, drop the 'hasPracticeRecord' as this is not accurate, will merge with
        merged_df.drop(['startDate', 'createDate', 'endDate'],axis=1,inplace=True)

        return merged_df

if __name__ == '__main__':
    conn = Db.db_conn()

    Prac=Practice(conn)
    df=Prac.get_practice_features()
    print(df.head())
