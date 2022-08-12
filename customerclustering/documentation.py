import pandas as pd
from db_connection import Db

class Documenting:
    def __init__(self, conn):
        self.conn = conn
        

    # Ratio of users subscribed to account age
    def get_ratio_subs_per_user(self):
        """
        Returns a DataFrame with:
        account_age, number of times a user has subscribed, stripeCustID and usedID
        """
        df = pd.read_sql_query(
            """
                SELECT 
                    Product, 
                    Status, 
                    userID,
                    stripeCustID,
                    COUNT(*) AS num_subs,
                    DATEDIFF(NOW(), createDate) AS account_age
                FROM stripe_subscription
                JOIN user ON user.stripeCustID = stripe_subscription.`Customer ID`
                GROUP BY `Customer ID` 
                ORDER BY COUNT(*) DESC
            """, self.conn)

        return df
    
    # Returns a df with averaeg minutes documented per resource, total mins and total documentations
    def get_total_mins_doc(self):
        """
        Returns a DataFrame with:
        total minutes documented, total documentations and the ratio of the two. 
        Takes a while to run.
        """
        df_doc = pd.read_sql_query(
            """
            SELECT
                owner,
                SUM(min) AS total_mins_doc,
                COUNT(*) AS total_docs
            FROM activity_20220808
            GROUP BY owner
            """, self.conn)
        
        df_doc['mins_per_documentation'] = round(df_doc['total_mins_doc'] / df_doc['total_docs'], 0)
        
        return df_doc
    
if __name__ == '__main__':
    conn = Db.db_conn()
    documenting = Documenting(conn)
    view_the_df = documenting.get_ratio_subs_per_user()
    print(view_the_df)