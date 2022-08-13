# Create a training set data frame
import pandas as pd
from db_connection import Db
from documentation import Documenting
from practice import *

class GetData:
    def __init__(self, conn):
        self.conn=conn

# Returns a dataframe with the specified number of rows.
# If no row value is passed, all rows in the activity table will be returned
    def activity_table_df(self, rows = 0):
        if rows == 0:
            df_acts = pd.read_sql_query(
                "SELECT * FROM activity_20220808",
                self.conn)
            return df_acts
        else:
            df_acts = pd.read_sql_query(
                f"SELECT * FROM activity_20220808 LIMIT {rows}",
                self.conn)
            return df_acts


if __name__ == '__main__':
    conn = Db.db_conn()

    get_data = GetData(conn)
    df = get_data.activity_table_df(2000)
    print(df)
