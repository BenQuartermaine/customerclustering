import pandas as pd
import numpy as np


class Documenting:
    def __init__(self):
        # DB connection
        env_path = find_dotenv()
        load_dotenv(env_path)
        conn = pymysql.connect(
        host = os.getenv('HOST'),
        port = int(3306),
        user = os.getenv('USER_DB'),
        password = os.getenv('PASSWORD_DB'),
        db = os.getenv('DB'),
        charset = 'utf8mb4')
        

    
    # Ratio of users subscribed to account age
    def get_ratio_subs_per_user(conn):
        df = pd.read_sql_query(

            """Returns number of times a users has had sub divided by account age, excludes users who have never had a sub"""
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
            """, conn)

        return df
    

    return document_training_set
