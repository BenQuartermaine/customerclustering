import os
import pymysql
from dotenv import load_dotenv, find_dotenv


def db_conn():
    env_path = find_dotenv()
    load_dotenv(env_path)

    conn = pymysql.connect (
        host = os.getenv('HOST'),
        port = int(3306),
        user = os.getenv('USER_DB'),
        password = os.getenv('PASSWORD_DB'),
        db = os.getenv('DB'),
        charset = 'utf8mb4'
    )
    return conn