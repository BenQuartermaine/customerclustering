import pandas as pd
from customerclustering.db_connection import Db
from datetime import datetime
import numpy as np

#doubt: there are  over 307k users, 212k cust_sub account per user table;
#there are only 31k unique cust_ID in the stripesubsrciption table;
#do we have enough plan data to make this a viable feature;
#there are also dupe subscriptions in the stripe_subscription data, ie 'Customer ID` ='cus_CwbLsaFBPovaL4'

#define plan type based on the plan name whether there is periodic wording in it.

def plan_type(x):
    if "quarterly" in x:
        y = "quarterly"
    elif "monthly" in x:
        y = "monthly"
    elif "annually" in x:
        y = "annually"
    return y

def end_date(date):
    if date == '':
        date = datetime.today().strftime('%Y-%m-%d')
    else:
        date
    return date

class Subscribe:
    def __init__(self, conn):
        # Import data only once

        self.conn=conn

        self.sub_df = pd.read_sql_query("""select userID,
                                       stripeCustID,
                                       stripe_subscription.id,
                                       stripe_subscription.Plan,
                                       stripe_subscription.Quantity,
                                       stripe_subscription.Amount,
                                       stripe_subscription.Status,
                                       stripe_subscription.`Start (UTC)`,
                                       stripe_subscription.`Canceled At (UTC)`,
                                       stripe_subscription.`Ended At (UTC)`,
                                       stripe_subscription.`cancellation_reason (metadata)`
                                FROM user
                                inner join stripe_subscription
                                on user.stripeCustID = stripe_subscription.`Customer ID` ;""", conn)


    def subscriber_features(self):
        print('Getting subscriber features')
        sub_df = self.sub_df

        #re-format the date columns as datetime datatype;

        sub_df['start_date'] = pd.to_datetime(sub_df['Start (UTC)'])
        sub_df['cancel_date']= pd.to_datetime(sub_df['Canceled At (UTC)'])

        #applying plan type function to find out what periodic plan the customers signed up for

        sub_df['plan_type'] = sub_df['Plan'].apply(lambda x: plan_type(x))

        #assumption: use the latest plan subscribed as the plan type for the customer
        #if the customer has signed up mutilple times and different plans in the past, only use the lastest one

        plan_time_df = pd.pivot_table(sub_df, values = ['start_date'],\
                                            index = ['userID','plan_type'], aggfunc = np.max).\
                                            reset_index().sort_values(by ='userID').drop_duplicates()

        idx = plan_time_df.groupby(['userID'])['start_date'].transform(max) == plan_time_df['start_date']
        plan_df = plan_time_df[idx]


        #subscribed days as the difference between plan ended date and the plan started date;
        #where the plan is still active, using the current date as the end date for simplicity

        sub_df['ended_date'] = pd.to_datetime(sub_df['Ended At (UTC)'].apply(lambda x: end_date(x)))
        sub_df['subscribe_days']=(sub_df['ended_date']-sub_df['start_date']).dt.days
        active_df = pd.pivot_table(sub_df, values = ['subscribe_days'],\
                                            index = ['userID'], aggfunc = np.sum).reset_index()

        subscriber_df = plan_df.merge(active_df, how = 'inner', on = 'userID').drop(columns = 'start_date')

        return subscriber_df



if __name__ == '__main__':
    conn = Db.db_conn()

    subscribe=Subscribe(conn)
    df=subscribe.subscriber_features()
    print(df.head())
