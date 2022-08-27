import io
import pandas as pd
import math
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn import set_config; set_config(display='diagram')

class GetViz:

    def get_df(self):
        def cluster_transform(x):
            if math.isnan(x) is True:
                return x
            else:
                return f"cluster {str(int(x)+1)}"

        #get data from the csv file with the label information
        cluster_df = pd.read_csv("raw_data/withlabel_tsne.csv",index_col=[0])
        cluster_df['label'] = cluster_df['label'].apply(cluster_transform)
        feature_df = cluster_df.drop(columns=['label'])
        return cluster_df, feature_df

    def get_feature_cat(self):

        cluster_df, feature_df = self.get_df()
        #getting date types from feature dataframe
        buffer = io.StringIO()
        feature_df.info(buf=buffer)
        lines = buffer.getvalue().splitlines()
        column_df = (pd.DataFrame([x.split() for x in lines[6:-2]], columns=lines[3].split())
            .drop('Count',axis=1)
            .rename(columns={'Non-Null':'Non-Null Count'}))

        #getting num and cat features
        cat_columns = column_df[column_df['Dtype']=='object'].reset_index().Column
        num_columns = column_df[column_df['Dtype']!='object'].reset_index().Column

        return cat_columns, num_columns


    def get_Kmeans(self):

        cluster_df, feature_df = self.get_df()
        cat_columns, num_columns = self.get_feature_cat()

        #create df for different feature categories to help with visualisation
        num_features_df = cluster_df[num_columns]
        num_features_df['label'] = cluster_df['label']

        cat_features_df = cluster_df[cat_columns]
        cat_features_df['label'] = cluster_df['label']

        #calculate and consolidate all kmeans in one dataframe
        num_kmean_df = num_features_df.groupby('label').mean().transpose()
        cat_kmean_df = cat_features_df.groupby('label')[cat_columns].agg(pd.Series.mode).transpose()
        kmean_df = num_kmean_df.append(cat_kmean_df)
        return kmean_df

    def clusters_one_feature(self,feature):

        cluster_df, feature_df = self.get_df()
        cat_columns, num_columns = self.get_feature_cat()

        #viz for num features using boxplot
        if feature in list(num_columns):
            viz = sns.boxplot(x = 'label', y = feature, hue = 'label', data = cluster_df, palette='bright'\
                    ,dodge = True)

            return viz

        #viz for cat features using heatmap
        else:
            cluster_df['count'] = 1
            heat_df = pd.pivot_table(cluster_df, values='count', index=[feature],
                                columns=['label'], aggfunc=np.sum, fill_value=0)

            for col in heat_df.columns:
                total = heat_df[col].sum()
                heat_df[col] = round(heat_df[col]/total,2)
            viz = sns.heatmap(heat_df, annot = True)

            return viz

    def cluster_two_feature(self,cluster,feature1,feature2):
        cluster_df, feature_df = self.get_df()

        cluster_df_filtered = cluster_df[cluster_df['label']== cluster]
        viz1 = sns.scatterplot(data = cluster_df_filtered, x = feature1, y = feature2, palette='bright')
        return viz1

if __name__ == "__main__":
    print(GetViz().get_Kmeans())
