# customise encoders to keep column names
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import OneHotEncoder,OrdinalEncoder,LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.manifold import TSNE



class CustomOneHotEncoder(OneHotEncoder):


    def transform(self, *arg,**kwargs):
        # YOUR CODE HERE

        array=super().transform(*arg,**kwargs)
        self.column_names=self.get_feature_names_out()
        return pd.DataFrame(array,columns=self.column_names)

    def fit_transform(self,*arg,**kwargs):
        array=super().fit_transform(*arg,**kwargs)
        self.column_names=self.get_feature_names_out()
        return pd.DataFrame(array,columns=self.get_feature_names_out())



class CustomColumnTransformer(ColumnTransformer):

    def transform(self, *args, **kwargs):
        array = super().transform(*args, **kwargs)
        df = pd.DataFrame(array,columns = self.get_feature_names_out())
        return df

    def fit_transform(self, *args, **kwargs):
        array = super().fit_transform(*args, **kwargs)
        df = pd.DataFrame(array,columns = self.get_feature_names_out())
        return df


class CustomStandardizer(TransformerMixin, BaseEstimator):

    def __init__(self):
        pass

    def fit(self, X, y=None):
        # Store what needs to be stored as instance attributes. Return "self" to allow chaining fit and transform.
        # $CHALLENGIFY_BEGIN
        self.means = X.mean()
        self.stds = X.std(ddof=0)
        # Return self to allow chaining & fit_transform
        return self


    def transform(self, X, y=None):
        # $CHALLENGIFY_BEGIN
        if not (hasattr(self, "means") and hasattr(self, "stds")):
            raise NotFittedError("This CustomStandardScaler instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.")
        # Standardization
        standardized_feature = (X - self.means) / self.stds
        return standardized_feature


    # $DELETE_BEGIN
    def inverse_transform(self, X, y=None):
        if not (hasattr(self, "means") and hasattr(self, "stds")):
            raise NotFittedError("This CustomStandardScaler instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.")
        return X * self.stds + self.means


class customTSNE(BaseEstimator, TransformerMixin):
    def __init__(self,n_components=2, perplexity=30,
                 random_state=None,n_jobs=-1,method='exact'):
        self.n_components = n_components
        self.perplexity=perplexity
        self.method = method
        self.random_state = random_state
        self.n_jobs=n_jobs

    def fit(self, X):
        ts = TSNE(n_components = self.n_components, perplexity=self.perplexity,
        method = self.method, random_state = self.random_state)
        self.X_tsne = ts.fit_transform(X)
        return self

    def transform(self, X):
        return X


def get_ord_encoder():
    # order columns
    feature_1_sorted_values = ['remote area','other rural area','small rural centre',
                                  'large rural centre','other metropolitan centre','metropolitan centre','capital city'] # 'located', need to check this
    feature_2_sorted_values = ['canceled','incomplete_expired','past_due', 'trialing','active',]
    feature_3_sorted_values = ['never','sometimes' ,'usually','always']
    feature_4_sorted_values = ['monthly','quarterly',  'annually']
    feature_5_sorted_values = [ 'minimal','moderate','high']
    feature_6_sorted_values = [ 'low complexity','generally complex', 'very complex','high complexity']



    # create categories iteratively: the shape of categories has to be (n_feature,)
    categories=[]

    categories_base=[
        feature_1_sorted_values,
        feature_2_sorted_values,
        feature_3_sorted_values,
        feature_4_sorted_values,
        feature_5_sorted_values,
        feature_6_sorted_values
        ]

    categories=categories_base


        #print(categories)

    ord_enc = OrdinalEncoder(
        categories=categories,
        handle_unknown="use_encoded_value",
        unknown_value=-1,
        encoded_missing_value=-1
        )
    return ord_enc
