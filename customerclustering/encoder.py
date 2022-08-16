# customise encoders to keep column names
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import OneHotEncoder,OrdinalEncoder,LabelEncoder
from sklearn.compose import ColumnTransformer



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
