""" Library that includes the classes and objects to prepare the data, train
the blackbox-model and evaluate its performance. The blacbox model then will
be used to create the explanations using shap.
"""

from typing import List, Dict, Union, Any
from pydantic import BaseModel, ValidationError
from ._utils import _syntetic_value
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from tqdm.auto import tqdm
import shap

class GenAIExplainer(BaseModel):
    """ Class to prepare data, train the blackbox model and evaluate its
    performance.

    Attributes
    ----------
    metrics : Dict[str, List[float]]
        Dictionary of strings to List of float numbers that defines the 
        Generative AI metrics already calculated that require explanations.
    features : Dict[str, List[Union[bool, List[str]]]]
        Features to be preprocessed.
    preprocessed_features : Dict[str, List[float]]
        Preprocessed features to be used for training the blackbox model.
    
    """

    metrics : Dict[str, List[float]]
    features : Dict[str, List[Union[bool, List[str]]]]
    preprocessed_features : Dict[str, List[float]] = {}
    models_ : Dict[str, Any] = {}
    r2_scores_ : Dict[str, float] = {}
    explainers_ : Dict[str, Any] = {}

    @classmethod
    def from_pandas(
        cls, 
        df_test_dataset : pd.DataFrame,
        df_features : pd.DataFrame
    ):
        """ Class method to create a Featurizer instance from two pandas 
        DataFrames. (1) df_test_dataset, that should contain at least one 
        column of type float and variance greater than zero that will be used 
        as Generative AI metrics columns, and (2) df_features that should 
        have the same number of records as df_test_dataset (same index) and all
        its columns will be used as features to train the BlackBoxModel.

        Parameters
        ----------
        df_test_dataset : pandas.DataFrame
            Pandas DataFrame that should at least one numerical column, with 
            variance greater than zero, that will be assumed to be the 
            Generative AI metrics columns that require explanations.
        df_features : pandas.DataFrame
            Pandas DataFrame that should have the same index as df_test_dataset
            and all its columns will be used as features to train the 
            BlackBoxModel
        """

        metrics = {}
        for column in df_test_dataset.columns:
            
            # Look for numerical columns, with no-nan values and non-zero 
            # variance
            if df_test_dataset[column].dtype in ("float64", "int64") and \
            not df_test_dataset[column].isna().any() and \
            df_test_dataset[column].var() > 0.0:
                metrics[column] = df_test_dataset[column].astype(
                    'float'
                ).to_list()
                
        if len(metrics) == 0:
            raise ValidationError(
                "No metrics columns found in df_test_dataset. Check that the "
                "metrics columns are numerical, with no-null values and its "
                "variance is greater than zero."
            )

        if df_features.shape[0] != df_test_dataset.shape[0]:
            raise ValidationError(
                "The number of records of df_features is different than the "
                "number of records of df_test_dataset."
            )

        if not (df_test_dataset.index == df_features.index).all():
            raise ValidationError(
                "The index of df_features is different than the index of "
                "df_test_dataset."
            )

        for column in df_features.columns:
            
            # Validation of non-null values
            if df_features[column].isna().any():
                raise ValidationError(
                    f"Column {column} has null values"
                )

            # Validation of the possible data types: List[str] or bool
            
            # List[str] case
            if df_features[column].apply(lambda x: isinstance(x, list)).all():
                # Check that each element of the lists are strings
                if not df_features[column].apply(
                    lambda lst: all(isinstance(item, str) for item in lst)
                ).all():
                    raise ValidationError(
                        f"Column {column} has non-list-of-strings elements"
                    )
            
            # bool case
            elif df_features[column].apply(
                lambda x: isinstance(x, bool)
            ).all():
                continue

            else:
                raise ValidationError(
                    f"Column {column} has unsupported dtype."
                )

            features = df_features.to_dict(orient='list')
            
        return cls(metrics = metrics, features = features)

    def _feature_engineering_bool(
        self,
        bool_feature : str
    ) -> pd.DataFrame:
        """ Method to perform engineering to a boolean feature

        Parameters
        ----------
        bool_feature : str
            Bool feature to engineer
        """
        
        # Cast to float
        bool_feat_ser = pd.Series(
            self.features[bool_feature], 
            name=bool_feature
        ).astype(float)

        #Check variance
        if bool_feat_ser.var() == 0.0:
            return pd.DataFrame()

        return pd.DataFrame(bool_feat_ser)

    def _feature_engineering_list(
        self,
        lst_feature : List[str]
    ) -> pd.DataFrame:
        """ Method to perform engineering to a list-of-strings feature

        Parameters
        ----------
        lst_feature : str
            List-of-strings feature to engineer
        """

        value_set : set = set()

        for lst in self.features[lst_feature]:
            for val in lst:
                syntetic_value = _syntetic_value(val)
                value_set.add(syntetic_value)
        
        if len(value_set) == 0:
            return pd.DataFrame()

        # TODO: Use embeddings insted of one hot encoding
        one_hot_dict = {}
        for one_hot_val in value_set:
            col_name = lst_feature + '__' + one_hot_val
            
            one_hot_vals = []
            for lst in self.features[lst_feature]:
                if one_hot_val in [_syntetic_value(val) for val in lst]:
                    one_hot_vals.append(1.0)
                else:
                    one_hot_vals.append(0.0)
                    
            one_hot_dict[col_name] = one_hot_vals

        return pd.DataFrame(one_hot_dict)
    
    def feature_engineering(
        self
    ) -> None:
        """ Method to perform feature engineering to prepare data for training
        """
        
        df_preprocessed = pd.DataFrame()
        
        for feat in self.features:
            feat_ser = pd.Series(self.features[feat])
            
            if feat_ser.apply(
                lambda x: isinstance(x, bool)
            ).all():
                df_feat_eng = self._feature_engineering_bool(feat)
            else:
                df_feat_eng = self._feature_engineering_list(feat)

            if df_preprocessed.shape[0] == 0:
                df_preprocessed = df_feat_eng
            else:
                df_preprocessed = df_preprocessed.join(df_feat_eng)

        self.preprocessed_features = df_preprocessed.to_dict(orient='list')

    # TODO: Test with other models like XGBoost
    def create_explainers(
        self,
    ) -> None:
        """ Method to train the BlackBoxModel
        """
        parameters = {
            'n_estimators': [20, 50, 100, 200],
            'max_depth': [10, 20, None]
            
        }

        X = pd.DataFrame(self.preprocessed_features)

        self.models_ = {}
        self.r2_scores_ = {}
        self.explainers_ = {}
        
        for metric in tqdm(self.metrics):
        
            model = GridSearchCV(
                estimator = RandomForestRegressor(n_jobs=-1, random_state=42), 
                param_grid = parameters,
                n_jobs=-1
            )
            y = pd.Series(self.metrics[metric], name=metric)
            model.fit(X,y)
            self.models_[metric] = model
            self.r2_scores_[metric] = r2_score(y, model.predict(X))
            self.explainers_[metric] = shap.Explainer(model.best_estimator_)
            
    