""" Library that includes the classes and objects to prepare the data, train
the blackbox-model and evaluate its performance. The blacbox model then will
be used to create the explanations using shap.
"""

from typing import List, Dict, Union, Any
from pydantic import BaseModel, ValidationError
from ._utils import (
    _syntetic_value, 
    _calculate_minimum_sample_size, 
    _automl,
    _samples_from_different_population,
    _pairs_out_of_range
)
import pandas as pd
import warnings
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
    estimators_ : Dict[str, Any] = {}
    r2_scores_ : Dict[str, float] = {}
    explainers_ : Dict[str, Any] = {}
    is_out_of_range_ : Dict[str, List[bool]] = {}

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

        if df_features.shape[0] < _calculate_minimum_sample_size():
            warnings.warn(
                "The number of samples is too small for the training of a "
                "black-box model that should produce estimated metrics whose"
                "distribution should be statistically comparable with the "
                "original metric distribution."
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

    def create_explainers(
        self,
        alpha_sample : float = 0.05,
        alpha_instance : float = 0.005
    ) -> None:
        """ Method to train the BlackBoxModel and create explainers.

        Parameters
        ----------
        alpha_sample : float
            Alpha level (significance level) to be used for the t-test to 
            evaluate if the estimated values follows the same distribution as 
            the original values.
        alpha_instance: float
            Alpha level (significance level) to be used for the calculation of
            the conficende interval to check whether an estimated value is too
            far from the original value.
        """
        
        X = pd.DataFrame(self.preprocessed_features)

        self.estimators_ = {}
        self.r2_scores_ = {}
        self.explainers_ = {}
        self.is_out_of_range_ = {}
        
        for metric in tqdm(self.metrics, "Metric loop"):
            y = pd.Series(self.metrics[metric], name=metric)
            best_estimator, r2_score = _automl(metric, X, y)
            
            self.estimators_[metric] = best_estimator
            self.r2_scores_[metric] = r2_score
            try:
                self.explainers_[metric] = shap.Explainer(best_estimator)
            except TypeError:
                self.explainers_[metric] = shap.KernelExplainer(
                    best_estimator.predict,
                    X
                )

            # Check if the model trained estimate the metric producing values
            # that are statistically comparable with the original metrics

            y_hat = best_estimator.predict(X)
            
            if _samples_from_different_population(
                y, 
                y_hat, 
                alpha=alpha_sample
            ):
                warnings.warn(
                    f"The best estimator for metric {metric} is producing "
                    "results that differs significantly with respect to the "
                    "original metrics. This could lead to produce misleading "
                    "explanations. Please change the features used to train "
                    "the black-box model."
                )

            # Check whether there are estimated values too far from the 
            # original values

            is_out_of_range, out_of_range = _pairs_out_of_range(
                y, 
                y_hat, 
                alpha=alpha_instance
            )
            if len(out_of_range) > 0:
                warnings.warn(
                    f"There are {len(out_of_range)} estimated values in the "
                    f"metric {metric} too far from the original values. The "
                    f"following is the list of indexes {out_of_range}."
                )
            self.is_out_of_range_[metric] = is_out_of_range