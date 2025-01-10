""" Library of internal reusable utilities
"""

import os
from pydantic import ValidationError
from openai import AzureOpenAI
from enum import Enum
from statsmodels.stats.power import TTestIndPower
import math
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from tqdm.auto import tqdm
from sklearn.metrics import r2_score
import numpy as np
from scipy.stats import ttest_rel
from scipy.stats import t

def _automl(
    metric: str,
    X: pd.DataFrame,
    y: pd.Series,
    random_state: int = 42,
    n_jobs: int = -1
):
    _models_and_params = {
        "XGBRegressor": {
            "estimator": xgb.XGBRegressor(
                objective='reg:squarederror', 
                random_state=random_state,
                n_jobs = n_jobs
            ),
            "params": {
                'n_estimators': [50, 100, 200, None],
                'max_depth': [3, 6, 10, None]
            }
        },
        "RandomForestRegressor": {
            "estimator": RandomForestRegressor(
                random_state=random_state,
                n_jobs = n_jobs
            ),
            "params": {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, 30, None]
            }
        },
        "SVR": {
            "estimator": SVR(),
            "params": {
                'C': [0.1, 1, 10, 100]
            }
        }
    }

    best_estimator = None
    best_r2_score:float = -1
    
    for _, estimator_info in tqdm(_models_and_params.items(), metric):
        grid_search = GridSearchCV(
            estimator=estimator_info['estimator'],
            param_grid=estimator_info['params'],
            scoring='neg_mean_squared_error',
            cv=5,
            n_jobs=n_jobs 
        )
        grid_search.fit(X, y)
        score = r2_score(y, grid_search.best_estimator_.predict(X))
        if score > best_r2_score:
            best_r2_score = score
            best_estimator = grid_search.best_estimator_

    return best_estimator, best_r2_score
        
def _samples_from_different_population(
    y: pd.Series, 
    y_hat: np.array,
    alpha: float = 0.05
) -> bool:
    """ Function to validate if y and y_hat are statistically from the 
    same population. No mayor differences. 
    
    Returns false if we failed to reject the null hypothesis of both samples,
    y and y_hat, came from the same population. False, otherwise.
    """
    _, p_value = ttest_rel(y, y_hat)

    return p_value < alpha

def _pairs_out_of_range(
    y: pd.Series, 
    y_hat: np.array,
    alpha: float = 0.005
) -> (list[bool], list[int]):
    """ Function to validate the difference of all pairs actual_value vs 
    estimated_value to check which pairs are out of the confidence interval 
    using an specific alpha 
    """

    differences = y - y_hat
    
    # Compute mean and standard error of the differences
    mean_diff = np.mean(differences)
    std_err = np.std(differences, ddof=1) / np.sqrt(len(differences))

    # Confidence level
    confidence = 1 - alpha
    degrees_of_freedom = len(differences) - 1

    # Critical t-value
    t_critical = t.ppf(confidence + alpha / 2, df=degrees_of_freedom)
    
    # Confidence interval
    ci_lower = mean_diff - t_critical * std_err
    ci_upper = mean_diff + t_critical * std_err

    is_out_of_range = []
    out_of_range = []
    for index in range(len(differences)):
        
        # Check if a specific pair is statistically different
        specific_pair_diff = y[index] - y_hat[index]
        is_out = False
        if specific_pair_diff < ci_lower or specific_pair_diff > ci_upper:
            is_out = True
            out_of_range.append(index)
        is_out_of_range.append(is_out)

    return is_out_of_range, out_of_range

def create_azure_openai_client() -> AzureOpenAI:
    """ Function to validate if the required environment variables are setted
    and creates the Azure OpenAI client
    """
    
    if not 'OPENAI_API_VERSION' in os.environ:
        raise ValidationError(
            "OPENAI_API_VERSION environment variable is not set"
        )

    if not 'AZURE_OPENAI_ENDPOINT' in os.environ:
        raise ValidationError(
            "AZURE_OPENAI_ENDPOINT environment variable is not set"
        )

    if not 'OPENAI_API_KEY' in os.environ:
        raise ValidationError(
            "OPENAI_API_KEY environment variable is not set"
        )

    return AzureOpenAI(
        api_version = os.environ['OPENAI_API_VERSION'],
        azure_endpoint = os.environ['AZURE_OPENAI_ENDPOINT'],
        api_key = os.environ['OPENAI_API_KEY']
    )

def split_list(lst: list, chunk_size: int) -> list:
    """ Function to split a list on batches of size chunck_size
    """
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]

def question_formating(questions: list[str]) -> str:
    """ Formats a list of questions into a string
    """
    return "\n".join([f"[QUESTION {idx}] {user_input}" \
                      for idx, user_input \
                      in zip(range(1,len(questions)+1), questions)])

class FType(str, Enum):
    """ Feature type class to define the supported features types.
    """
    BOOLEAN = "boolean"
    LIST_OF_STRINGS = "list_of_strings"

def _syntetic_value(val:str):
    """ Returns the syntetic representation of the val string

    Parameters
    ----------
    val : str
        Value to convert to systetic representation
    """
    return val.lower().strip().replace(' ','_')

def _calculate_minimum_sample_size(
    effect_size: float = 0.65, 
    alpha: float = 0.05, 
    power: float = 0.8
):
    """
    Calculate the minimum sample size required for a t-test.

    Parameters:
        effect_size (float): The standardized effect size (Cohen's d).
        alpha (float): The significance level (commonly 0.05).
        power (float): The power of the test (commonly 0.8 or 0.9).

    Returns:
        float: Minimum sample size per group.
    """
    power_analysis = TTestIndPower()
    sample_size = power_analysis.solve_power(effect_size=effect_size, 
                                             alpha=alpha, 
                                             power=power, 
                                             alternative='two-sided')
    return math.ceil(sample_size)