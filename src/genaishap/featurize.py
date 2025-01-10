"""This module automatize the creation of features to describe the "questions",
or "user inputs" included in the test dataset, and also automates the filling 
out of such features. Those features will be used later as "regressors" to 
train a black-box model to add explainability to the Generative AI metrics 
calculated
"""

from typing import List, Dict, Any, Union
from pydantic import BaseModel, ValidationError, PositiveInt
import pandas as pd
from openai import AzureOpenAI
import os
import math
import random
from tqdm.auto import tqdm
import warnings
from ._utils import (
    create_azure_openai_client, 
    split_list, 
    question_formating,
    FType
)
from .prompts import _CREATE_FEATURES_PROMPT, _FILL_OUT_FEATURES_PROMPT

class Feature(BaseModel):
    feature : str
    ftype: FType

class Features(BaseModel):
    features : List[Feature]

class FeatureValue(BaseModel):
    feature : str
    value : Union[bool, List[str]]

class FeaturesWithValue(BaseModel):
    question : str
    features : List[FeatureValue]

class FeatureValues(BaseModel):
    feature_values : List[FeaturesWithValue]

class Featurizer(BaseModel):
    """ Class to automate the creation of the features to be used as 
    regressors.

    Attributes
    ----------
    user_inputs : List[str]
        List of user inputs or questions of the test dataset.
    features : Dict[str, str]
        Dictionary that defines the list of features of the questions or
        user inputs that will be used as regressors for the black-box model
        to create the explanations.
    """
    
    user_inputs : List[str]
    features : Features = Features(features=[])
    _feature_values : FeatureValues(feature_values=[])

    @classmethod
    def from_pandas(cls, df : pd.DataFrame):
        """ Class method to create a Featurizer instance from a pandas 
        DataFrame. The DataFrame should contain a column named user_input of
        type string.

        Parameters
        ----------
        df : pandas.DataFrame
            Pandas DataFrame that should have a "user_input" column with the
            list of questions that are part of the test dataset.
        """

        if not isinstance(df, pd.DataFrame):
            raise ValidationError("Parameter is not a pandas DataFrame.")
        
        if 'user_input' not in df.columns:
            raise ValidationError("Missing user_input column in DataFrame.")

        if df['user_input'].isna().any():
            raise ValidationError("user_input column has null values")

        if not df['user_input'].is_unique:
            raise ValidationError("user_input column has duplicated values")
        
        try:
            user_inputs = df['user_input'].astype("string").to_list()
        except:
            raise ValidationError("Column user_input has non-string values.")

        return cls(user_inputs=user_inputs)

    # TODO: Include support to more model clients like OpenAI, HuggingFace or 
    # Llama
    def _create_features(
        self,
        model_client : Any,
        model_name : str,
        sample_size : float = 1.0,
        num_features : int = 12,
        seed : int = 42
    ) -> None:
        """ Generic method to ceate the question features using a language 
        model.

        Parameters
        ----------
        model_client : Any
            Client of the model to be used to call the model. Only AzureOpenAI
            is supported by now.
        model_name : str
            Name of the model to be used to create the question features.
        sample_size : float
            The default value of this parameter is 1.0 (100%), and it defines 
            the size of the sample of the list of user inputs (questions) to be
            used to create the features.  This parameter should be used only 
            when the total list of user inputs does not fit in the maximum 
            prompt size of the model used. 
        num_features : int
            Number of features to create, default 12.
        seed : int
            Random seed to be used when sample_size is less than 1.
        """

        if not isinstance(model_client, AzureOpenAI):
            raise ValidationError(
                "Only AzureOpenAI model client is supported by now."
            )

        if sample_size <= 0.0 or sample_size > 1.0:
            raise ValidationError(
                "sample_size value is not valid, > 0.0 and <= 1.0"
            )

        if sample_size == 1.0:
            questions = self.user_inputs
        else:
            sample_length = math.ceil(len(self.user_inputs) * sample_size)
            random.seed(seed)
            questions = random.sample(self.user_inputs, sample_length)
        
        questions_string = question_formating(questions)

        prompt = _CREATE_FEATURES_PROMPT.format(
            num_features=num_features, 
            questions=questions_string
        )

        # TODO: Check how to do it in a more generic way
        completion = model_client.beta.chat.completions.parse( 
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=0.0,
            response_format=Features 
        )
        
        self.features = completion.choices[0].message.parsed
        
    def create_features_using_azure_openai(
        self,
        deployment_name : str,
        sample_size : float = 1.0,
        num_features : int = 12,
        seed : int = 42
    ) -> None:
        """ Method to create the question features using an Azure OpenAI model.
        The following environment variables should be created to setup the 
        Azure OpenAI connection:
        * OPENAI_API_VERSION
        * AZURE_OPENAI_ENDPOINT
        * OPENAI_API_KEY

        Parameters
        ----------
        deployment_name : str
            Azure OpenAI deployment name of the model to be used to ceate the
            features.
        sample_size : float
            The default value of this parameter is 1.0 (100%), and it defines 
            the size of the sample of the list of user inputs (questions) to be
            used to create the features.  This parameter should be used only 
            when the total list of user inputs does not fit in the maximum 
            prompt size of the model used.
        num_features : int
            Number of features to create, default 12.
        seed : int
            Random seed to be used when sample_size is less than 1.
        """

        try: 
            sample_size = float(sample_size)
        except:
            raise ValidationError("sample_size is not numerical.")

        if sample_size <= 0.0 or sample_size > 1.0:
            raise ValidationError(
                "sample_size value is not valid, > 0.0 and <= 1.0"
            )

        azure_openai_client : AzureOpenAI = create_azure_openai_client()

        self._create_features(
            model_client = azure_openai_client,
            model_name = deployment_name,
            sample_size = sample_size,
            num_features = num_features,
            seed = seed
        )

    def _validate_fill_out_result(
        self,
        feature_values : FeatureValues
    ) -> None:
        """ Private Method to validate if the featurizer produced the same
        number of records as input and the features values are related to
        the same exact questions.

        Parameters
        ----------
        features_values : FeatureValues
            Final output of the featurizer.
        """

        if len(feature_values.feature_values) != len(self.user_inputs):
            raise ValidationError(
                "Filling output size is different than the input size. "
                "Consider the use of a bigger model >=gpt-4o."
            )
        
        differences = []
        for i in range(len(feature_values.feature_values)):
            question_output = feature_values.feature_values[i].question 
            question_input = self.user_inputs[i]
            if question_input.strip() != question_output.strip():
                differences.append((question_input,question_output))
        if len(differences) > 0:
            message = (
                "There are differences in some question input and output "
                "review the version of the language model, try to use a "
                "bigger model >=gpt-4o.  The differences are: \n"
            )
            message += "\n".join(
                [f"QIN: {qin}\nQOU: {qou}\n" for qin, qout in differences]
            )
            warnings.warn(message)


    # TODO: Include support to more model clients like OpenAI, HuggingFace or 
    # Llama
    def _fill_out_features(
        self,
        model_client : Any,
        model_name : str,
        batch_size : PositiveInt = 20
    ) -> None:
        """ Generic method to fillout the features.
        
        Parameters
        ----------
        model_client : Any
            Client of the model to be used to call the model. Only AzureOpenAI
            is supported by now.
        model_name : str
            Name of the model to be used to create the question features.
        batch_size : str
            Size of the batch of questions to fill out the features.
        """

        if not isinstance(model_client, AzureOpenAI):
            raise ValidationError(
                "Only AzureOpenAI model client is supported by now."
            )

        batches = split_list(self.user_inputs, batch_size)
        features_json = self.features.model_dump_json(indent=4)

        feature_value_list = []

        for questions in tqdm(batches):
        
            questions_string = question_formating(questions)
        
            prompt = _FILL_OUT_FEATURES_PROMPT.format(
                features = features_json,
                question_batch = questions_string
            )
    
            # TODO: Check how to do it in a more generic way
            completion = model_client.beta.chat.completions.parse( 
                model=model_name,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                temperature=0.0,
                response_format=FeatureValues 
            )
            
            feature_value_list.append(completion.choices[0].message.parsed)

        unified_list : List[FeaturesWithValue] = []

        for feature_values in feature_value_list:
            unified_list += feature_values.feature_values

        feature_values = FeatureValues(feature_values=unified_list)
        
        self._validate_fill_out_result(feature_values)

        self._feature_values = feature_values
                    
    def fill_out_features_using_azure_openai(
        self,
        deployment_name : str,
        batch_size : PositiveInt = 20
    ) -> None:
        """ Method to fillout the features using Azure OpenAI.
        
        The following environment variables should be created to setup the 
        Azure OpenAI connection:
        * OPENAI_API_VERSION
        * AZURE_OPENAI_ENDPOINT
        * OPENAI_API_KEY

        Parameters
        ----------
        deployment_name : str
            Azure OpenAI deployment name of the model to be used to fill out
            the features.
        batch_size : str
            Size of the batch of questions to fill out the features.
        """

        azure_openai_client : AzureOpenAI = create_azure_openai_client()
        
        self._fill_out_features(
            model_client = azure_openai_client,
            model_name = deployment_name,
            batch_size = batch_size
        )

    def to_pandas(self) -> pd.DataFrame:
        """ Method to convert the result of the featurizer into a Pandas 
        DataFrame
        """
        values_list = []
        for features_with_value in self._feature_values.feature_values:
            value_dict = {}
            for feature_value in features_with_value.features:
                value_dict[feature_value.feature] = feature_value.value
            values_list.append(value_dict)
        return pd.DataFrame(values_list)