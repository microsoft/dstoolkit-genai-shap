""" Library of internal reusable utilities
"""

import os
from pydantic import ValidationError
from openai import AzureOpenAI
from enum import Enum

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
