import os
import pickle
from typing import Any


def store_obj(obj, path: str) -> None:
    # Add the .pkl extension to the file path
    path += '.pkl'

    # Create directories if they do not exist
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(obj, f)



def load_obj(path: str):
    path += '.pkl'
    with open(path, 'rb') as f:
        return pickle.load(f)


def set_field(obj: Any, field_name: str, value: Any) -> None:
    """
    Recursively sets a specified field to a given value in an object or its nested sub-objects.

    :param obj: The object to inspect and modify.
    :param field_name: The name of the attribute to set.
    :param value: The value to set for the attribute.
    """
    # Check if the object itself has the field
    if hasattr(obj, field_name):
        setattr(obj, field_name, value)

    # Check nested objects (attributes of the current object)
    if hasattr(obj, '__dict__'):
        for attr_value in vars(obj).values():
            set_field(attr_value, field_name, value)