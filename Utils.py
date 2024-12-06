import os
import pickle
from typing import Any
import pickletools


def store_obj(obj, path: str) -> None:
    # Add the .pkl extension to the file path
    path += '.pkl'

    # Create directories if they do not exist
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        try:
            pickle.dump(obj, f)
        except TypeError as e:
            print(f"Failed to pickle object of type {type(obj)}: {e}")
            inspect_object(obj)



def inspect_object(obj, depth=0, max_depth=5):
    """Recursively inspect and print attributes to find the problematic object."""
    if depth > max_depth:  # Avoid infinite recursion
        print("Reached max depth of inspection.")
        return

    if hasattr(obj, '__dict__'):
        for key, value in obj.__dict__.items():
            try:
                print(f"{' ' * (depth * 2)}Inspecting attribute '{key}' of type {type(value)}")
                pickle.dumps(value)  # Test if the attribute can be pickled
            except Exception as e:
                print(f"{' ' * (depth * 2)}Cannot pickle attribute '{key}': {e}")
                inspect_object(value, depth + 1, max_depth)  # Inspect further into this attribute



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