import os
import pickle


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