import pickle


def store_obj(obj, path: str) -> None:
    path += '.pkl'
    with open(path, 'wb') as f:
        pickle.dump(obj, f)



def load_obj(path: str):
    path += '.pkl'
    with open(path, 'rb') as f:
        return pickle.load(f)