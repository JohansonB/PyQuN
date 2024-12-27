from RaQuN_Lab.utils.Utils import store_obj, load_obj


class MetaData:
    pass


class DataSet:
    def __init__(self, metadata: MetaData, data_model: 'DataModel'):
        self.metadata = metadata
        self.data_model = data_model

    def get_data_model(self) -> 'DataModel':
        return self.data_model

    def store(self, path: str) -> None:
        store_obj(self,path)

    @staticmethod
    def load(path: str) -> 'DataSet':
        return load_obj(path)

