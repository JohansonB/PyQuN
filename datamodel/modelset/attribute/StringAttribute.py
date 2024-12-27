from RaQuN_Lab.datamodel.modelset.attribute.DefaultAttribute import DefaultAttribute


class StringAttribute(DefaultAttribute):
    def __init__(self, value:str = None) -> None:
        if not value is None and not isinstance(value, str):
            raise ValueError(f"StringAttribute must be initialized with a string. Got {type(value)} instead.")

        super().__init__(value)
