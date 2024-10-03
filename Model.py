class Element:
    def __init__(self, attributes=[]):
        self.attributes = attributes

    def add_attribute(self, attribute):
        self.attributes.append(attribute)

    def get_attribute(self, index):
        return self.attributes[index]

    def shares_attribute(self, other):
        for att in other:
            if att in self.attributes:
                return True

        return False


    def __iter__(self):
        return iter(self.attributes)

class Model:
    def __init__(self, elements=[]):
        self.elements = elements

    def add_element(self, element):
        self.elements.append(element)

    def get_element(self, index):
        return self.elements[index]


    def __iter__(self):
        return iter(self.elements)

def element_list(models):
    return [ele for model in models for ele in models]