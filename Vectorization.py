from abc import ABC, abstractmethod
import numpy as np

class Vectorizer(ABC):
    def __init__(self, models):
        self.models = models
        self.model_starts = self.model_start_indices()

    @abstractmethod
    #the vecs of each model need to be grouped together to allow the identification of the model a element belongs to
    def verctorize(self):
        #takes as input an iterable of all elements to process and returns a vector representation of each of them
        pass
    #helper function which returns the start index of each models elements in the output array of vectorize
    def model_start_indices(self):
        ret = [0]
        count = 0
        for elements in self.models:
            count+= len(elements)
            ret.append(count)
        return ret[:-1]

    # helper function which converts a vector index to a model and element index
    # returns (model_index, element_index)
    def vec_to_model_index(self, index):
        count = 0
        for start in self.model_starts:
            if start >= index:
                return count, start-index
            count += 1

        return len(self.model_starts)-1, self.model_starts[-1]-index

        # helper function which converts a model and element index to a vector index
        def vec_to_model_index(self, model_index, vector_index):
           return self.model_starts[model_index]+ vector_index


#create a 0-1 vector where each entry of the vetor represents an attribute in the attribute space
#for each element a 0 represents the absence and a 1 the presence of said attribute
class Zero_One_Vectorizer(Vectorizer):

    def verctorize(self):
        dic = {}
        att_count = 0
        #iterate over all elements and attributes and store in a dic the mapping from attributes to vector indices in dic
        tot_len = 0
        for elements in self.models:
            tot_len+= len(elements)
            for element in elements:
                for attribute in element:
                    if attribute not in dic:
                        dic[attribute] = att_count
                        att_count += 1

        #create the matrix creating the vectors representing the elements
        ret_array = np.zeros(len(tot_len), att_count)
        count = 0
        for elements in self.models:
            for element in elements:
                for attribute in element:
                    ret_array[dic[attribute]][count] = 1
                count += 1

        return ret_array

class Mean_Count_Vectorizizer(Vectorizer):

    def vectorize(self):
        tot_len = 0
        for elements in self.models:
            tot_len += len(elements)

        ret_array = np.zeros(len(tot_len),2)

        count = 0
        for elements in self.models:
            for element in elements:
                ret_array[0][count] = len(element)
                ave = 0
                for att in element:
                    ave += len(att)
                ave /= len(element)
                ret_array[1][count] = ave








