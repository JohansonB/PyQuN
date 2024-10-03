import numpy as np
from sklearn.neighbors import KDTree, BallTree
from Vectorization import Zero_One_Vectorizer
from Similarity import Jaccard_Index
from Model import element_list

class Neighbour_Tuple:
    def __init__(self, similarity, e1, e2):
        self.similarity = similarity
        self.e1 = e1
        self.e2 = e2
def get_matching(T, ele):
    for index, match in enumerate(T):
        if ele in T:
            return index, match

def is_valid_match(t1,t2):
    for e in t1:
        if e in t2:
            return False
    return True



def RaQuN(models, leaf_size=10, k=5, similarity = Jaccard_Index()):
    #Phase 1
    ele_list = element_list(models)
    vectorizer = Zero_One_Vectorizer(models)
    #vectors = vectorizer.verctorize()
    model_starts = vectorizer.model_start_indices()

    rng = np.random.RandomState(0)
    vectors  = rng.random_sample((100, 20))  # 10 points in 3 dimensions

    tree = KDTree(vectors, leaf_size=leaf_size)

    #Phase 2
    P = []
    #iterate over all vectors and determine its closest neighbours
    for i in range(vectors.shape[0]):
        dist, ind = tree.query(vectors[:i], k=k)
        cur_mod, cur_ele = vectorizer.vec_to_model_index(i)
        #iterate over the neighbours and add a Neighbour_Tuple to P if they dont belong to the same model
        for index, dist in zip(ind,dist):
            other_mod, other_ele = vectorizer.vec_to_model_index(index)

            if not cur_mod == other_mod:
                e1 = models[cur_mod].get_element(cur_ele)
                e2 = models[other_mod].get_element(other_ele)
                simi = similarity.similarity(e1,e2)
                #filter any tuples with no attributes in common (not taking zero due to precision errors maybe?)
                if(simi>0.0001):
                    P.append(Neighbour_Tuple(simi,e1, e2))

    #Phase 3
    P = sorted(P, key=lambda x: x.similarity , reverse=True)
    #initialize the matching list T
    T = [{e} for e in ele_list]

    for tuple in P:
        i1, t1 = get_matching(T,tuple.e1)
        i2, t2 = get_matching(T,tuple.e2)
        if is_valid_match(t1,t2) and similarity.should_match(t1, t2):
            if i1<i2:
                del T[i2]
                del T[i1]
            else:
                del T[i1]
                del T[i2]
            T.append(set.union(t1,t2))

    return T







if __name__ == "__main__":
    rng = np.random.RandomState(0)
    X = rng.random_sample((10, 3))  # 10 points in 3 dimensions
    tree = KDTree(X, leaf_size=2)
    dist, ind = tree.query(X[:1], k=3)
    print(ind)  # indices of 3 closest neighbors


