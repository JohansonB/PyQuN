from RaQuN_Lab.strategies.RaQuN.candidatesearch.NNCandidateSearch.knn.KNN import KNN

from sklearn.neighbors import KDTree, BallTree



class TreeKNN(KNN):
    def __init__(self, tree_type="KDTree", leaf_size=40, metric="euclidean" ):
        self.tree_type = tree_type
        self.leaf_size = leaf_size
        self.metric = metric
        self.tree = None
        self.data_matrix = None
        self.vm_set = None

    def set_data(self, vm_set):

        if self.tree_type == "KDTree":
            self.tree = KDTree(vm_set.get_vec_mat(), leaf_size=self.leaf_size, metric=self.metric)
        elif self.tree_type == "BallTree":
            self.tree = BallTree(vm_set.get_vec_mat(), leaf_size=self.leaf_size, metric=self.metric)
        else:
            Exception("parameter type needs to be \"KDTree\" or \"BallTree\"")
        self.data_matrix = vm_set.get_vec_mat()
        self.vm_set = vm_set

    def get_neighbours(self, element, size):
        ret = []
        dist, ind = self.tree.query(self.data_matrix[self.vm_set.get_vec_index(element)].reshape(1,-1), k=size)
        ind = ind[0]
        for i in ind:
            ret.append(self.vm_set.get_ele_by_index(int(i)))
        return ret

