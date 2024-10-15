from PyQuN_Lab.CandidateSearch import NNCandidateSearch, TreeKNN, BFKNN
from PyQuN_Lab.DataModel import StringAttribute, Element, Model, ModelSet, Match
from PyQuN_Lab.Similarity import JaccardIndex, WeightMetric
from PyQuN_Lab.Vectorization import ZeroOneVectorizer, MeanCountVectorizer, VectorizedModelSet


def test_element(*string_attrs):
    ele = Element()
    for s in string_attrs:
        ele.add_attr(StringAttribute(s))
    return ele

def test_model(*elements):
    model = Model()
    for ele in elements:
        model.add_element(ele)
    return model

def test_model_set(*models):
    ms = ModelSet()
    for m in models:
        ms.add_model(m)



if __name__ == "__main__":
    #Model tests
    #input
    b = test_element("b","c","d","e")
    a = test_element("a","b","c")
    c = test_element("a","r", "f","g","h")
    d = test_element("a","f","h","t","i","j")
    e = test_element("l","o","p","q","a","s")
    f = test_element("l","r","a","s")
    g = test_element("h","i","j","t")

    model1 = test_model(a, b)
    model2 = test_model(c,d)
    model3 = test_model(e,f)
    model4 = test_model(g)

    ms = ModelSet()
    ms.add_model(model1)
    ms.add_model(model2)
    ms.add_model(model3)
    wm = JaccardIndex()
    match1 = Match(elements={a,c})
    match2 = Match(elements={e,g})
    print(Match.merge_matches(match1,match2))





