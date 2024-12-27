from concurrent.futures import ThreadPoolExecutor

from RaQuN_Lab.datamodel.matching.similarities.match.WeightMetric import WeightMetric
from RaQuN_Lab.datamodel.matching.similarities.matching.SimilarityScore import SimilarityScore
from RaQuN_Lab.datamodel.matching.similarities.matching.statistics.Sum import Sum
from RaQuN_Lab.evaluation.MatchingView import MatchingView
from RaQuN_Lab.evaluation.ResultView import ResultView, NormalizedResultView
from RaQuN_Lab.evaluation.XYPlot import XYPlot
from RaQuN_Lab.evaluation.aggregators.AverageAggregator import AverageAgg
from RaQuN_Lab.experiment.DoMatching import DoMatching
from RaQuN_Lab.experiment.ExperimentManager import ExperimentManager
from RaQuN_Lab.experiment.VaryDimension import VaryDimension
from RaQuN_Lab.experiment.VarySize import VarySize
from RaQuN_Lab.strategies.RaQuN.RaQuN import VanillaRaQuN
from RaQuN_Lab.strategies.RaQuN.candidatesearch.NNCandidateSearch.NNCandidateSearch import NNCandidateSearch
from RaQuN_Lab.strategies.RaQuN.candidatesearch.NNCandidateSearch.knn.BFKNN import BFKNN
from RaQuN_Lab.strategies.RaQuN.candidatesearch.NNCandidateSearch.knn.LSHKNN import LSHKNN
from RaQuN_Lab.strategies.RaQuN.candidatesearch.NNCandidateSearch.vectorization.DimensionalityReduction.SVDReduction import \
    SVDReduction
from RaQuN_Lab.strategies.RaQuN.candidatesearch.NNCandidateSearch.vectorization.ZeroOneVectorization import \
    ZeroOneVectorizer

if __name__ == "__main__":
    #create the matching algorithms
    low_dim = VanillaRaQuN("2D-raqun")
    high_dim = VanillaRaQuN("high_dim_raqun", candidate_search=NNCandidateSearch(vectorizer=ZeroOneVectorizer()))
    bfknn = VanillaRaQuN("bfknn_raqun", candidate_search=NNCandidateSearch(knn=BFKNN()))
    svd_k10 = VanillaRaQuN("svd_k10_raqun",
                      candidate_search=NNCandidateSearch(vectorizer=ZeroOneVectorizer(), reduction=SVDReduction()))
    svd_k50 = VanillaRaQuN("svd_k50_raqun",
                      candidate_search=NNCandidateSearch(vectorizer=ZeroOneVectorizer(), reduction=SVDReduction(50)))
    lsh = VanillaRaQuN("LSH_raqun", candidate_search=NNCandidateSearch(vectorizer=ZeroOneVectorizer(), knn=LSHKNN()))
    lsh_svd = VanillaRaQuN("LSH_SVD_raqun", candidate_search=NNCandidateSearch(vectorizer=ZeroOneVectorizer(), knn=LSHKNN(),
                                                                          reduction=SVDReduction(50)))
    #initialize the experiments
    match_all = DoMatching("do_matching_all", 5)
    vary_size = VarySize(0.1, 5,"vary_len",5)

    #add the experiments to the execution pipeline
    ExperimentManager.add_experiment(match_all)
    ExperimentManager.add_experiment(vary_size)

    #setup the matchall experiment
    ExperimentManager.add_strategies(match_all,[low_dim,high_dim,bfknn,svd_k10,svd_k50,lsh,lsh_svd])
    ExperimentManager.add_datasets(["hosp", "ppu", "argouml", "bcms", "bcs", "ppu_statem", "random", "randomLoose", "randomTight", "warehouses"],match_all)
    ExperimentManager.add_strategies(vary_size, [low_dim, high_dim, bfknn, svd_k10, svd_k50, lsh, lsh_svd])
    ExperimentManager.add_datasets(["hosp", "ppu", "argouml", "bcms", "bcs", "ppu_statem", "random", "randomLoose", "randomTight", "warehouses"],vary_size)

    #run the experiments
    ExperimentManager.run_unfinished_experiments(ThreadPoolExecutor(max_workers=10))









    '''from PyQuN_Lab.Experiment import ExperimentResult

    XYPlot().plot("vary_dim", SimilarityScore(), AverageAgg(),["random"])
    view = ResultView(
        "do_matching_all")  # ,exclude_datasets=["randomLoose","randomTight","random"], exclude_strategies=["svd_k_10_high_dim_raqun", "random"])
    view.print_statistics(SimilarityScore(statistic=Sum()))
    #MatchingView.display(ExperimentResult.load("do_matching_all", "raqun", "hosp", 0, 0))'''