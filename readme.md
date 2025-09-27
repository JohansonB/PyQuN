# Raqun Lab

**Raqun Lab** is a comprehensive benchmarking program for comparing model matching algorithms. It provides an end-to-end framework for loading datasets, executing algorithms, comparing methods, and visualizing results. 

---

## **Key Features**

1. **Dataloader**
   - Efficiently loads and preprocesses datasets for benchmarking.
2. **Experiment Environment**
   - Streamlined setup for executing model matching algorithms.
3. **Comparison Framework**
   - Enables quantitative and qualitative comparison of algorithm outputs.
4. **Visualization Tools**
   - Visualize matches and comparative results of different algorithms.
5. **RaQuN Algorithm Variants**
  - Implements several variants of the RaQuN model matching algorithm.
  - Introduces a novel variation that applies dimensionality reduction as a preprocessing step, combining the benefits of both high-dimensional and low-dimensional vector encodings.

Detailed descriptions of each component are available in the respective subdirectories:

- [Dataloader](./dataloading)
- [Experiment Environment & Comparison Framework ](./experiment)
- [Visualization Tools ](./evaluation)

---

## **Example Use Case**

Below is an example illustrating the use of Raqun Lab to set up and execute experiments:

```python
if __name__ == "__main__":
    # Create the matching algorithms
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

    # Initialize the experiments
    match_all = DoMatching("do_matching_all", 5)
    vary_size = VarySize(0.1, 5,"vary_len",5)

    # Add the experiments to the execution pipeline
    ExperimentManager.add_experiment(match_all)
    ExperimentManager.add_experiment(vary_size)

    # Setup the matchall experiment
    ExperimentManager.add_strategies(match_all,[low_dim,high_dim,bfknn,svd_k10,svd_k50,lsh,lsh_svd])
    ExperimentManager.add_datasets(["hosp", "ppu", "argouml", "bcms", "bcs", "ppu_statem", "random", "randomLoose", "randomTight", "warehouses"],match_all)
    ExperimentManager.add_strategies(vary_size, [low_dim, high_dim, bfknn, svd_k10, svd_k50, lsh, lsh_svd])
    ExperimentManager.add_datasets(["hosp", "ppu", "argouml", "bcms", "bcs", "ppu_statem", "random", "randomLoose", "randomTight", "warehouses"],vary_size)

    # Run the experiments
    ExperimentManager.run_unfinished_experiments(ThreadPoolExecutor(max_workers=10))
```

This code snippet demonstrates:
- Creation of various model matching algorithms.
- Configuration of experiments.
- Addition of datasets and strategies to the experiment pipeline.
- Execution of experiments in parallel using a thread pool.