
# Experiment Visualization and Plotting

This repository provides several classes for visualizing and analyzing the results of experiments: **MatchingView**, **XYPlot**, **NormalizedResultView**, and **ResultView**. These classes are designed to help users present experiment results and generate meaningful plots for comparative analysis. Below is a high-level summary of each class and its use case.

## 1. **MatchingView**

### Overview
The `MatchingView` class is used to display the results of a matching algorithm in a human-readable table format. Each match from the experiment is presented as a table where:

- **Rows** represent elements of a match.
- **Columns** represent attributes of those elements.
- Attributes are ordered by their occurrence frequency, with more common attributes appearing first.

This is ideal for visualizing the internal structure of a matching result, making it easy to spot trends, patterns, and rare attributes.

### Key Features:
- **Table display**: The results are shown in a tabular format with columns for the element ID, name, and dynamically ordered attributes.
- **Sorting by occurrence**: Attributes are sorted based on their frequency, with the most common attributes appearing first.
- **Flexible**: The class can be easily extended to accommodate different experiment result types.
  
### Example Usage:
```python
matching_view = MatchingView()
matching_view.display(experiment_result)
```

### Output:
For each match, the class outputs a formatted table where:
- The first two columns display the element ID and name.
- Additional columns are dynamically generated for each attribute, sorted by their occurrence count.

---

## 2. **XYPlot**

### Overview
The `XYPlot` class is designed to generate **line plots** comparing different strategies based on a specific metric and aggregated results. It takes data from an experiment, processes it, and generates both similarity and runtime comparison plots.

- **Similarity plots** show how different strategies compare across various datasets based on a specific evaluation metric.
- **Runtime plots** show the comparison of runtime (in seconds) for each strategy across datasets.

### Key Features:
- **Metric Aggregation**: Allows aggregation of evaluation metrics and runtime data across multiple strategies.
- **Customizable**: Users can exclude specific strategies from the comparison by providing a list of excluded strategies.
- **Error and Runtime matrices**: Handles error and runtime matrices, and can aggregate values for visual clarity.
- **Easy plotting**: Automatically generates line plots with customized titles, labels, and legends for clarity.

### Example Usage:
```python
xy_plot = XYPlot()
xy_plot.plot(experiment, metric, aggregator)
```

### Output:
- **Similarity plots**: Show how each strategy performs relative to the other strategies.
- **Runtime plots**: Display runtime comparison for each strategy, allowing users to evaluate the efficiency of different approaches.

---

## 3. **NormalizedResultView**

### Overview
The `NormalizedResultView` class is used to display **normalized similarity scores** for different strategies in an experiment, relative to a baseline strategy. The class computes and presents the similarity scores, normalized by the baseline strategy's performance, for each strategy across different datasets. This allows for a clearer comparison between the strategies and helps identify which strategies perform better when compared to the baseline.

### Key Features:
- **Normalization**: The class normalizes the similarity scores of each strategy by dividing them by the performance of a baseline strategy.
- **Flexible Configuration**: You can exclude specific strategies or datasets from the comparison.
- **Aggregated Results**: Results are aggregated using a specified aggregator, such as the average (default).
- **Pretty Table Output**: The results are displayed in a well-formatted table, making it easy to compare the normalized similarity scores across strategies.

### Example Usage:
```python
normalized_view = NormalizedResultView(experiment, run=0, repetition=-1, baseline_strategy="BaselineStrategy")
normalized_view.print_normalized_statistics(metric, baseline_strategy="BaselineStrategy")
```

### Output:
The class will output a table showing the **normalized similarity score** for each strategy (excluding the baseline strategy and any excluded strategies) for each dataset. This normalized score reflects how each strategy performs relative to the baseline, making it easier to understand the strategies' relative effectiveness.

---

## 4. **ResultView**

### Overview
The `ResultView` class is used to display detailed similarity statistics and runtime information for each strategy in an experiment. It computes and aggregates the performance of each strategy on a specific dataset, and displays the results in a clear, tabular format.

### Key Features:
- **Statistics Display**: Displays similarity scores and runtime data for each strategy across datasets.
- **Flexible Dataset Selection**: You can choose to display results for a single dataset or all datasets.
- **Run and Repetition Specific**: Results can be shown for specific experiment runs and repetitions, or for all runs and repetitions.
- **Runtime Information**: The runtime data for each strategy is included, and aggregation is done using a specified aggregator.
- **Pretty Table Output**: The results are displayed in a formatted table for easy comparison.

### Example Usage:
```python
result_view = ResultView(experiment, run=0, repetition=-1, dataset="Dataset1")
result_view.print_statistics(metric)
```

### Output:
For each dataset, the class will output a table showing:
- The **similarity score** for each strategy.
- The **runtime data** for each strategy, sorted by stopwatch keys.
This helps compare both the effectiveness (similarity) and efficiency (runtime) of each strategy in the experiment.