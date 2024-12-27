# Experiment Visualization and Plotting

This repository provides two key classes for visualizing and analyzing the results of experiments: **MatchingView** and **XYPlot**. These classes are designed to help users present experiment results and generate meaningful plots for comparative analysis. Below is a high-level summary of each class and its use case.

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