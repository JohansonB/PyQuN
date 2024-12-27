# Experiment Framework

The **Experiment Framework** provides a robust structure for benchmarking and comparing model matching algorithms. This framework enables the execution of experiments with various strategies (representing model matching algorithms) on specified datasets, repeated for a defined number of runs. It also allows for the variation of independent variables through user-defined experiments.

---

## **Overview**

An **Experiment** consists of:

1. **Strategies**:
   - These represent the model matching algorithms to be compared.
   - Defined as a `Strategy` interface.

2. **Datasets**:
   - The datasets on which the strategies will be executed.

3. **Repetitions**:
   - The number of times the experiment is repeated to ensure statistical significance.

4. **Independent Variable**:
   - Allows for varying one parameter of the experiment, such as the size of the dataset or dimensions of the model.
   - Users define this by implementing the `setup_experiment` method.

---

## **Abstract Base Class: `Experiment`**

The `Experiment` class serves as the foundation for all experiments. Users can extend this class to create custom experiments.

### Key Responsibilities:

- **Manage Experiment Metadata**:
  - Includes strategies, datasets, and the number of repetitions.
- **Persist and Load Experiments**:
  - Save experiment metadata to storage and load it when required.
- **Run Experiment Instances**:
  - Execute individual runs of the experiment.
- **User-Defined Setup**:
  - The `setup_experiment` method allows users to define how independent variables are varied across runs.

### Abstract Methods:

Users must implement the following methods:

1. **`num_runs()`**:
   - Returns the number of experimental runs.

2. **`is_sequential()`**:
   - Indicates whether the experiment runs should be executed sequentially.

3. **`setup_experiment(state, ze_input, strategy)`**:
   - Configures the input and strategy for each experimental run.

---

## **Example Implementation: `VaryDimension`**

The `VaryDimension` class demonstrates how to implement a custom experiment that varies the number of models in the dataset across runs.

### Code:

```python
class VaryDimension(IndependentExperiment):

    def __init__(self, num_runs, name : str, num_experiments: int, strategies : List['Strategy'] = [], datasets : List[str] = []):
        self.__runs = num_runs
        super().__init__(name, num_experiments, strategies, datasets)

    def num_runs(self) -> int:
        return self.__runs

    def setup_experiment(self, index: int, ze_input: 'ModelSet', strategy: 'Strategy') -> Tuple['ModelSet', 'Strategy']:
        num_models = int(len(ze_input)*index/self.__runs)
        if num_models < 2:
            num_models = 2
        return ze_input.get_subset(num_models), strategy

    def index_set(self):
        return [((index+1)/self.__runs) for index in range(self.num_runs())]

    def index_name(self):
        return "relative dimension"
```

### Explanation:

- **Purpose**:
  - Varies the number of models in the dataset relative to the total number of models across runs.

- **Key Methods**:
  - `setup_experiment`: Adjusts the subset of models based on the run index.
  - `index_set`: Defines the relative sizes used in the experiment.
  - `index_name`: Labels the independent variable.

---
# ExperimentManager

The **ExperimentManager** is a central class in the Experiment Framework that facilitates the execution and management of experiments. It provides functionality for adding strategies, datasets, and experiments to the execution pipeline, as well as handling the results of the experiments. This class plays a critical role in managing the setup, execution, and storage of experiments, making it easier to benchmark and compare model matching algorithms.

---

## Features

1. **Manage Experiment Metadata**:
   - Stores and retrieves strategies, datasets, and experiments for reuse.
   
2. **Run Experiments**:
   - Executes experiments in either parallel or sequential mode based on the configuration.
   
3. **Store and Load Results**:
   - Organizes experiment results in a directory structure, making them easy to access and analyze later.
   
4. **Add Strategies and Datasets**:
   - Allows users to add strategies and datasets to experiments, ensuring that all required elements are properly stored and accessible.

---

## Key Components

### 1. **ExperimentEnv**:
   - A helper class responsible for loading datasets and strategies, executing the experiment, and storing results.

### 2. **ExperimentConfig**:
   - An object that contains information about the experiment configuration, including the strategy, dataset, and experiment index.

### 3. **Methods for Running Experiments**:
   - **run_sequential_experiment**: A function to run experiments sequentially.
   - **get_unfinished_experiments**: Retrieves experiments that are not yet completed, allowing for resumption of interrupted experiments.
   
### 4. **Managing Strategies and Datasets**:
   - **add_strategy**: Adds a strategy to an experiment.
   - **add_datasets**: Adds datasets to an experiment, either by name or by file path.
   - **set_data_loader**: Sets the data loader for a strategy, making it compatible with the dataset.

---
