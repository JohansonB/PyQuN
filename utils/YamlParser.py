import yaml

from RaQuN_Lab.experiment.DoMatching import DoMatching
from RaQuN_Lab.experiment.ExperimentManager import ExperimentManager
from RaQuN_Lab.experiment.VaryDimension import VaryDimension
from RaQuN_Lab.experiment.VaryParameter import VaryParameter
from RaQuN_Lab.experiment.VarySize import VarySize


def parse_experiment_config(yaml_file: str) -> None:
    with open(yaml_file, 'r') as file:
        config = yaml.safe_load(file)

    experiment_name = config['name']
    experiment_type = config['type']
    num_experiments = config.get('num_experiments', 1)
    strategies = config.get('strategies', [])
    datasets = config.get('datasets', [])

    if experiment_type == "DoMatching":
        e = DoMatching(experiment_name, num_experiments, strategies, datasets)

    elif experiment_type == "VaryDimension":
        num_runs = config['num_runs']
        e = VaryDimension(num_runs, experiment_name, num_experiments, strategies, datasets)

    elif experiment_type == "VarySize":
        init_length = config['init_length']
        num_runs = config['num_runs']
        e =  VarySize(init_length, num_runs, experiment_name, num_experiments, strategies, datasets)

    elif experiment_type == "VaryParameter":
        parameter_name = config['parameter_name']
        parameter_values = config['parameter_values']
        e =  VaryParameter(parameter_name, parameter_values, experiment_name, num_experiments, strategies, datasets)

    else:
        raise ValueError(f"Unsupported experiment type: {experiment_type}")

    ExperimentManager.add_experiment(e)