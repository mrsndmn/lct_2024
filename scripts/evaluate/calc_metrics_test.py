# import pytest
from datasets import Dataset
from scripts.evaluate.calc_metrics import EvaluationConfig, evaluate_metrics

def test_metrics_calc():
    dataset_path = "data/music_caps/pipeline/1718132751/metrics/metrics.dataset/"
    metrics_dataframe = Dataset.load_from_disk(dataset_path).to_pandas()

    config = EvaluationConfig()
    metrics_values = evaluate_metrics(config, metrics_dataframe)
    print('metrics_values', metrics_values)

    raise Exception


if __name__ == '__main__':
    test_metrics_calc()