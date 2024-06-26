import tempfile
import os
import time

from scripts.fingerprints.generate import FingerprintConfig, generate_fingerprints
from scripts.evaluate.calc_metrics import evaluate_matching, EvaluationConfig

from qdrant_client import models

from dataclasses import dataclass, field

@dataclass
class PipelineConfig:
    pipeline_dir: str = field(default='data/music_caps/pipeline')

    sampling_rate: int = field(default=16000)

    # Intervals Config
    interval_step: int = field(default=0.1)
    interval_duration_in_seconds: int = field(default=5)
    full_interval_duration_in_seconds: int = field(default=10)  # максимальная длинна заимствованного интервала для валидации

    threshold = 0.95

    # common data config
    embeddings_normalization: bool = field(default=True)
    audio_normalization: bool = field(default=True)

    model_name: str = field(default='UniSpeechSatForXVector')  # UniSpeechSatForXVector, Wav2Vec2ForXVector, WavLMForXVector, Data2VecAudioForXVector
    model_from_pretrained: str = field(default='data/models/UniSpeechSatForXVector_mini_finetuned/electric-yogurt-97')
    # model_name = 'Wav2Vec2ForXVector'

    # Validation Data Config
    validation_audios: str = field(default='data/music_caps/augmented_audios')
    validation_dataset: str = field(default='data/music_caps/augmented_audios.dataset')
    few_validation_samples: int = field(default=10)

    # Index Data Config
    index_audios: str = field(default= 'data/music_caps/audios')
    index_dataset: str = field(default= 'data/music_caps/audios.dataset')
    few_index_samples: int = field(default= 10)

    verbose: bool = field(default=True)


def run_pipeline(pipeline_config: PipelineConfig):

    pipeline_config.pipeline_dir = os.path.join(pipeline_config.pipeline_dir, str(int(time.time())))
    os.makedirs(pipeline_config.pipeline_dir)

    # print("pipeline_config.pipeline_dir", pipeline_config.pipeline_dir)

    index_embeddings_directory = tempfile.TemporaryDirectory(prefix="index_embeddings_", dir=pipeline_config.pipeline_dir)
    index_embeddings_directory = os.path.join(pipeline_config.pipeline_dir, 'index_embeddings')
    os.mkdir(index_embeddings_directory)

    index_fingerprint_config = FingerprintConfig(
        embeddings_out_dir=str(index_embeddings_directory),
        sampling_rate=pipeline_config.sampling_rate,
        base_audio_audio_path=pipeline_config.index_audios,
        dataset_path=pipeline_config.index_dataset,
        few_dataset_samples=pipeline_config.few_index_samples,
        interval_duration_in_seconds=pipeline_config.interval_duration_in_seconds,
        interval_step=pipeline_config.interval_step,
        batch_size=12,
        embeddings_normalization=pipeline_config.embeddings_normalization,
        audio_normalization=pipeline_config.audio_normalization,
        model_name=pipeline_config.model_name,
        model_from_pretrained=pipeline_config.model_from_pretrained,
    )
    generate_fingerprints(index_fingerprint_config)

    validation_embeddings_directory = os.path.join(pipeline_config.pipeline_dir, 'validation_embeddings')
    os.mkdir(validation_embeddings_directory)

    validation_fingerprint_config = FingerprintConfig(
        embeddings_out_dir=str(validation_embeddings_directory),
        sampling_rate=pipeline_config.sampling_rate,
        base_audio_audio_path=pipeline_config.validation_audios,
        dataset_path=pipeline_config.validation_dataset,
        few_dataset_samples=pipeline_config.few_validation_samples,
        interval_duration_in_seconds=pipeline_config.interval_duration_in_seconds,
        interval_step=pipeline_config.interval_step,
        batch_size=12,
        embeddings_normalization=pipeline_config.embeddings_normalization,
        audio_normalization=pipeline_config.audio_normalization,
        model_name=pipeline_config.model_name,
        model_from_pretrained=pipeline_config.model_from_pretrained,
    )
    generate_fingerprints(validation_fingerprint_config)

    metrics_path = os.path.join(pipeline_config.pipeline_dir, "metrics")
    eval_config = EvaluationConfig(
        index_embeddings_path=str(index_embeddings_directory),
        query_embeddings_path=str(validation_embeddings_directory),
        metrics_log_path=str(metrics_path),
        sampling_rate=pipeline_config.sampling_rate,
        full_interval_duration_in_seconds=pipeline_config.full_interval_duration_in_seconds,
        interval_duration_in_seconds=pipeline_config.interval_duration_in_seconds,
        interval_step=pipeline_config.interval_step,
        query_interval_step=pipeline_config.interval_step,
        threshold=pipeline_config.threshold,
        verbose=pipeline_config.verbose,
    )
    eval_metrics = evaluate_matching(eval_config)
    print("eval_metrics", eval_metrics)

    return eval_metrics

if __name__ == '__main__':

    pipeline_config = PipelineConfig()
    run_pipeline(pipeline_config)

    raise Exception()
