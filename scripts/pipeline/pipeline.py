import tempfile
import os
import time

from scripts.fingerprints.generate import FingerprintConfig, generate_fingerprints
from scripts.evaluate.calc_metrics import evaluate_matching, EvaluationConfig

from qdrant_client import models

class PipelineConfig:
    pipeline_dir = 'data/music_caps/pipeline'

    sampling_rate = 16000

    # Intervals Config
    interval_step = 2.5
    interval_duration_in_seconds = 5
    full_interval_duration_in_seconds = 10  # максимальная длинна заимствованного интервала для валидации

    # matched_threshold = 0.9

    # common data config
    embeddings_normalization = True
    audio_normalization = True

    model_name = 'UniSpeechSatForXVector' # UniSpeechSatForXVector, Wav2Vec2ForXVector, WavLMForXVector, Data2VecAudioForXVector
    model_from_pretrained = 'data/models/UniSpeechSatForXVector_finetuned/polished-meadow-36/'
    # model_name = 'Wav2Vec2ForXVector'

    # Validation Data Config
    validation_audios = 'data/music_caps/augmented_audios'
    validation_dataset = 'data/music_caps/augmented_audios.dataset'
    few_validation_samples = 10

    # Index Data Config
    index_audios = 'data/music_caps/audios'
    index_dataset = 'data/music_caps/audios.dataset'
    few_index_samples = 10

    # evaluation config
    augmented_dataset_path = 'data/music_caps/augmented_audios.dataset'
    distance_metric = models.Distance.COSINE


if __name__ == '__main__':

    pipeline_config = PipelineConfig()
    pipeline_config.pipeline_dir = os.path.join(pipeline_config.pipeline_dir, str(int(time.time())))
    os.makedirs(pipeline_config.pipeline_dir)

    print("pipeline_config.pipeline_dir", pipeline_config.pipeline_dir)

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
        base_embeddings_path=str(index_embeddings_directory),
        base_augmented_embeddings_path=str(validation_embeddings_directory),
        metrics_log_path=str(metrics_path),
        augmented_dataset_path=pipeline_config.augmented_dataset_path,
        sampling_rate=pipeline_config.sampling_rate,
        full_interval_duration_in_seconds=pipeline_config.full_interval_duration_in_seconds,
        interval_duration_in_seconds=pipeline_config.interval_duration_in_seconds,
        interval_step=pipeline_config.interval_step,
        distance_metric=pipeline_config.distance_metric,
        # matched_threshold=pipeline_config.matched_threshold
    )
    evaluate_matching(eval_config)

    # evaluation_config = TODO
