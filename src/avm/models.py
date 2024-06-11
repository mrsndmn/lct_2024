from transformers import AutoFeatureExtractor, Wav2Vec2ForXVector, Data2VecAudioForXVector, WavLMForXVector, UniSpeechSatForXVector


def get_model(model_name, from_pretrained=None):
    if model_name == 'Wav2Vec2ForXVector':
        feature_extractor = AutoFeatureExtractor.from_pretrained("anton-l/wav2vec2-base-superb-sv")
        model = Wav2Vec2ForXVector.from_pretrained("anton-l/wav2vec2-base-superb-sv")
    elif model_name == 'Data2VecAudioForXVector':
        feature_extractor = AutoFeatureExtractor.from_pretrained("hf-tiny-model-private/tiny-random-Data2VecAudioForXVector")
        model = Data2VecAudioForXVector.from_pretrained("hf-tiny-model-private/tiny-random-Data2VecAudioForXVector")
    elif model_name == 'WavLMForXVector':
        feature_extractor = AutoFeatureExtractor.from_pretrained("hf-tiny-model-private/tiny-random-WavLMForXVector")
        model = WavLMForXVector.from_pretrained("hf-tiny-model-private/tiny-random-WavLMForXVector")
    elif model_name == 'UniSpeechSatForXVector':
        feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/unispeech-sat-base-plus-sv")

        if from_pretrained is None:
            from_pretrained = "microsoft/unispeech-sat-base-plus-sv"

        model = UniSpeechSatForXVector.from_pretrained(from_pretrained)
    else:
        raise ValueError(f"unknown model: {model_name}")

    return model, feature_extractor

