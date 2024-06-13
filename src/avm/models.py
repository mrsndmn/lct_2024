from transformers import AutoFeatureExtractor, Wav2Vec2ForXVector, Data2VecAudioForXVector, WavLMForXVector, UniSpeechSatForXVector, UniSpeechSatConfig


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
            # from_pretrained = "microsoft/unispeech-sat-base-plus-sv"
            # претрейн оригинальной модельки очень тяжеловесный для нас
            config = UniSpeechSatConfig(
                vocab_size=32,
                hidden_size=256,
                num_hidden_layers=6,
                num_attention_heads=8,
                intermediate_size=512,
                hidden_act="gelu",
                hidden_dropout=0.1,
                activation_dropout=0.1,
                attention_dropout=0.1,
                feat_proj_dropout=0.0,
                feat_quantizer_dropout=0.0,
                final_dropout=0.1,
                layerdrop=0.1,
                initializer_range=0.02,
                layer_norm_eps=1e-5,
                feat_extract_norm="group",
                feat_extract_activation="gelu",
                conv_dim=(512, 512, 512, 512, 512, 512, 512),
                conv_stride=(5, 2, 2, 2, 2, 2, 2),
                conv_kernel=(10, 3, 3, 3, 3, 2, 2),
                conv_bias=False,
                num_conv_pos_embeddings=128,
                num_conv_pos_embedding_groups=16,
                do_stable_layer_norm=False,
                apply_spec_augment=True,
                mask_time_prob=0.05,
                mask_time_length=10,
                mask_time_min_masks=2,
                mask_feature_prob=0.0,
                mask_feature_length=10,
                mask_feature_min_masks=0,
                num_codevectors_per_group=320,
                num_codevector_groups=2,
                contrastive_logits_temperature=0.1,
                num_negatives=100,
                codevector_dim=256,
                proj_codevector_dim=256,
                diversity_loss_weight=0.1,
                ctc_loss_reduction="mean",
                ctc_zero_infinity=False,
                use_weighted_layer_sum=False,
                classifier_proj_size=256,
                tdnn_dim=(512, 512, 512),
                tdnn_kernel=(5, 3, 1),
                tdnn_dilation=(1, 3, 1),
                xvector_output_dim=256,
                pad_token_id=0,
                bos_token_id=1,
                eos_token_id=2,
                num_clusters=504,
            )
            model = UniSpeechSatForXVector(config)
        else:
            model = UniSpeechSatForXVector.from_pretrained(from_pretrained)
    else:
        raise ValueError(f"unknown model: {model_name}")

    return model, feature_extractor

