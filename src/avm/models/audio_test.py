import pytest

from avm.models.audio import get_default_audio_model

def test_model_params():

    model, _ = get_default_audio_model()

    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("num_parameters", num_parameters)

    assert num_parameters < 11000000

    return
