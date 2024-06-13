import pytest

from avm.models import get_model

def test_model_params():

    model, _ = get_model('UniSpeechSatForXVector')

    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("num_parameters", num_parameters)

    assert num_parameters < 1100000

    return
