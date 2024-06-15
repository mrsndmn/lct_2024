import pytest

from avm.models.image import get_default_image_model

def test_model_params():

    model = get_default_image_model()

    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("num_parameters", num_parameters)

    # num_parameters 6 862 692
    assert num_parameters < 10000000

    return
