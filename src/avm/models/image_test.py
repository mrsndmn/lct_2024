import pytest

from avm.models.image import get_default_image_model_for_x_vector

def test_model_params():

    model = get_default_image_model_for_x_vector()

    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("num_parameters", num_parameters)

    # num_parameters 6 862 692
    assert num_parameters < 10000000

    return
