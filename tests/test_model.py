import torch

from src.model import get_model


def test_model_output_shape():
    model = get_model("simple_cnn", 10)

    x = torch.randn(2, 3, 32, 32)
    output = model(x)

    assert output.shape == (2, 10)
