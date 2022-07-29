import config
import torch
from model import YoloV3


def test_model_output_shape():
    x = torch.rand(1, 3, 416, 416)
    model = YoloV3(80)
    y = model(x)
    assert y[0].shape == torch.Size([1, 3, 13, 13, 85])
    assert y[1].shape == torch.Size([1, 3, 26, 26, 85])
    assert y[2].shape == torch.Size([1, 3, 52, 52, 85])
