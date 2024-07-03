import logging
import torch
from typing import Tuple, Union


torch.backends.cudnn.benchmark = True
AvailableDevice = "cuda" if torch.cuda.is_available() else "cpu"


def torch_inference(model: torch.nn.Module, *tensors: torch.Tensor):
    global AvailableDevice

    # Cast model, tensor to GPU
    if AvailableDevice == "cuda":
        logging.info(f"🔥 GPU Infer on device {torch.cuda.get_device_name()}")

        tensors = [t.to(torch.device("cuda"), non_blocking=True)
                   for t in tensors]

        # Cast model if model is not cast to device
        if next(model.parameters()).device.type != "cuda":
            model = model.to(torch.device("cuda"), non_blocking=True)

    # Inference
    model.eval()
    with torch.inference_mode():
        outputs: Union[torch.Tensor, Tuple[torch.Tensor]] = model(*tensors)

    # Cast back to CPU
    if AvailableDevice == "cuda":
        if isinstance(outputs, tuple):
            outputs = [o.cpu() for o in outputs]
        else:
            outputs = outputs.cpu()

    return outputs
