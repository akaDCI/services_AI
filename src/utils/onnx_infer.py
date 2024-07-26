import logging
import onnxruntime
import numpy.typing as npt


def onnx_interence_session(model_path: str):
    return onnxruntime.InferenceSession(model_path, providers=[
        "CPUExecutionProvider", "AzureExecutionProvider"
    ])


def onnx_inference(session: onnxruntime.InferenceSession, *inputs: npt.NDArray) -> npt.NDArray:
    logging.info(f"🚅 ONNX Infer with {', '.join(session.get_providers())}")
    _inputs = {_i.name: _input for _i,
               _input in zip(session.get_inputs(), inputs)}
    _output = session.run(None, _inputs)
    return _output[0]
