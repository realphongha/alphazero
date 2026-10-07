"""Export a trained ZeroNet (``.pt``) to ONNX for lightweight CPU inference.

This is the *torch side* of the ONNX path: you run ``export`` once on a machine
with PyTorch (to produce an ``.onnx`` file), then use that file on any machine
with just ``onnxruntime`` (e.g. an Apple M1 Mac) via ``OnnxPolicyProvider`` /
``kind="onnx"`` bots -- no PyTorch required at inference time.

The exported graph is the model's raw ``forward``:
    obs (B, 3, n, n)  ->  policy_logits (B, n*n),  value (B, 1)
The legal-move masking + softmax are applied at inference time in
``OnnxPolicyProvider.forward_batch`` (identical to ``ZeroNet.predict_batch``).
"""
from __future__ import annotations

from typing import Optional


def export_to_onnx(model, path: str, board_dim: int, opset: int = 13,
                   model_name: Optional[str] = None) -> str:
    """Export ``model`` (a torch ``ZeroNet`` in eval mode) to ONNX at ``path``.

    Parameters
    ----------
    model : ZeroNet
        The trained network (any device). It is traced in eval mode.
    path : str
        Output ``.onnx`` file path.
    board_dim : int
        Board size ``n`` (so the input is ``B x 3 x n x n``).
    opset : int
        ONNX opset version (13 is broadly supported by onnxruntime).
    """
    import torch  # only needed at export time

    n = int(board_dim)
    x = torch.zeros(1, 3, n, n, device=model.device)
    x[:, 2] = 1.0  # channel 2 = legal mask; arbitrary values are fine for tracing
    model.eval()
    with torch.no_grad():
        torch.onnx.export(
            model, (x,), path,
            input_names=["obs"],
            output_names=["policy_logits", "value"],
            dynamic_axes={
                "obs": {0: "B"},
                "policy_logits": {0: "B"},
                "value": {0: "B"},
            },
            opset_version=opset,
            do_constant_folding=True,
            dynamo=False,  # use the legacy exporter (stable for these nets)
        )
    return path
