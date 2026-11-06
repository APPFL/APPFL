from collections import OrderedDict
import pickle
from typing import List, Union
from omegaconf import DictConfig
import numpy as np
import torch
from .base_compressor import BaseCompressor


class Int4Compressor(BaseCompressor):
    """
    Int4Compressor quantizes floating-point model parameters into packed 4-bit integers
    using block-level dynamic scaling, reducing communication footprint by approximately 8x
    without requiring external C/C++ library dependencies.

    :param compressor_config: configuration of the compressor
        - block_size (int): elements per quantization block (default: 64)
        - param_cutoff (int): minimum element threshold for compression (default: 128)
    """

    def __init__(self, compressor_config: DictConfig = None):
        super().__init__(compressor_config)
        self.cfg = compressor_config
        self.block_size = (
            getattr(compressor_config, "block_size", 64) if compressor_config else 64
        )
        self.param_cutoff = (
            getattr(compressor_config, "param_cutoff", 128)
            if compressor_config
            else 128
        )

    def _compress_tensor(self, tensor: torch.Tensor):
        if not tensor.is_floating_point() or tensor.numel() < self.param_cutoff:
            return ("raw", tensor.cpu().clone())

        orig_shape = tensor.shape
        flat = tensor.detach().flatten().float().cpu()
        numel = flat.numel()

        pad_len = (self.block_size - (numel % self.block_size)) % self.block_size
        padded = (
            torch.nn.functional.pad(flat, (0, pad_len), value=0.0)
            if pad_len > 0
            else flat
        )

        m = padded.numel() // self.block_size
        blocks = padded.view(m, self.block_size)

        max_abs = torch.max(torch.abs(blocks), dim=1, keepdim=True).values
        scale = torch.clamp(max_abs / 7.0, min=1e-8)

        quantized = torch.clamp(torch.round(blocks / scale), -8, 7).to(torch.int32)
        nibbles = (quantized + 8).to(torch.uint8)

        paired = nibbles.view(m, self.block_size // 2, 2)
        packed = ((paired[:, :, 0] << 4) | (paired[:, :, 1] & 0x0F)).flatten()

        return (
            "int4",
            packed.numpy().tobytes(),
            scale.squeeze(-1).numpy(),
            orig_shape,
            tensor.dtype,
        )

    def _decompress_tensor(self, payload):
        tag = payload[0]
        if tag == "raw":
            return payload[1]

        _, packed_bytes, scale_arr, orig_shape, dtype = payload
        scale = torch.from_numpy(scale_arr).float()
        m = scale.numel()
        block_bytes = self.block_size // 2

        packed_t = torch.from_numpy(
            np.frombuffer(packed_bytes, dtype=np.uint8).copy()
        ).view(m, block_bytes)
        high = (packed_t >> 4) & 0x0F
        low = packed_t & 0x0F

        unpacked = torch.stack([high, low], dim=2).view(m, self.block_size)
        values = unpacked.float() - 8.0
        scaled = values * scale.view(m, 1)

        total_elements = torch.Size(orig_shape).numel()
        return scaled.flatten()[:total_elements].view(orig_shape).to(dtype)

    def compress_model(
        self,
        model: Union[dict, OrderedDict, List[Union[dict, OrderedDict]]],
        batched: bool = False,
    ) -> bytes:
        if batched:
            if isinstance(model, list):
                compressed = [self.compress_model(m, batched=False) for m in model]
                return pickle.dumps(compressed)
            if isinstance(model, (dict, OrderedDict)):
                compressed = OrderedDict(
                    (k, self.compress_model(v, batched=False)) for k, v in model.items()
                )
                return pickle.dumps(compressed)

        compressed_dict = OrderedDict()
        for k, v in model.items():
            if isinstance(v, torch.Tensor):
                compressed_dict[k] = self._compress_tensor(v)
            elif isinstance(v, (dict, OrderedDict)):
                compressed_dict[k] = self.compress_model(v, batched=False)
            else:
                compressed_dict[k] = v

        return pickle.dumps(compressed_dict)

    def decompress_model(
        self,
        compressed_model: bytes,
        model: Union[dict, OrderedDict] = None,
        batched: bool = False,
    ) -> Union[OrderedDict, dict, List[Union[OrderedDict, dict]]]:
        data = pickle.loads(compressed_model)
        if batched:
            if isinstance(data, list):
                return [self.decompress_model(d, model, batched=False) for d in data]
            if isinstance(data, (dict, OrderedDict)):
                return OrderedDict(
                    (k, self.decompress_model(v, model, batched=False))
                    for k, v in data.items()
                )

        decompressed = OrderedDict()
        for k, v in data.items():
            if isinstance(v, tuple) and len(v) in (2, 5) and v[0] in ("raw", "int4"):
                decompressed[k] = self._decompress_tensor(v)
            elif isinstance(v, bytes):
                decompressed[k] = self.decompress_model(v, model, batched=False)
            else:
                decompressed[k] = v

        return decompressed
