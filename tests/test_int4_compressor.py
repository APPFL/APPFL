from collections import OrderedDict
import pickle
import torch
from appfl.compressor import Int4Compressor


def test_compress_decompress_single_model():
    compressor = Int4Compressor()
    model = OrderedDict(
        [
            ("conv.weight", torch.randn(16, 3, 3, 3)),
            ("conv.bias", torch.randn(16)),
            ("fc.weight", torch.randn(64, 128)),
            ("step", torch.tensor(15, dtype=torch.int64)),
        ]
    )

    compressed_bytes = compressor.compress_model(model)
    assert isinstance(compressed_bytes, bytes)

    decompressed = compressor.decompress_model(compressed_bytes)

    assert decompressed["step"] == model["step"]
    assert decompressed["fc.weight"].shape == model["fc.weight"].shape
    assert decompressed["conv.weight"].shape == model["conv.weight"].shape

    # Check reconstruction MSE
    mse = torch.mean((decompressed["fc.weight"] - model["fc.weight"]) ** 2).item()
    assert mse < 0.05


def test_compress_decompress_batched_models():
    compressor = Int4Compressor()
    model1 = OrderedDict([("w", torch.randn(128))])
    model2 = OrderedDict([("w", torch.randn(128))])

    batched_models = [model1, model2]
    compressed_bytes = compressor.compress_model(batched_models, batched=True)
    restored_models = compressor.decompress_model(compressed_bytes, batched=True)

    assert len(restored_models) == 2
    assert restored_models[0]["w"].shape == model1["w"].shape
    assert restored_models[1]["w"].shape == model2["w"].shape


def test_param_cutoff_preservation():
    compressor = Int4Compressor()  # default cutoff 128
    small_tensor = torch.tensor([1.23, 4.56, 7.89], dtype=torch.float32)
    large_tensor = torch.randn(256, dtype=torch.float32)

    model = OrderedDict([("small", small_tensor), ("large", large_tensor)])
    compressed = compressor.compress_model(model)
    restored = compressor.decompress_model(compressed)

    # Small tensor below cutoff must be exactly identical
    assert torch.equal(restored["small"], small_tensor)


def test_compression_ratio():
    compressor = Int4Compressor()
    large_tensor = torch.randn(2048, dtype=torch.float32)
    model = OrderedDict([("weight", large_tensor)])

    raw_pickle = len(pickle.dumps(model))
    compressed = compressor.compress_model(model)
    compressed_len = len(compressed)

    assert compressed_len < raw_pickle // 3
