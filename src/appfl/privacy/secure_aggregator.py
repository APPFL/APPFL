import hmac
import hashlib
import struct
import math
from typing import Dict, Tuple, Iterable, List
import torch


class SecureAggregator:
    """
    Pairwise-masking secure aggregation (simulation).
    Clients mask their flattened delta using pairwise PRG seeds derived
    from (round_id, client_i, client_j, secret). Masks cancel when summed.
    """

    def __init__(
        self,
        client_id: str,
        all_client_ids: Iterable[str],
        secret: bytes,
        device: torch.device,
    ):
        self.client_id = str(client_id)
        self.all_client_ids = sorted([str(x) for x in all_client_ids])
        self.secret = secret
        self.device = device

    # ---- helpers ----
    @staticmethod
    def flatten_state_dict(
        sd: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, List[Tuple[str, Tuple[int, ...]]]]:
        parts = []
        shapes = []
        for k, v in sd.items():
            t = v.detach().to(torch.float32).reshape(-1)
            parts.append(t)
            shapes.append((k, tuple(v.shape)))
        flat = (
            torch.cat(parts)
            if len(parts) > 0
            else torch.tensor([], dtype=torch.float32)
        )
        return flat, shapes

    @staticmethod
    def unflatten_to_state_dict(
        flat: torch.Tensor,
        shapes: List[Tuple[str, Tuple[int, ...]]],
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        out = {}
        idx = 0
        for name, shp in shapes:
            num = 1
            for d in shp:
                num *= int(d)
            seg = flat[idx : idx + num].reshape(shp).to(device)
            out[name] = seg
            idx += num
        return out

    # deterministic seed derivation (symmetric)
    def _derive_seed(self, round_id: int, a: str, b: str) -> bytes:
        if a > b:
            a, b = b, a
        msg = f"rnd={round_id}|a={a}|b={b}".encode()
        return hmac.new(self.secret, msg, hashlib.sha256).digest()

    # PRG -> float32 normals (Box-Muller)
    def _seed_to_normals(self, numel: int, seed: bytes) -> torch.Tensor:
        vals = []
        ctr = 0
        while len(vals) < numel:
            block = hmac.new(seed, struct.pack(">Q", ctr), hashlib.sha256).digest()
            for off in range(0, len(block), 8):
                if len(vals) >= numel:
                    break
                chunk = block[off : off + 8]
                if len(chunk) < 8:
                    break
                u = struct.unpack(">Q", chunk)[0]  # 64-bit int
                u1 = (((u >> 32) & 0xFFFFFFFF) % (10**9) + 0.5) / (10**9 + 1.0)
                u2 = ((u & 0xFFFFFFFF) % (10**9) + 0.5) / (10**9 + 1.0)
                z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
                vals.append(z)
            ctr += 1
        return torch.tensor(vals[:numel], dtype=torch.float32, device=self.device)

    def make_pairwise_mask(self, flat_len: int, round_id: int) -> torch.Tensor:
        mask = torch.zeros(flat_len, dtype=torch.float32, device=self.device)
        for other in self.all_client_ids:
            if other == self.client_id:
                continue
            seed = self._derive_seed(round_id, self.client_id, other)
            r = self._seed_to_normals(flat_len, seed)
            if self.client_id < other:
                mask += r
            else:
                mask -= r
        return mask

    def mask_update(
        self, delta_state: Dict[str, torch.Tensor], round_id: int
    ) -> Tuple[torch.Tensor, List[Tuple[str, Tuple[int, ...]]]]:
        """
        Input: delta_state (state_dict of local - global)
        Returns: (masked_flat_tensor, shapes)
        shapes: list of (name, shape_tuple)
        """
        flat, shapes = self.flatten_state_dict(delta_state)
        mask = self.make_pairwise_mask(flat.numel(), round_id)
        return (flat + mask), shapes
