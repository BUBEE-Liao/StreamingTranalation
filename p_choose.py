from typing import Optional, final
import torch
import torch.nn as nn
# from fairseq2.typing import DataType, Device, finaloverride
from torch import Tensor
from torch.nn import AvgPool1d, Module, ModuleList, ReLU
from torch.nn.parameter import Parameter
from typing import Optional, Tuple
from loss import isNAN_isINF

class EnergyProjection(Module):
    def __init__(
        self,
        model_dim: int,
        num_layers: int,
        bias: bool = True,
        dtype: Optional[torch.dtype] = None,
        # device: Optional[Device] = None,
    ) -> None:
        super().__init__()

        if num_layers < 1:
            raise ValueError(
                f"Invalid `num_layers`: {num_layers} for EnergyProjectionLayer."
            )

        self.layers = ModuleList()

        for _ in range(num_layers):
            self.layers.append(
                nn.Linear(model_dim, model_dim, bias, dtype=dtype)
            )
            self.layers.append(ReLU())

    def forward(self, seqs: Tensor) -> Tensor:
        for layer in self.layers:
            seqs = layer(seqs)
        return seqs


@final
class PChooseLayer(Module):
    """Represents a PChoose layer."""

    model_dim: int
    num_heads: int
    energy_bias: Parameter
    monotonic_temperature: float
    q_energy_proj: EnergyProjection
    k_energy_proj: EnergyProjection
    keys_pooling: AvgPool1d

    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        energy_bias_value: float,
        monotonic_temperature: float,
        num_monotonic_energy_layers: int,
        pre_decision_ratio: int,
        *,
        bias: bool = True,
        # device: Optional[Device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        """
        :param model_dim:
            The dimensionality of the model.
        :param num_heads:
            The number of attention heads.
        :param bias:
            If ``True``, query, key energy projection layers learn an
            additive bias.
        """
        super().__init__()

        self.model_dim = model_dim
        self.num_heads = num_heads

        if energy_bias_value != 0.0:
            self.energy_bias = Parameter(
                torch.full([1], energy_bias_value, dtype=dtype)
            )
        else:
            self.register_module("energy_bias", None)

        self.monotonic_temperature = monotonic_temperature

        if num_monotonic_energy_layers <= 0:
            raise ValueError("Number of monotonic energy layers must be > 0.")

        self.q_energy_proj = EnergyProjection(
            self.model_dim,
            num_monotonic_energy_layers,
            bias,
            # device=device,
            dtype=dtype,
        )
        self.k_energy_proj = EnergyProjection(
            self.model_dim,
            num_monotonic_energy_layers,
            bias,
            # device=device,
            dtype=dtype,
        )

        self.keys_pooling = AvgPool1d(
            kernel_size=pre_decision_ratio,
            stride=pre_decision_ratio,
            ceil_mode=True,
        )

    # @finaloverride
    def forward(self, seqs: Tensor, keys: Tensor) -> Tensor:
        # print('seqs:', seqs)
        # print('keys:', keys)
        q = self.q_energy_proj(seqs)

        # (N, S, M) -> (N, H, S, K)
        q = q.unflatten(-1, (self.num_heads, -1)).transpose(1, 2)

        # (N, S_kv, M) -> (N, M, S_kv) -> (N, M, S_p)
        pooled_keys = self.keys_pooling(keys.transpose(1, 2))

        # (N, M, S_p) -> (N, S_p, M)
        pooled_keys = pooled_keys.transpose(1, 2)

        # if we want to pool the encoder sequence length, uncomment these three lines
        k = self.k_energy_proj(pooled_keys)
        # k = self.k_energy_proj(keys)

        # (N, S_p, M) -> (N, H, S_p, K)
        k = k.unflatten(-1, (self.num_heads, -1)).transpose(1, 2)

        # (N, H, S, K) @ (N, H, K, S_p) = (N, H, S, S_p)
        monotonic_energy = torch.matmul(q, k.transpose(-1, -2))

        monotonic_energy = monotonic_energy * (q.size(-1) ** -0.5)

        if self.energy_bias is not None:
            monotonic_energy += self.energy_bias
            

        # p_choose: (N, H, S, S_p)
        p_choose = torch.sigmoid(monotonic_energy / self.monotonic_temperature)
        if(isNAN_isINF(p_choose)):
            with open('outputLOG.txt', 'a')as f:
                print('p_choose', file=f)
                print(p_choose, file=f)
                print('seqs:', seqs, file=f)
                print('keys:', keys, file=f)
        return p_choose
