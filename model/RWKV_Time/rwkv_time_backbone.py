"""RWKV-Time 分位数预测骨干网络。"""

import keras
from keras import ops
from keras.layers import Input
from keras_hub.models import Backbone

from .attention_layer import TransfomrmerBlock
from .rwkv_time_layer import (
    Time_RWKV7_Block,
    TimeSeriesOutHead,
    TimeSeriesPatchEmbedding,
)


def rwkv_kernel_initializer(stddev=0.02):
    """构造 RWKV 权重使用的截断正态初始化器。

    Args:
        stddev: float。标准差。

    Returns:
        keras.initializers.TruncatedNormal 实例。
    """
    return keras.initializers.TruncatedNormal(stddev=stddev)


class RWKVTimeBackbone(Backbone):
    """RWKV-7 与因果 Transformer 交替堆叠的时间序列分位数预测骨干。

    Args:
        hidden_size: int。隐藏维度。
        head_size: int。RWKV 单头维度。
        num_layers: int。RWKV 块层数。
        intermediate_dim: int。FFN 与 patch 投影的中间维度。
        n_quantiles: int。输出分位数个数。
        patch_size: int。patch 长度。
        gate_lora: int。门控 LoRA 维度。
        mv_lora: int。value 混合 LoRA 维度。
        aaa_lora: int。alpha LoRA 维度。
        decay_lora: int。衰减 LoRA 维度。
        dropout_rate: float。Transformer 分支的 dropout 概率。
        dtype: 计算 dtype 策略，或 None 使用默认。
        mix_transformer_layer_num: int。Transformer 块总数，须为 num_layers 的
            正整数倍；为 0 时退化为纯 RWKV 堆叠。
        **kwargs: 其他层参数。

    Call arguments:
        inputs: dict，键为：
            time_series: [B, T, patch_size], float32。
            padding_mask: [B, T, patch_size], float32。1 表示有效。
            nan_mask: [B, T, patch_size], float32。1 表示该位置为 NaN。

    Returns:
        quantiles: [B, T, n_quantiles, patch_size], float32。
    """

    def __init__(
        self,
        hidden_size=512,
        head_size=64,
        num_layers=12,
        intermediate_dim=1024,
        n_quantiles=9,
        patch_size=32,
        gate_lora=128,
        mv_lora=32,
        aaa_lora=64,
        decay_lora=64,
        dropout_rate=0.0,
        dtype=None,
        mix_transformer_layer_num=0,
        **kwargs,
    ):
        self.patch_embed = TimeSeriesPatchEmbedding(
            patch_size=patch_size,
            hidden_dim=hidden_size,
            proj_h_dim=intermediate_dim,
            kernel_initializer=rwkv_kernel_initializer(),
            dtype="float32",
            name="patch_embed",
        )

        self.rwkv_blocks = [
            Time_RWKV7_Block(
                hidden_size=hidden_size,
                head_size=head_size,
                intermediate_dim=intermediate_dim,
                gate_lora=gate_lora,
                mv_lora=mv_lora,
                aaa_lora=aaa_lora,
                decay_lora=decay_lora,
                kernel_initializer=rwkv_kernel_initializer(),
                dtype=dtype,
                first_block=i == 0,
                name=f"time_rwkv_block_{i}",
            )
            for i in range(num_layers)
        ]

        if mix_transformer_layer_num != 0:
            assert mix_transformer_layer_num % num_layers == 0
            self.transformer_block = [
                TransfomrmerBlock(
                    intermediate_dim=intermediate_dim,
                    num_heads=hidden_size // head_size,
                    dropout=dropout_rate,
                    layer_norm_epsilon=1e-5,
                    kernel_initializer=rwkv_kernel_initializer(),
                    name=f"time_transformer_block_{i}",
                )
                for i in range(mix_transformer_layer_num)
            ]

        self.output_layer_norm = keras.layers.LayerNormalization(
            epsilon=1e-5,
            name="output_norm",
            dtype="float32",
        )
        self.output_head = TimeSeriesOutHead(
            n_quantiles=n_quantiles,
            out_patch_size=patch_size,
            hidden_dim=hidden_size,
            proj_h_dim=intermediate_dim,
            kernel_initializer=rwkv_kernel_initializer(),
            dtype="float32",
            name="output_head",
        )

        ts_input = Input(shape=(None, patch_size), dtype="float32", name="time_series")
        nan_mask = Input(shape=(None, patch_size), dtype="float32", name="nan_mask")
        padding_mask = Input(
            shape=(None, patch_size), dtype="float32", name="padding_mask"
        )
        h, padding_mask_patch = self.patch_embed(ts_input, padding_mask, nan_mask)

        v_first = None
        state = None
        if mix_transformer_layer_num == 0:
            for i in range(num_layers):
                h, v_first, state = self.rwkv_blocks[i](
                    h, v_first, padding_mask_patch, state
                )
        else:
            # 每 num_layers 个 RWKV 块前插入 mix_rate 个 Transformer 块
            mix_rate = mix_transformer_layer_num // num_layers
            mask_bool = ops.cast(padding_mask_patch, dtype="bool")
            for i in range(num_layers):
                for j in range(mix_rate):
                    h = self.transformer_block[i * mix_rate + j](h, mask_bool)
                h, v_first, state = self.rwkv_blocks[i](
                    h, v_first, padding_mask_patch, state
                )

        h = self.output_layer_norm(h)
        quantiles = self.output_head(h)
        super().__init__(
            inputs={
                "time_series": ts_input,
                "padding_mask": padding_mask,
                "nan_mask": nan_mask,
            },
            outputs=quantiles,
            dtype=dtype,
            **kwargs,
        )

        self.hidden_size = hidden_size
        self.head_size = head_size
        self.num_layers = num_layers
        self.intermediate_dim = intermediate_dim
        self.n_quantiles = n_quantiles
        self.patch_size = patch_size
        self.gate_lora = gate_lora
        self.mv_lora = mv_lora
        self.aaa_lora = aaa_lora
        self.decay_lora = decay_lora
        self.dropout_rate = dropout_rate
        self.mix_transformer_layer_num = mix_transformer_layer_num

    def get_config(self):
        return {
            **super().get_config(),
            "hidden_size": self.hidden_size,
            "head_size": self.head_size,
            "num_layers": self.num_layers,
            "intermediate_dim": self.intermediate_dim,
            "n_quantiles": self.n_quantiles,
            "patch_size": self.patch_size,
            "gate_lora": self.gate_lora,
            "mv_lora": self.mv_lora,
            "aaa_lora": self.aaa_lora,
            "decay_lora": self.decay_lora,
            "dropout_rate": self.dropout_rate,
            "mix_transformer_layer_num": self.mix_transformer_layer_num,
        }
