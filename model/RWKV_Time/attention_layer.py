"""RWKV-Time 骨干中交替使用的 Transformer 注意力与 FFN 组件。"""

import keras
from keras import ops
from keras_hub.src.models.qwen3.qwen3_attention import Qwen3Attention


def fp32_swish(x):
    """fp32 精度 swish。

    Args:
        x: 任意形状张量。

    Returns:
        与输入同 dtype 的 swish 结果。
    """
    dtype = x.dtype
    x = ops.cast(x, "float32")
    return ops.cast(ops.swish(x), dtype)


def fp32_sigmoid(x):
    """fp32 精度 sigmoid。

    Args:
        x: 任意形状张量。

    Returns:
        与输入同 dtype 的 sigmoid 结果。
    """
    dtype = x.dtype
    x = ops.cast(x, "float32")
    return ops.cast(ops.nn.sigmoid(x), dtype)


class GateAttention(Qwen3Attention):
    """在 Qwen3 因果自注意力上增加输出门控的注意力层。

    Args:
        沿用 Qwen3Attention 的参数。

    Call arguments:
        hidden_states: [B, T, C], float。
        attention_mask: [B, T, T], bool 或 None。True 表示可见。
        training: bool 或 None。

    Returns:
        [B, T, C]，与输入同 dtype。
    """

    def build(self, inputs_shape):
        super().build(inputs_shape)
        self._gate_dense = keras.layers.EinsumDense(
            equation="bkm,mvh->bkvh",
            output_shape=(
                None,
                self.num_key_value_heads,
                self.head_dim,
            ),
            kernel_initializer=self.kernel_initializer,
            dtype=self.dtype_policy,
            name="gate",
        )
        self._gate_dense.build(inputs_shape)

    def call(
        self,
        hidden_states,
        attention_mask=None,
        training=None,
    ):
        query = self._query_dense(hidden_states)
        gate = self._gate_dense(hidden_states)
        query = self._query_dense_layer_norm(query)

        query = self.rotary_embedding_layer(query)

        def _compute_key_value(x):
            key = self._key_dense(x)
            key = self._key_dense_layer_norm(key)
            key = self.rotary_embedding_layer(key)

            value = self._value_dense(x)

            return key, value

        key, value = _compute_key_value(hidden_states)

        key = ops.repeat(key, repeats=self.num_key_value_groups, axis=2)
        value = ops.repeat(value, repeats=self.num_key_value_groups, axis=2)

        attention_output = self._compute_attention(
            query,
            key,
            value,
            attention_mask,
        )

        attention_output = self._dropout_layer(attention_output, training=training)

        attention_output = self._output_dense(attention_output * fp32_sigmoid(gate))

        return attention_output


class FFN(keras.layers.Layer):
    """带门控的前馈网络，使用 RMSNorm 预归一化与残差连接。

    Args:
        intermediate_dim: int。中间维度。
        dropout: float。dropout 概率。
        kernel_initializer: 权重初始化器。
        **kwargs: 其他层参数。

    Call arguments:
        x: [B, T, C], float。
        training: bool 或 None。

    Returns:
        [B, T, C]，与输入同 dtype。
    """

    def __init__(
        self,
        intermediate_dim,
        dropout=0,
        kernel_initializer="glorot_uniform",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.intermediate_dim = intermediate_dim
        self.dropout = dropout
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.supports_masking = True
        self.autocast = False

    def build(self, inputs_shape):
        hidden_dim = inputs_shape[-1]
        self._feedforward_layer_norm = keras.layers.RMSNormalization(
            dtype=self.dtype_policy,
            name="feedforward_layer_norm",
        )
        self._feedforward_layer_norm.build(inputs_shape)
        self._feedforward_intermediate_dense = keras.layers.Dense(
            self.intermediate_dim,
            kernel_initializer=self.kernel_initializer,
            use_bias=False,
            dtype=self.dtype_policy,
            name="feedforward_intermediate_dense",
        )
        self._feedforward_intermediate_dense.build(inputs_shape)

        self._feedforward_gate_dense = keras.layers.Dense(
            self.intermediate_dim,
            kernel_initializer=self.kernel_initializer,
            use_bias=False,
            dtype=self.dtype_policy,
            name="feedforward_intermediate_dense",
        )
        self._feedforward_gate_dense.build(inputs_shape)

        self._feedforward_output_dense = keras.layers.Dense(
            hidden_dim,
            kernel_initializer=self.kernel_initializer,
            use_bias=False,
            dtype=self.dtype_policy,
            name="feedforward_output_dense",
        )

        intermediate_shape = list(inputs_shape)
        intermediate_shape[-1] = self.intermediate_dim
        self._feedforward_output_dense.build(tuple(intermediate_shape))
        self._feedforward_dropout = keras.layers.Dropout(
            rate=self.dropout,
            dtype=self.dtype_policy,
            name="feedforward_dropout",
        )

    def call(self, x, training=None):
        residual = x
        x = self._feedforward_layer_norm(x)
        key = self._feedforward_intermediate_dense(x)
        gate = self._feedforward_gate_dense(x)

        x = self._feedforward_output_dense(fp32_swish(gate) * key)
        x = self._feedforward_dropout(x, training=training)
        x = ops.cast(x, residual.dtype) + residual
        return x


class TransfomrmerBlock(keras.layers.Layer):
    """预归一化的因果 Transformer 块，与 RWKV 块交替堆叠。

    Args:
        intermediate_dim: int。FFN 中间维度。
        num_heads: int。注意力头数。
        dropout: float。dropout 概率。
        layer_norm_epsilon: float。RMSNorm 的 epsilon。
        kernel_initializer: 权重初始化器。
        **kwargs: 其他层参数。

    Call arguments:
        inputs: [B, T, C], float。
        mask: [B, T], bool。True 表示该位置有效。
        training: bool 或 None。
        xx: [B, T, C], float 或 None。提供时用作注意力分支的输入。

    Returns:
        [B, T, C]，与输入同 dtype。
    """

    def __init__(
        self,
        intermediate_dim,
        num_heads,
        dropout=0,
        layer_norm_epsilon=1e-05,
        kernel_initializer="glorot_uniform",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.intermediate_dim = intermediate_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.layer_norm_epsilon = layer_norm_epsilon
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.supports_masking = True
        self.autocast = False

    def build(self, inputs_shape):
        hidden_dim = inputs_shape[-1]
        key_dim = int(hidden_dim // self.num_heads)
        self.head_size = hidden_dim // self.num_heads

        self._self_attention_layer = GateAttention(
            num_query_heads=self.num_heads,
            num_key_value_heads=self.num_heads,
            head_dim=key_dim,
            dropout=self.dropout,
            kernel_initializer=self.kernel_initializer,
            dtype=self.dtype_policy,
            name="self_attention_layer",
        )

        self._self_attention_layer.build(inputs_shape)

        self._self_attention_norm = keras.layers.RMSNormalization(
            epsilon=self.layer_norm_epsilon,
            dtype=self.dtype_policy,
            name="self_attention_layer_norm",
        )
        self._self_attention_norm.build(inputs_shape)

        self.ffn = FFN(self.intermediate_dim, self.dropout, self.kernel_initializer)
        self.ffn.build(inputs_shape)
        self.built = True

    def call(
        self,
        inputs,
        mask,
        training=None,
        xx=None,
    ):
        T = ops.shape(inputs)[1]
        attention_mask = mask[..., None] * mask[:, None]
        causal_mask = ops.arange(T)[None] <= ops.arange(T)[:, None]
        attention_mask = ops.logical_and(attention_mask, causal_mask[None])
        if xx is not None:
            x = self._self_attention_norm(xx)
        else:
            x = self._self_attention_norm(inputs)
        x = (
            self._self_attention_layer(
                hidden_states=x,
                attention_mask=attention_mask,
            )
            + inputs
        )
        x = self.ffn(x)

        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "intermediate_dim": self.intermediate_dim,
                "num_heads": self.num_heads,
                "dropout": self.dropout,
                "layer_norm_epsilon": self.layer_norm_epsilon,
                "kernel_initializer": keras.initializers.serialize(
                    self.kernel_initializer
                ),
            }
        )
        return config

    def compute_output_shape(self, inputs_shape):
        if isinstance(inputs_shape, list):
            inputs_shape = inputs_shape[0]
        return inputs_shape
