"""RWKV-7 时间序列骨干网络的基础层实现。"""

import keras
from keras import initializers
from keras import ops
from keras.layers import Layer
from rwkv_ops import (
    generalized_delta_rule,
    rwkv7_op_inference,
)

# fp32 精度的激活函数，避免 bfloat16 下的数值误差


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


def fp32_softplus(x):
    """fp32 精度 softplus。

    Args:
        x: 任意形状张量。

    Returns:
        与输入同 dtype 的 softplus 结果。
    """
    dtype = x.dtype
    x = ops.cast(x, "float32")
    return ops.cast(ops.nn.softplus(x), dtype)


def fp32_tanh(x):
    """fp32 精度 tanh。

    Args:
        x: 任意形状张量。

    Returns:
        与输入同 dtype 的 tanh 结果。
    """
    dtype = x.dtype
    x = ops.cast(x, "float32")
    return ops.cast(ops.tanh(x), dtype)


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


class TimeShift(Layer):
    """沿时间维右移一位的移位层（亦称 short conv）。

    Call arguments:
        inputs: [B, T, C], float。输入序列。
        cache_x: [B, 1, C], float 或 None。自回归推理时上一步的输入，
            提供时以它拼接而非零填充。

    Returns:
        [B, T, C]，与 inputs 同 dtype 的移位结果。
    """

    def call(self, inputs, cache_x=None):
        if cache_x is not None:
            x = ops.concatenate([cache_x, inputs], axis=1)
        else:
            x = ops.pad(inputs, [[0, 0], [1, 0], [0, 0]], constant_values=0.0)
        return x[:, :-1, :]

    def compute_output_shape(self, input_shape):
        return input_shape


class RWKV7_ChannelMix(Layer):
    """RWKV-7 通道混合层。

    Args:
        dim_ffn: int。前馈维度。
        kernel_initializer: 权重初始化器。
        **kwargs: 其他层参数。

    Call arguments:
        x: [B, T, C], float。

    Returns:
        [B, T, C]，与输入同 dtype。
    """

    def __init__(self, dim_ffn, kernel_initializer="glorot_uniform", **kwargs):
        super().__init__(**kwargs)
        self.dim_ffn = dim_ffn
        self.kernel_initializer = initializers.get(kernel_initializer)

    def call(self, x):
        xx = self.time_shift(x) - x

        k = x + xx * self.x_k
        k = self.key(k) * fp32_swish(self.gate(k))
        output = self.value(k)
        return output

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, list):
            return input_shape[0]
        return input_shape

    def build(self, input_shape):
        super().build(input_shape)
        if isinstance(input_shape, list):
            input_shape = input_shape[0]

        self.x_k = self.add_weight(
            shape=(input_shape[-1],),
            name="time_mix_k",
            initializer=self.kernel_initializer,
        )
        self.time_shift = TimeShift(dtype=self.dtype_policy)
        self.key = keras.layers.Dense(
            self.dim_ffn,
            use_bias=False,
            name="dense_k",
            kernel_initializer=self.kernel_initializer,
            dtype=self.dtype_policy,
        )
        self.gate = keras.layers.Dense(
            self.dim_ffn,
            use_bias=False,
            name="dense_g",
            kernel_initializer=self.kernel_initializer,
            dtype=self.dtype_policy,
        )
        self.value = keras.layers.Dense(
            input_shape[-1],
            use_bias=False,
            name="dense_v",
            kernel_initializer=self.kernel_initializer,
            dtype=self.dtype_policy,
        )
        self.key.build(input_shape)
        self.gate.build(input_shape)
        self.value.build([None, None, self.dim_ffn])

    def get_config(self):
        config = {
            "dim_ffn": self.dim_ffn,
            "kernel_initializer": initializers.serialize(self.kernel_initializer),
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


class LoRADense(Layer):
    """RWKV-7 使用的低秩全连接层，计算 OuterAct(Bias + InnerAct(x @ W1) @ W2)。

    Args:
        units: int。输出维度。
        lora_dim: int。LoRA 隐藏维度。
        inner_activation: callable 或 None。W1 之后的激活。
        outer_activation: callable 或 None。W2 之后的激活。
        use_bias: bool。是否叠加可学习的偏置向量。
        kernel_initializer: 权重初始化器。
        **kwargs: 其他层参数。

    Call arguments:
        x: [B, T, C_in], float。

    Returns:
        [B, T, units]，与输入同 dtype。
    """

    def __init__(
        self,
        units,
        lora_dim,
        inner_activation=None,
        outer_activation=None,
        use_bias=True,
        kernel_initializer="glorot_uniform",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.units = units
        self.lora_dim = lora_dim
        self.inner_activation = inner_activation
        self.outer_activation = outer_activation
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)

    def build(self, input_shape):
        input_dim = input_shape[-1]

        # W1：下投影
        self.lora_down = self.add_weight(
            shape=(input_dim, self.lora_dim),
            name="lora_down",
            initializer=self.kernel_initializer,
        )

        # W2：上投影
        self.lora_up = self.add_weight(
            shape=(self.lora_dim, self.units),
            name="lora_up",
            initializer=self.kernel_initializer,
        )

        if self.use_bias:
            self.bias = self.add_weight(
                shape=(self.units,),
                name="bias",
                initializer=self.kernel_initializer,
            )

        super().build(input_shape)

    def call(self, x):
        x = ops.matmul(x, self.lora_down)
        if self.inner_activation is not None:
            x = self.inner_activation(x)

        x = ops.matmul(x, self.lora_up)
        if self.use_bias:
            x = x + self.bias
        if self.outer_activation is not None:
            x = self.outer_activation(x)

        return x

    def get_config(self):
        config = {
            "units": self.units,
            "lora_dim": self.lora_dim,
            "use_bias": self.use_bias,
            "kernel_initializer": initializers.serialize(self.kernel_initializer),
        }
        return {**super().get_config(), **config}


class RWKV7_TimeMix(Layer):
    """RWKV-7 时间混合层。

    Args:
        hidden_size: int。隐藏维度 C。
        head_size: int。单头维度 N。
        gate_lora: int。门控 LoRA 维度。
        mv_lora: int。value 混合 LoRA 维度。
        aaa_lora: int。alpha LoRA 维度。
        decay_lora: int。衰减 LoRA 维度。
        kernel_initializer: 权重初始化器。
        add_v_first: bool。是否用 v_first 与当前 v 做门控混合。
        **kwargs: 其他层参数。

    Call arguments:
        x: [B, T, C], float。
        v_first: [B, T, C], float 或 None。前一层输出的 value 参考。
        padding_mask: [B, T, 1], float 或 None。0 表示 padding。
        training: bool 或 None。True 时走可微分的 chunkwise 实现。

    Returns:
        x: [B, T, C]，与输入同 dtype。
        v_first: [B, T, C]。供下一层混合的 value。
        final_state: [B, H, N, N], float32。序列末端状态。
    """

    def __init__(
        self,
        hidden_size,
        head_size,
        gate_lora=128,
        mv_lora=32,
        aaa_lora=64,
        decay_lora=64,
        kernel_initializer="glorot_uniform",
        add_v_first=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.head_size = head_size
        self.hidden_size = hidden_size
        self.n_head = hidden_size // self.head_size
        self.gate_lora = gate_lora
        self.mv_lora = mv_lora
        self.aaa_lora = aaa_lora
        self.decay_lora = decay_lora
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.add_v_first = add_v_first
        self.initial_state = None

        assert self.hidden_size % self.n_head == 0

    def build(self, input_shape):
        super().build(input_shape)
        if isinstance(input_shape[0], list):
            input_shape = input_shape[0]
        H = self.n_head
        N = self.head_size
        B, T, C = input_shape

        # 时间混合因子
        self.x_r = self.add_weight(
            shape=(C,), name="x_r", initializer=self.kernel_initializer
        )
        self.x_w = self.add_weight(
            shape=(C,), name="x_w", initializer=self.kernel_initializer
        )
        self.x_k = self.add_weight(
            shape=(C,), name="x_k", initializer=self.kernel_initializer
        )
        self.x_v = self.add_weight(
            shape=(C,), name="x_v", initializer=self.kernel_initializer
        )
        self.x_a = self.add_weight(
            shape=(C,), name="x_a", initializer=self.kernel_initializer
        )
        self.x_g = self.add_weight(
            shape=(C,), name="x_g", initializer=self.kernel_initializer
        )

        # 衰减 w = -softplus(-(w0 + tanh(xw @ w1) @ w2)) - 0.5，外层变换在 call 中完成
        self.lora_w = LoRADense(
            units=C,
            lora_dim=self.decay_lora,
            inner_activation=fp32_tanh,
            outer_activation=None,
            use_bias=True,
            kernel_initializer=self.kernel_initializer,
            name="lora_w",
        )

        # alpha a = sigmoid(a0 + (xa @ a1) @ a2)
        self.lora_a = LoRADense(
            units=C,
            lora_dim=self.aaa_lora,
            inner_activation=None,
            outer_activation=fp32_sigmoid,
            use_bias=True,
            kernel_initializer=self.kernel_initializer,
            name="lora_a",
        )

        # value 混合 v = sigmoid(v0 + (xv @ v1) @ v2)
        if self.add_v_first:
            self.lora_v = LoRADense(
                units=C,
                lora_dim=self.mv_lora,
                inner_activation=None,
                outer_activation=fp32_sigmoid,
                use_bias=True,
                kernel_initializer=self.kernel_initializer,
                name="lora_v",
            )

        # 门控 g = sigmoid(xg @ g1) @ g2，无偏置且 sigmoid 位于中间
        self.lora_g = LoRADense(
            units=C,
            lora_dim=self.gate_lora,
            inner_activation=fp32_sigmoid,
            outer_activation=None,
            use_bias=False,
            kernel_initializer=self.kernel_initializer,
            name="lora_g",
        )

        self.k_k = self.add_weight(
            shape=(C,), name="k_k", initializer=self.kernel_initializer
        )
        self.k_a = self.add_weight(
            shape=(C,), name="k_a", initializer=self.kernel_initializer
        )
        self.r_k = self.add_weight(
            shape=(H, N), name="r_k", initializer=self.kernel_initializer
        )

        self.time_shift = TimeShift(dtype=self.dtype_policy)
        self.receptance = keras.layers.Dense(
            C,
            use_bias=False,
            kernel_initializer=self.kernel_initializer,
            name="receptance",
            dtype=self.dtype_policy,
        )
        self.key = keras.layers.Dense(
            C,
            use_bias=False,
            kernel_initializer=self.kernel_initializer,
            name="key",
            dtype=self.dtype_policy,
        )
        self.value = keras.layers.Dense(
            C,
            use_bias=False,
            kernel_initializer=self.kernel_initializer,
            name="value",
            dtype=self.dtype_policy,
        )
        self.output_layer = keras.layers.Dense(
            C,
            use_bias=False,
            kernel_initializer=self.kernel_initializer,
            name="output_layer",
            dtype=self.dtype_policy,
        )

        self.ln_x = keras.layers.GroupNormalization(
            groups=H, epsilon=64e-5, dtype="float32"
        )

        self.lora_w.build((None, None, C))
        self.lora_a.build((None, None, C))
        self.lora_g.build((None, None, C))
        if self.add_v_first:
            self.lora_v.build((None, None, C))

        self.receptance.build(input_shape)
        self.value.build(input_shape)
        self.key.build(input_shape)
        self.output_layer.build(input_shape)
        self.ln_x.build((None, C))

    def call(self, x, v_first=None, padding_mask=None, training=None):
        if padding_mask is not None:
            if ops.ndim(padding_mask) == 2:
                padding_mask = padding_mask[..., None]
            padding_mask = ops.cast(padding_mask, x.dtype)
            x *= padding_mask

        B, T, C = ops.shape(x)
        H = self.n_head

        xx = self.time_shift(x) - x
        if padding_mask is not None:
            xx *= padding_mask

        xr = x + xx * self.x_r
        xw = x + xx * self.x_w
        xk = x + xx * self.x_k
        xv = x + xx * self.x_v
        xa = x + xx * self.x_a
        xg = x + xx * self.x_g

        r = self.receptance(xr)

        # 衰减 w = -softplus(-(w0 + tanh(xw @ w1) @ w2)) - 0.5
        w_raw = self.lora_w(xw)
        w = -fp32_softplus(-w_raw) - 0.5

        k = self.key(xk)
        v = self.value(xv)

        if v_first is None or not self.add_v_first:
            v_first = v
        else:
            v_mix_factor = self.lora_v(xv)
            v = v + (v_first - v) * v_mix_factor

        a = self.lora_a(xa)
        g = self.lora_g(xg)

        kk = k * self.k_k
        kk = self.normalize(ops.reshape(kk, (B, T, H, -1)))
        kk = ops.reshape(kk, (B, T, C))

        k = k * (1 + (a - 1) * self.k_a)

        if padding_mask is not None:
            w = ops.where(padding_mask, w, -1e9)
        if training:
            rwkv7_op = generalized_delta_rule
        else:
            rwkv7_op = rwkv7_op_inference

        x, final_state = rwkv7_op(
            ops.reshape(r, (B, T, self.n_head, self.head_size)),
            ops.reshape(w, (B, T, self.n_head, self.head_size)),
            ops.reshape(k, (B, T, self.n_head, self.head_size)),
            ops.reshape(v, (B, T, self.n_head, self.head_size)),
            ops.reshape(-kk, (B, T, self.n_head, self.head_size)),
            ops.reshape(kk * a, (B, T, self.n_head, self.head_size)),
            initial_state=None,
        )
        x = ops.reshape(x, (B, T, C))
        x = ops.reshape(self.ln_x(ops.reshape(x, (B * T, C))), ops.shape(x))
        x = ops.reshape(x, (B, T, C))

        r = ops.reshape(r, (B, T, H, -1))
        k = ops.reshape(k, (B, T, H, -1))
        v = ops.reshape(v, (B, T, C))

        rwkv = ops.sum(r * k * self.r_k, axis=-1, keepdims=True) * ops.reshape(
            v, (B, T, H, -1)
        )

        x = ops.cast(x, rwkv.dtype) + ops.reshape(rwkv, (B, T, C))
        x = self.output_layer(x * g)

        return x, v_first, final_state

    def compute_output_shape(self, input_shape):
        output_shapes = [
            [None, None, self.hidden_size],
            [None, None, self.hidden_size],
            [None, self.n_head, self.head_size, self.head_size],
        ]
        return output_shapes

    def normalize(self, x, eps=1e-12):
        """对最后一维做归一化，并在 fp32 下计算。

        Args:
            x: [..., N], float。
            eps: float。数值下界，避免除零。

        Returns:
            与输入同 dtype 的归一化结果。
        """
        dtype = x.dtype
        x = ops.cast(x, "float32")
        square_sum = ops.sum(ops.square(x), axis=-1, keepdims=True)
        inv_norm = ops.rsqrt(square_sum + eps)
        inv_norm = ops.maximum(inv_norm, eps)
        return ops.cast(x * inv_norm, dtype)

    def get_config(self):
        config = {
            "hidden_size": self.hidden_size,
            "head_size": self.head_size,
            "gate_lora": self.gate_lora,
            "mv_lora": self.mv_lora,
            "aaa_lora": self.aaa_lora,
            "decay_lora": self.decay_lora,
            "add_v_first": self.add_v_first,
            "kernel_initializer": initializers.serialize(self.kernel_initializer),
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Time_RWKV7_Block(Layer):
    """RWKV-7 时间混合与通道混合的预归一化残差块。

    Args:
        hidden_size: int。隐藏维度 C。
        head_size: int。单头维度。
        intermediate_dim: int。前馈维度。
        gate_lora: int。门控 LoRA 维度。
        mv_lora: int。value 混合 LoRA 维度。
        aaa_lora: int。alpha LoRA 维度。
        decay_lora: int。衰减 LoRA 维度。
        first_block: bool。首层额外做输入 LayerNorm，并关闭 value 混合。
        kernel_initializer: 权重初始化器。
        **kwargs: 其他层参数。

    Call arguments:
        x: [B, T, C], float。
        v_first: [B, T, C], float 或 None。前一层 value 参考。
        padding_mask: [B, T], float 或 None。0 表示 padding。
        state: 时间混合层的末端状态，或 None。

    Returns:
        x: [B, T, C]。
        v_first: [B, T, C]。
        state: 时间混合层返回的末端状态。
    """

    def __init__(
        self,
        hidden_size,
        head_size,
        intermediate_dim,
        gate_lora=128,
        mv_lora=32,
        aaa_lora=64,
        decay_lora=64,
        first_block=False,
        kernel_initializer="glorot_uniform",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.head_size = head_size
        self.hidden_size = hidden_size
        self.intermediate_dim = intermediate_dim
        self.gate_lora = gate_lora
        self.mv_lora = mv_lora
        self.aaa_lora = aaa_lora
        self.decay_lora = decay_lora
        self.first_block = first_block
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.autocast = False

    def build(self, input_shape):
        super().build(input_shape)

        C = self.hidden_size
        if self.first_block:
            self.ln0 = keras.layers.LayerNormalization(
                epsilon=1e-5, dtype=self.dtype_policy, name="init_norm"
            )
            self.ln0.build((None, None, C))

        self.ln1 = keras.layers.LayerNormalization(
            epsilon=1e-5, dtype=self.dtype_policy, name="att_norm"
        )
        self.ln1.build((None, None, C))

        self.ln2 = keras.layers.LayerNormalization(
            epsilon=1e-5, dtype=self.dtype_policy, name="ffn_norm"
        )
        self.ln2.build((None, None, C))

        self.att = RWKV7_TimeMix(
            self.hidden_size,
            self.head_size,
            self.gate_lora,
            self.mv_lora,
            self.aaa_lora,
            self.decay_lora,
            name="RWKV_TIME_MIX",
            add_v_first=not self.first_block,
            kernel_initializer=self.kernel_initializer,
            dtype=self.dtype_policy,
        )
        self.att.build((None, None, C))

        self.ffn = RWKV7_ChannelMix(
            self.intermediate_dim,
            name="RWKV_CMIX",
            kernel_initializer=self.kernel_initializer,
            dtype=self.dtype_policy,
        )
        self.ffn.build((None, None, C))

    def call(self, x, v_first=None, padding_mask=None, state=None):
        if padding_mask is not None:
            padding_mask = ops.cast(padding_mask, x.dtype)
            padding_mask = ops.expand_dims(padding_mask, axis=-1)
        if self.first_block:
            x = self.ln0(x)
        xx = self.ln1(x)
        # 返回末端状态供递归使用；本模型推理不消费该状态
        xx, v_first, state = self.att(
            xx,
            v_first=v_first,
            padding_mask=padding_mask,
        )
        x = ops.cast(x, xx.dtype) + xx
        xx = self.ln2(x)
        if padding_mask is not None:
            xx = xx * padding_mask
        x = x + self.ffn(xx)
        return x, v_first, state

    def compute_output_shape(self, input_shape):
        x_shape = [None, None, self.hidden_size]
        return [
            x_shape,
            [None, None, self.hidden_size],
            [
                None,
                self.hidden_size // self.head_size,
                self.head_size,
                self.head_size,
            ],
        ]

    def get_config(self):
        config = {
            "hidden_size": self.hidden_size,
            "head_size": self.head_size,
            "intermediate_dim": self.intermediate_dim,
            "gate_lora": self.gate_lora,
            "mv_lora": self.mv_lora,
            "aaa_lora": self.aaa_lora,
            "decay_lora": self.decay_lora,
            "first_block": self.first_block,
            "kernel_initializer": initializers.serialize(self.kernel_initializer),
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


class TimeSeriesOutHead(Layer):
    """分位数输出头，由残差 MLP 直接回归每个 patch 的分位数。

    Args:
        n_quantiles: int。分位数个数。
        out_patch_size: int。输出 patch 长度，等于输入 patch 长度。
        hidden_dim: int。输入隐藏维度。
        proj_h_dim: int。MLP 中间维度。
        kernel_initializer: 权重初始化器。
        **kwargs: 其他层参数。

    Call arguments:
        x: [B, T, C], float。

    Returns:
        [B, T, n_quantiles, out_patch_size]，与输入同 dtype。
    """

    def __init__(
        self,
        n_quantiles=9,
        out_patch_size=32,
        hidden_dim=512,
        proj_h_dim=1024,
        kernel_initializer="glorot_uniform",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.n_quantiles = n_quantiles
        self.out_patch_size = out_patch_size
        self.hidden_dim = hidden_dim
        self.proj_h_dim = proj_h_dim
        self.kernel_initializer = initializers.get(kernel_initializer)

    def build(self, input_shape):
        self.fc1 = keras.layers.Dense(
            self.proj_h_dim,
            activation="silu",
            use_bias=False,
            kernel_initializer=self.kernel_initializer,
            name="out_fc1",
        )
        self.fc2 = keras.layers.Dense(
            self.n_quantiles * self.out_patch_size,
            use_bias=False,
            kernel_initializer=self.kernel_initializer,
            name="out_fc2",
        )

        self.skip = keras.layers.Dense(
            self.n_quantiles * self.out_patch_size,
            use_bias=False,
            kernel_initializer=self.kernel_initializer,
            name="out_skip",
        )
        self.skip.build([None, None, self.hidden_dim])

        self.fc1.build([None, None, self.hidden_dim])
        self.fc2.build([None, None, self.proj_h_dim])
        super().build(input_shape)

    def call(self, x):
        h = self.fc1(x)
        out = self.fc2(h)
        out = out + self.skip(x)

        out = ops.reshape(out, (x.shape[0], -1, self.n_quantiles, self.out_patch_size))
        return out

    def compute_output_shape(self, _):
        return (None, None, self.n_quantiles, self.out_patch_size)

    def get_config(self):
        return {
            **super().get_config(),
            "n_quantiles": self.n_quantiles,
            "out_patch_size": self.out_patch_size,
            "hidden_dim": self.hidden_dim,
            "proj_h_dim": self.proj_h_dim,
            "kernel_initializer": initializers.serialize(self.kernel_initializer),
        }


class TimeSeriesPatchEmbedding(Layer):
    """时间序列分块嵌入：patch 值拼接观测掩码后投影为隐藏向量。

    输入按不重叠、步长为 patch_size 的方式切分，与 Moirai 2.0 一致。

    Args:
        patch_size: int。patch 长度，必填。
        hidden_dim: int。输出隐藏维度。
        proj_h_dim: int。残差分支中间维度。
        kernel_initializer: 权重初始化器。
        **kwargs: 其他层参数。

    Call arguments:
        patches: [B, n_patch, patch_size], float。
        obs_mask: [B, n_patch, patch_size], bool。有效观测掩码。
        nan_mask: [B, n_patch, patch_size], bool。由 NaN 位置构造的掩码。

    Returns:
        z: [B, n_patch, hidden_dim]，与 patches 同 dtype。
        patch_valid: [B, n_patch], bool。该 patch 是否含有效观测。
    """

    def __init__(
        self,
        patch_size,
        hidden_dim=768,
        proj_h_dim=768 * 4,
        kernel_initializer="glorot_uniform",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.proj_h_dim = proj_h_dim
        self.kernel_initializer = initializers.get(kernel_initializer)

    def build(self, input_shape):
        self.proj_in = keras.layers.Dense(
            self.proj_h_dim,
            use_bias=False,
            kernel_initializer=self.kernel_initializer,
            name="patch_proj_in",
            dtype=self.dtype_policy,
        )
        self.proj_main = keras.layers.Dense(
            self.hidden_dim,
            use_bias=False,
            kernel_initializer=self.kernel_initializer,
            name="patch_proj_main",
            dtype=self.dtype_policy,
        )

        self.proj_skip = keras.layers.Dense(
            self.hidden_dim,
            use_bias=False,
            kernel_initializer=self.kernel_initializer,
            name="patch_skip",
            dtype=self.dtype_policy,
        )
        self.proj_skip.build([None, None, self.patch_size * 3])

        # 拼接后的通道数：patch 值 + 观测掩码 + 非 NaN 掩码
        common_shape = [None, None, self.patch_size * 3]
        self.proj_in.build(common_shape)
        self.proj_main.build([None, None, self.proj_h_dim])
        super().build(input_shape)

    def call(self, patches, obs_mask, nan_mask):
        patch_valid = ops.any(obs_mask, axis=-1, keepdims=True)

        x_cat = ops.concatenate(
            [
                patches,
                ops.cast(obs_mask, patches.dtype),
                ops.cast(nan_mask, patches.dtype),
            ],
            axis=-1,
        )

        h = ops.silu(self.proj_in(x_cat))
        z = self.proj_main(h)

        z = z + self.proj_skip(x_cat)

        # 无效 patch 清零，避免 padding 参与后续计算
        z = z * ops.cast(patch_valid, z.dtype)

        return z, patch_valid[..., 0]

    def compute_output_shape(self, _):
        return [
            (None, None, self.hidden_dim),
            (None, None),
        ]

    def get_config(self):
        return {
            **super().get_config(),
            "patch_size": self.patch_size,
            "hidden_dim": self.hidden_dim,
            "proj_h_dim": self.proj_h_dim,
            "kernel_initializer": initializers.serialize(self.kernel_initializer),
        }
