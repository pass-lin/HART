"""RWKV-Time 推理期数据预处理：分块与因果标准化。"""

import numpy as np
from keras_hub.src.models.causal_lm_preprocessor import CausalLMPreprocessor
from keras_hub.src.utils.tensor_utils import no_convert_scope
from keras_hub.src.utils.tensor_utils import tf

from .data_processing import causal_standardize

if tf is not None:
    causal_standardize = tf.function(causal_standardize, reduce_retracing=True)


def preprocessing_function(fn):
    """包装预处理函数，使其固定在 CPU 且不触发张量自动转换。

    Args:
        fn: 形如 fn(self, x, **kwargs) 的预处理函数。

    Returns:
        包装后的函数；当 tensorflow 不可用时原样返回 fn。
    """
    if tf is None:
        return fn

    def wrapper(self, x, **kwargs):
        with tf.device("cpu"):
            with no_convert_scope():
                x = fn(self, x, **kwargs)
            return x

    return wrapper


class RWKV_time_CausalLMPreprocessor(CausalLMPreprocessor):
    """把原始时间序列切成 patch 并做因果标准化，供骨干网络推理。

    Args:
        patch_size: int。patch 长度，必须大于 0。
        history_vaild_num_threshold: int。有效点数量下限。
        z_score_threshold: float。软截断阈值。
        std_threshold: float。标准差下限。
        history_nan_rate_threshold: float。有效点占比下限。
        soft_clip_scale: float。软截断强度。
        predict_token_num: int。推理时预测的 patch 数。
        parall_predict_num: int。并行预测的 patch 数。
        **kwargs: 其他层参数，包含 sequence_length。
    """

    def __init__(
        self,
        patch_size=32,
        history_vaild_num_threshold=10,
        z_score_threshold=100,
        std_threshold=0.01,
        history_nan_rate_threshold=0.5,
        soft_clip_scale=1,
        predict_token_num=1,
        parall_predict_num=9,
        **kwargs,
    ):
        super().__init__(tokenizer=None, add_start_token=False, **kwargs)
        assert patch_size > 0
        self.patch_size = int(patch_size)
        self.history_vaild_num_threshold = history_vaild_num_threshold
        self.z_score_threshold = z_score_threshold
        self.std_threshold = std_threshold
        self.history_nan_rate_threshold = history_nan_rate_threshold
        self.soft_clip_scale = soft_clip_scale
        self.predict_token_num = predict_token_num
        self.parall_predict_num = parall_predict_num

    def build(self, input_shape):
        self.built = True

    def padding_to_patch_size(self, x):
        """在序列前端补零，使其长度为 patch_size 的整数倍。

        Args:
            x: [T], float。

        Returns:
            [T'], float。T' 为不小于 T 的 patch_size 倍数。
        """
        padding_size = (
            self.patch_size - tf.shape(x)[0] % self.patch_size
        ) % self.patch_size
        return tf.pad(x, [[padding_size, 0]])

    def patch_reshape_padding(self, x, sequence_length):
        """切分 patch 并裁剪或补零到固定的 patch 数。

        Args:
            x: [T], float。
            sequence_length: int。目标 patch 数。

        Returns:
            [sequence_length, patch_size]，与 x 同 dtype。
        """
        x = self.padding_to_patch_size(x)
        x = tf.reshape(x, [-1, self.patch_size])
        x = x[-sequence_length:]
        padding_size = sequence_length - tf.shape(x)[0]
        return tf.pad(x, [[padding_size, 0], [0, 0]])

    def stand_data_single_series(self, ts, sequence_length):
        """对单条序列做因果标准化并切成 patch。

        Args:
            ts: [T], float。原始序列。
            sequence_length: int。目标 patch 数。

        Returns:
            ts: [sequence_length, patch_size], float32。
            mask: [sequence_length, patch_size], bool。
            mean: [], float32。
            std: [], float32。
        """
        ts = tf.convert_to_tensor(ts, "float32")
        ts = tf.reshape(ts, [-1])
        mask = tf.ones_like(ts, dtype=tf.bool)
        ts, _, mean, std = causal_standardize(
            ts,
            history_vaild_num_threshold=self.history_vaild_num_threshold,
            z_score_threshold=self.z_score_threshold,
            std_threshold=self.std_threshold,
            history_nan_rate_threshold=self.history_nan_rate_threshold,
            soft_clip_scale=self.soft_clip_scale,
            filter_data=False,
        )
        ts = self.patch_reshape_padding(ts, sequence_length)
        mask = self.patch_reshape_padding(mask, sequence_length)

        return ts, mask, mean, std

    def stand_data_batch_series(self, x, sequence_length):
        """对一批序列做因果标准化并切成 patch。

        Args:
            x: 序列列表，每个元素为 [T] 的数组或列表。
            sequence_length: int。目标 patch 数，不能为 None。

        Returns:
            ts: [B, sequence_length, patch_size], float32。
            mask: [B, sequence_length, patch_size], bool。
            nan_mask: [B, sequence_length, patch_size], bool。True 表示该位置为 NaN。
            mean: [B], float32。
            std: [B], float32。

        Raises:
            ValueError: sequence_length 为 None。
        """
        try:
            # 一维输入时包装为单元素 batch
            float(x[0])
            x = [x]
        except (TypeError, ValueError):
            pass

        if sequence_length is None:
            raise ValueError("`sequence_length` must be specified.")

        outputs = [
            self.stand_data_single_series(input_data, sequence_length)
            for input_data in x
        ]
        ts, mask, mean, std = (
            [tf.stack(col, 0) for col in zip(*outputs)] if outputs else [[], [], [], []]
        )

        nan_mask = tf.math.is_nan(ts)
        ts = tf.where(nan_mask, 0.0, ts)
        return ts, mask, nan_mask, mean, std

    @preprocessing_function
    def generate_preprocess(self, x, sequence_length=None):
        """构造骨干网络的推理输入，并在尾部预留待预测 patch。

        Args:
            x: 序列列表，每个元素为 [T] 的历史序列。
            sequence_length: int 或 None。待预测 patch 数，默认取 predict_token_num。

        Returns:
            inputs: dict，含 time_series / padding_mask / nan_mask。
            mean: [B, 1, 1], float32。用于反标准化。
            std: [B, 1, 1], float32。用于反标准化。
        """
        generate_sequence = sequence_length or self.predict_token_num
        generate_sequence -= 1
        sequence_length = self.sequence_length
        ts, mask, nan_mask, mean, std = self.stand_data_batch_series(x, sequence_length)
        bsz = tf.shape(ts)[0]
        return (
            {
                "time_series": tf.concat(
                    [
                        ts,
                        tf.zeros(
                            [bsz, generate_sequence, self.patch_size], dtype=ts.dtype
                        ),
                    ],
                    axis=1,
                ),
                "padding_mask": tf.concat(
                    [
                        mask,
                        tf.ones(
                            [bsz, generate_sequence, self.patch_size], dtype=mask.dtype
                        ),
                    ],
                    axis=1,
                ),
                "nan_mask": tf.concat(
                    [
                        nan_mask,
                        tf.ones(
                            [bsz, generate_sequence, self.patch_size],
                            dtype=nan_mask.dtype,
                        ),
                    ],
                    axis=1,
                ),
            },
            np.array(tf.reshape(mean, [-1, 1, 1])),
            np.array(tf.reshape(std, [-1, 1, 1])),
        )

    def get_config(self):
        config = super().get_config()
        config.update({"patch_size": self.patch_size})
        return config
