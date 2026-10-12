"""基于 TensorFlow 的时间序列因果标准化工具。"""

import tensorflow as tf


def compute_std_mean_by_mask(x, mask):
    """在掩码有效位置统计均值与标准差。

    Args:
        x: [T], float32。原始序列，可含 NaN。
        mask: [T], bool。True 表示该位置参与统计。

    Returns:
        mean: [], float32。有效点均值。
        std: [], float32。有效点标准差，下限为 1e-6。
    """
    count = tf.reduce_sum(tf.cast(mask, tf.float32))
    count = tf.maximum(count, 1.0)

    masked_x = tf.where(mask, x, 0.0)
    sum_ = tf.reduce_sum(masked_x)
    mean = sum_ / count

    eps = 1e-6
    var_sum = tf.reduce_sum(tf.where(mask, tf.square(x - mean), 0.0))
    # 以 count - 1 作分母（Bessel 校正）
    var = var_sum / tf.maximum(count - 1.0, 1.0)
    std = tf.maximum(tf.sqrt(var), eps)
    return mean, std


def stand_data(x, mean, std):
    """按均值方差做 z-score 标准化。

    Args:
        x: [T], float32。
        mean: [], float32。
        std: [], float32。

    Returns:
        [T], float32，标准化结果。
    """
    return (x - mean) / std


def soft_clip(z, z_clip=4, scale=0.25):
    """对超出阈值的 z-score 做对数尺度压缩，避免离群点主导统计量。

    Args:
        z: [T], float32，原始 z-score。
        z_clip: float，触发压缩的绝对值阈值。
        scale: float，对数压缩强度。

    Returns:
        [T], float32，压缩后的 z-score。
    """
    clip_mask = tf.greater(tf.abs(z), z_clip)
    z_clipped = tf.sign(z) * (z_clip + scale * tf.math.log(1 + tf.abs(z) - z_clip))
    return tf.where(clip_mask, z_clipped, z)


def causal_standardize(
    x,
    history_vaild_num_threshold=10,
    z_score_threshold=4,
    std_threshold=0.01,
    history_nan_rate_threshold=0.5,
    soft_clip_scale=0.25,
    filter_data=True,
    mask=None,
):
    """对单条序列做因果标准化，并按有效性阈值决定是否保留。

    统计量只在历史有效点上计算，因此不引入未来信息。

    Args:
        x: [T], float32。输入序列，可含 NaN。
        history_vaild_num_threshold: int。有效点数量下限。
        z_score_threshold: float。软截断阈值。
        std_threshold: float。标准差下限，低于该值判定为常数序列。
        history_nan_rate_threshold: float。有效点占比下限。
        soft_clip_scale: float。软截断强度。
        filter_data: bool。是否启用有效性过滤。
        mask: [T], bool 或 None。为 None 时由非 NaN 位置推断。

    Returns:
        x_std: [T], float32。标准化序列；判定为无效时整条填充 NaN。
        mask: [T], bool。更新后的有效掩码。
        mean: [], float32。
        std: [], float32。
    """
    seq_len = tf.shape(x)[0]

    if mask is None:
        mask = tf.logical_not(tf.math.is_nan(x))

    mean, std = compute_std_mean_by_mask(x, mask)

    if filter_data:
        vaild_data_num = tf.reduce_sum(tf.cast(mask, "int32"))
        vaild_data_num = tf.cast(vaild_data_num, "float32")

        # 有效点数量足够多，且占比足够高
        nan_vaild_cond = tf.logical_and(
            tf.greater(vaild_data_num, history_vaild_num_threshold),
            tf.greater(
                vaild_data_num,
                tf.cast(seq_len, "float32") * history_nan_rate_threshold,
            ),
        )

        # 均值非零且标准差大于阈值，排除死数据与直线
        statistic_cond = tf.logical_and(
            tf.not_equal(mean, 0.0), tf.greater(std, std_threshold)
        )

        should_keep = tf.logical_and(nan_vaild_cond, statistic_cond)
        mask = tf.logical_and(mask, should_keep)

        def _keep():
            return soft_clip(
                stand_data(x, mean, std), z_score_threshold, soft_clip_scale
            )

        def _drop():
            return tf.fill(tf.shape(x), float("nan"))

        x_std = tf.cond(should_keep, _keep, _drop)
    else:
        x_std = soft_clip(stand_data(x, mean, std), z_score_threshold, soft_clip_scale)

    return x_std, mask, mean, std
