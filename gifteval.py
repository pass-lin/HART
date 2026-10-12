"""RWKV-Time 在 GIFT-Eval 上的零样本评估脚本。"""

# ruff: noqa: E402
# 后端与可见设备环境变量必须在导入 keras / tensorflow 之前设置，
# 因此本文件的 import 语句无法全部集中在文件顶部。

import os

# 后端与可见设备必须在导入 keras / tensorflow 之前设置
os.environ["KERAS_BACKEND"] = "torch"
os.environ["KERNEL_TYPE"] = "cuda"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".95"
os.environ["XLA_FLAGS"] = "--xla_gpu_autotune_level=4"
# GIFT-Eval 数据集根目录，可通过环境变量覆盖
os.environ.setdefault("GIFT_EVAL", "gift_eval")

import argparse
import gc
import json
import logging
import sys
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from tqdm import trange

parser = argparse.ArgumentParser(description="GIFT-Eval 评估")
parser.add_argument("--cuda_devices", default="0", type=str)
parser.add_argument("--model_path", default="weights/Hart-36M", type=str)
parser.add_argument("--max_length", default=2048, type=int)
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import keras
import tensorflow as tf
from dotenv import load_dotenv
from gluonts.ev.metrics import (
    MAE,
    MAPE,
    MASE,
    MSIS,
    MSE,
    ND,
    NRMSE,
    RMSE,
    SMAPE,
    MeanWeightedSumQuantileLoss,
)
from gluonts.model import Forecast, evaluate_model
from gluonts.model.forecast import QuantileForecast
from gluonts.time_feature import get_seasonality

from gift_eval.data import Dataset
from model.RWKV_Time.rwkv_time_backbone import RWKVTimeBackbone
from model.RWKV_Time.rwkv_time_causal_lm_preprocessor import (
    RWKV_time_CausalLMPreprocessor,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("RWKV-Time-GIFT-Eval")

# 模型加载
model_weights: str = args.model_path
model_config_path: str = os.path.join(model_weights, "config.json")
weights_path: str = os.path.join(model_weights, "model.weights.h5")

if not os.path.exists(model_config_path):
    raise FileNotFoundError(f"模型配置文件不存在: {model_config_path}")
if not os.path.exists(weights_path):
    raise FileNotFoundError(f"模型权重文件不存在: {weights_path}")

with open(model_config_path, "r") as f:
    model_config: Dict[str, Any] = json.load(f)
logger.info(f"加载模型配置: {model_config}")

patch_size: int = model_config["patch_size"]
hidden_size: int = model_config["hidden_size"]
n_quantiles: int = model_config["n_quantiles"]
args.max_length *= patch_size
logger.info(
    f"patch_size: {patch_size}, hidden_size: {hidden_size}, n_quantiles: {n_quantiles}"
)

# 全局只加载一次模型，供所有预测器复用
logger.info("开始加载 RWKV-Time 模型...")
backbone: RWKVTimeBackbone = RWKVTimeBackbone(**model_config)
backbone.summary()
backbone.compile()
backbone.load_weights(weights_path, True)
logger.info("模型加载完成")


def predict_warper(fn):
    """在 PyTorch 后端下用 no_grad 包裹推理函数并释放显存。

    Args:
        fn: 推理函数。

    Returns:
        包装后的函数；非 PyTorch 后端直接返回原函数。
    """

    def wrapper(*args, **kwargs):
        if keras.config.backend() == "torch":
            import torch

            with torch.no_grad():
                output = fn(*args, **kwargs)
                torch.cuda.empty_cache()
                return output

        return fn(*args, **kwargs)

    return wrapper


class RWKVTimePredictor:
    """把 RWKV-Time 骨干包装成 GluonTS 预测器。

    Args:
        model_path: str。模型目录。
        prediction_length: int。预测步长。
        context_length: int。上下文 patch 数。
        quantiles_index: int。用作 mean 代理的分位数下标。
        batch_size: int。预测批大小。
    """

    def __init__(
        self,
        model_path: str,
        prediction_length: int,
        context_length: int = 128,
        quantiles_index: int = 4,
        batch_size: int = 2048,
    ) -> None:
        self.model_path = model_path
        self.prediction_length = prediction_length
        self.context_length = context_length
        self.quantiles_index = quantiles_index
        self.batch_size = batch_size

        self.num_tokens: int = (prediction_length + patch_size - 1) // patch_size

        self.preprocessor: RWKV_time_CausalLMPreprocessor = (
            RWKV_time_CausalLMPreprocessor(
                patch_size=patch_size,
                predict_token_num=self.num_tokens,
                sequence_length=self.context_length,
            )
        )

    @predict_warper
    def predict(self, test_data_input: List[Dict[str, Any]]) -> List[Forecast]:
        """对测试窗口批量预测，返回 GluonTS 分位数预测。

        Args:
            test_data_input: GluonTS 测试样本列表，每个样本含 target 与 start。

        Returns:
            与输入等长的 QuantileForecast 列表。
        """
        forecasts: List[Forecast] = []
        max_predict_data = self.batch_size
        test_data_input = [t for t in test_data_input]
        n_batches = len(test_data_input) // max_predict_data + int(
            len(test_data_input) % max_predict_data != 0
        )
        for i in trange(0, n_batches):
            batch_data = test_data_input[
                i * max_predict_data : (i + 1) * max_predict_data
            ]
            inputs = [item["target"] for item in batch_data]

            # 预处理器按当前预测步长选择待预测 patch 数
            self.preprocessor.predict_token_num = self.num_tokens
            if keras.config.backend() == "torch":
                length = max(len(item) for item in inputs)
                self.preprocessor.sequence_length = length // patch_size + 1

            model_inputs, mean, std = self.preprocessor.generate_preprocess(
                [t[-(patch_size * self.context_length) :] for t in inputs],
                sequence_length=self.num_tokens,
            )
            if keras.config.backend() == "jax":
                padding_size = (
                    self.batch_size - len(inputs) % self.batch_size
                ) % self.batch_size
                for k, v in model_inputs.items():
                    with tf.device("cpu"):
                        model_inputs[k] = tf.pad(v, [[0, padding_size], [0, 0], [0, 0]])

            model_outputs = backbone.predict(
                model_inputs, batch_size=self.batch_size, verbose=3
            )
            model_outputs = model_outputs[: len(inputs)]

            pred = np.transpose(model_outputs[:, -self.num_tokens :], [0, 1, 3, 2])
            pred = np.reshape(pred, [len(pred), -1, backbone.n_quantiles])
            output_rescale = (pred * std + mean)[:, : self.prediction_length, :]

            def build_output(x):
                x = np.transpose(x)
                mean_values = x[self.quantiles_index, None]
                return np.concatenate([x, mean_values], axis=0)

            for idx, item in enumerate(batch_data):
                forecast_start_date: pd.Period = item["start"] + len(item["target"])
                forecast_arrays = build_output(output_rescale[idx])

                forecast: QuantileForecast = QuantileForecast(
                    forecast_arrays=forecast_arrays,
                    forecast_keys=[str(i / 10) for i in range(1, 10)] + ["mean"],
                    start_date=forecast_start_date,
                    item_id=item.get("item_id"),
                )
                forecasts.append(forecast)

            del model_inputs
            gc.collect()
        return forecasts


# 评估配置
load_dotenv()

# GIFT-Eval 标准的 short 与 medium/long 数据集列表
SHORT_DATASETS: str = "m4_yearly m4_quarterly m4_monthly m4_weekly m4_daily m4_hourly electricity/15T electricity/H electricity/D electricity/W solar/10T solar/H solar/D solar/W hospital covid_deaths us_births/D us_births/M us_births/W saugeenday/D saugeenday/M saugeenday/W temperature_rain_with_missing kdd_cup_2018_with_missing/H kdd_cup_2018_with_missing/D car_parts_with_missing restaurant hierarchical_sales/D hierarchical_sales/W LOOP_SEATTLE/5T LOOP_SEATTLE/H LOOP_SEATTLE/D SZ_TAXI/15T SZ_TAXI/H M_DENSE/H M_DENSE/D ett1/15T ett1/H ett1/D ett1/W ett2/15T ett2/H ett2/D ett2/W jena_weather/10T jena_weather/H jena_weather/D bitbrains_fast_storage/5T bitbrains_fast_storage/H bitbrains_rnd/5T bitbrains_rnd/H bizitobs_application bizitobs_service bizitobs_l2c/5T bizitobs_l2c/H"

MED_LONG_DATASETS: str = "electricity/15T electricity/H solar/10T solar/H kdd_cup_2018_with_missing/H LOOP_SEATTLE/5T LOOP_SEATTLE/H SZ_TAXI/15T M_DENSE/H ett1/15T ett1/H ett2/15T ett2/H jena_weather/10T jena_weather/H bitbrains_fast_storage/5T bitbrains_rnd/5T bizitobs_application bizitobs_service bizitobs_l2c/5T bizitobs_l2c/H"

all_datasets: List[str] = list(set(SHORT_DATASETS.split() + MED_LONG_DATASETS.split()))
logger.info(f"评估数据集总数: {len(all_datasets)}")

dataset_properties_path = os.path.join(SCRIPT_DIR, "dataset_properties.json")
if not os.path.exists(dataset_properties_path):
    raise FileNotFoundError(f"{dataset_properties_path} 不存在")
with open(dataset_properties_path, "r") as f:
    dataset_properties_map: Dict[str, Any] = json.load(f)
logger.info("数据集属性加载完成")

# 与 GIFT-Eval 榜单一致的标准指标
metrics: List[Any] = [
    MSE(forecast_type="mean"),
    MSE(forecast_type=0.5),
    MAE(),
    MASE(),
    MAPE(),
    SMAPE(),
    MSIS(),
    RMSE(),
    NRMSE(),
    ND(),
    MeanWeightedSumQuantileLoss(
        quantile_levels=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    ),
]

# 结果输出目录
model_name: str = os.path.basename(os.path.normpath(model_weights))
output_dir: str = os.path.join(SCRIPT_DIR, "results", "gift_eval")
os.makedirs(output_dir, exist_ok=True)
csv_file_path: str = os.path.join(output_dir, "full_results.csv")

# 数据集名称美观化映射
pretty_names: Dict[str, str] = {
    "saugeenday": "saugeen",
    "temperature_rain_with_missing": "temperature_rain",
    "kdd_cup_2018_with_missing": "kdd_cup_2018",
    "car_parts_with_missing": "car_parts",
}

columns: List[str] = [
    "dataset",
    "model",
    "eval_metrics/MSE[mean]",
    "eval_metrics/MSE[0.5]",
    "eval_metrics/MAE[0.5]",
    "eval_metrics/MASE[0.5]",
    "eval_metrics/MAPE[0.5]",
    "eval_metrics/sMAPE[0.5]",
    "eval_metrics/MSIS",
    "eval_metrics/RMSE[mean]",
    "eval_metrics/NRMSE[mean]",
    "eval_metrics/ND[0.5]",
    "eval_metrics/mean_weighted_sum_quantile_loss",
    "domain",
    "num_variates",
]

all_results: List[Dict[str, Any]] = []

# 遍历所有数据集与周期进行评估
for ds_num, ds_name in enumerate(all_datasets):
    logger.info(f"处理数据集: {ds_name} ({ds_num + 1}/{len(all_datasets)})")

    ds_key: str = ds_name.split("/")[0].lower()
    for term in ["long", "short", "medium"]:
        # medium/long 只在高频长序列数据集上评估
        if term in ["medium", "long"] and ds_name not in MED_LONG_DATASETS.split():
            logger.info(f"跳过 {ds_name} 的 {term} 周期")
            continue

        if "/" in ds_name:
            ds_freq = ds_name.split("/")[1]
            ds_key_processed = pretty_names.get(ds_key, ds_key)
        else:
            ds_key_processed = pretty_names.get(ds_key, ds_key)
            ds_freq = dataset_properties_map[ds_key_processed]["frequency"]

        ds_config: str = f"{ds_key_processed}/{ds_freq}/{term}"

        # 多变量数据集拆分为单变量评估
        temp_dataset = Dataset(name=ds_name, term=term, to_univariate=False)
        is_multivariate = temp_dataset.target_dim > 1
        dataset = Dataset(name=ds_name, term=term, to_univariate=is_multivariate)

        try:
            next(iter(dataset.test_data.label))
        except (IndexError, StopIteration) as e:
            logger.error(f"数据集 {ds_name} 无法获取样本: {e}")
            continue

        predictor: RWKVTimePredictor = RWKVTimePredictor(
            model_path=model_weights,
            prediction_length=dataset.prediction_length,
            # 动态计算上下文长度，确保不超过模型最大容量
            context_length=(args.max_length - dataset.prediction_length) // patch_size
            - 1,
            quantiles_index=4,
            batch_size=32,
        )

        logger.info("开始评估...")
        results: pd.DataFrame = evaluate_model(
            predictor,
            test_data=dataset.test_data,
            metrics=metrics,
            batch_size=4096,
            axis=None,
            mask_invalid_label=True,
            allow_nan_forecast=False,
            seasonality=get_seasonality(dataset.freq),
        )

        all_results.append(
            {
                "dataset": ds_config,
                "model": model_name,
                "eval_metrics/MSE[mean]": float(results["MSE[mean]"].values[0]),
                "eval_metrics/MSE[0.5]": float(results["MSE[0.5]"].values[0]),
                "eval_metrics/MAE[0.5]": float(results["MAE[0.5]"].values[0]),
                "eval_metrics/MASE[0.5]": float(results["MASE[0.5]"].values[0]),
                "eval_metrics/MAPE[0.5]": float(results["MAPE[0.5]"].values[0]),
                "eval_metrics/sMAPE[0.5]": float(results["sMAPE[0.5]"].values[0]),
                "eval_metrics/MSIS": float(results["MSIS"].values[0]),
                "eval_metrics/RMSE[mean]": float(results["RMSE[mean]"].values[0]),
                "eval_metrics/NRMSE[mean]": float(results["NRMSE[mean]"].values[0]),
                "eval_metrics/ND[0.5]": float(results["ND[0.5]"].values[0]),
                "eval_metrics/mean_weighted_sum_quantile_loss": float(
                    results["mean_weighted_sum_quantile_loss"].values[0]
                ),
                "domain": dataset_properties_map[ds_key_processed]["domain"],
                "num_variates": dataset_properties_map[ds_key_processed][
                    "num_variates"
                ],
            }
        )
        logger.info(
            f"数据集 {ds_config} 评估完成，MASE={float(results['MASE[0.5]'].values[0]):.4f}, "
            f"CRPS={float(results['mean_weighted_sum_quantile_loss'].values[0]):.4f}"
        )

        del dataset, temp_dataset, predictor, results
        gc.collect()

# 保存结果并统计平均值
logger.info("所有评估完成，正在保存结果...")
df_results: pd.DataFrame = pd.DataFrame(all_results, columns=columns)
df_results.to_csv(csv_file_path, index=False)
logger.info(f"结果文件: {csv_file_path}")
logger.info(f"数据集-周期组合总数: {len(all_results)}")

numeric_columns = [col for col in columns if col.startswith("eval_metrics/")]

average_results = {col: df_results[col].mean() for col in numeric_columns}
df_average = pd.DataFrame([average_results])
average_csv_path = os.path.join(output_dir, "average_results.csv")
df_average.to_csv(average_csv_path, index=False)

logger.info("各指标平均值:")
for metric, value in average_results.items():
    logger.info(f"{metric:<50}: {value:.6f}")
logger.info(f"平均值结果已保存到: {average_csv_path}")

# 按领域分组的平均值
domain_averages = []
for domain in df_results["domain"].unique():
    domain_data = df_results[df_results["domain"] == domain]
    domain_avg = {"domain": domain}
    for col in numeric_columns:
        domain_avg[col] = domain_data[col].mean()
    domain_averages.append(domain_avg)

df_domain_avg = pd.DataFrame(domain_averages)
domain_csv_path = os.path.join(output_dir, "domain_average_results.csv")
df_domain_avg.to_csv(domain_csv_path, index=False)
logger.info(f"按领域分组的结果已保存到: {domain_csv_path}")
