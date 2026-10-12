"""RWKV-Time 在 TIME benchmark 上的零样本评估脚本。"""

# ruff: noqa: E402
# 后端与可见设备环境变量必须在导入 keras / tensorflow 之前设置，
# 因此本文件的 import 语句无法全部集中在文件顶部。

import os

os.environ["KERAS_BACKEND"] = "torch"
os.environ["KERNEL_TYPE"] = "cuda"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".95"
os.environ["XLA_FLAGS"] = "--xla_gpu_autotune_level=4"
# TIME benchmark 数据集根目录，可通过环境变量覆盖
os.environ.setdefault("TIME_DATASET", "TIME/data")

import argparse
import gc
import json
import logging
import sys
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from tqdm import trange

parser = argparse.ArgumentParser(description="TIME benchmark 评估")
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
from gluonts.time_feature import get_seasonality
from timebench.evaluation.data import (
    Dataset,
    get_dataset_settings,
    load_dataset_config,
)
from timebench.evaluation.saver import save_window_predictions
from timebench.evaluation.utils import get_available_terms

from model.RWKV_Time.rwkv_time_backbone import RWKVTimeBackbone
from model.RWKV_Time.rwkv_time_causal_lm_preprocessor import (
    RWKV_time_CausalLMPreprocessor,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("RWKV-Time-TIME-Eval")

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
    """把 RWKV-Time 骨干包装成逐窗口批量预测器。

    Args:
        prediction_length: int。预测步长。
        context_length: int。上下文 patch 数。
        batch_size: int。预测批大小。
    """

    def __init__(
        self,
        prediction_length: int,
        context_length: int = 128,
        batch_size: int = 2048,
    ) -> None:
        self.prediction_length = prediction_length
        self.context_length = context_length
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
    def predict(self, test_data_input: List[Dict[str, Any]]) -> np.ndarray:
        """对整个测试集的所有 series × window 实例批量预测。

        Args:
            test_data_input: 测试样本列表，每个样本含 target。

        Returns:
            [num_total_instances, prediction_length, n_quantiles], float32。
        """
        all_predictions: List[np.ndarray] = []
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
                model_inputs, batch_size=self.batch_size, verbose=0
            )
            model_outputs = model_outputs[: len(inputs)]

            pred = np.transpose(model_outputs[:, -self.num_tokens :], [0, 1, 3, 2])
            pred = np.reshape(pred, [len(pred), -1, backbone.n_quantiles])
            output_rescale = (pred * std + mean)[:, : self.prediction_length, :]
            all_predictions.append(output_rescale)

            del model_inputs
            gc.collect()

        return np.concatenate(all_predictions, axis=0)


# 评估配置
load_dotenv()

timebench_config: Dict[str, Any] = load_dataset_config()
all_dataset_configs: Dict[str, Any] = timebench_config.get("datasets", {})
all_datasets: List[str] = list(all_dataset_configs.keys())
logger.info(f"评估数据集总数: {len(all_datasets)}")

# 结果输出目录
model_name: str = os.path.basename(os.path.normpath(model_weights))
output_base_dir: str = os.path.join(SCRIPT_DIR, "results", "time_bench", model_name)
os.makedirs(output_base_dir, exist_ok=True)
csv_file_path: str = os.path.join(output_base_dir, "full_results.csv")

all_results: List[Dict[str, Any]] = []

# 遍历所有数据集与周期进行评估
for ds_num, ds_name in enumerate(all_datasets):
    logger.info(f"处理数据集: {ds_name} ({ds_num + 1}/{len(all_datasets)})")

    # 自动检测该数据集定义了哪些周期（short/medium/long）
    terms: List[str] = get_available_terms(ds_name, timebench_config)
    if not terms:
        logger.warning(f"数据集 {ds_name} 在 datasets.yaml 中没有可用周期，跳过")
        continue

    for term in terms:
        # 从 datasets.yaml 读取该任务的预测长度与测试长度
        settings = get_dataset_settings(ds_name, term, timebench_config)

        # 多变量数据集拆分为单变量评估
        temp_dataset = Dataset(name=ds_name, term=term, to_univariate=False)
        is_multivariate = temp_dataset.target_dim > 1
        dataset = Dataset(
            name=ds_name,
            term=term,
            to_univariate=is_multivariate,
            prediction_length=settings["prediction_length"],
            test_length=settings["test_length"],
            val_length=settings["val_length"],
        )

        predictor: RWKVTimePredictor = RWKVTimePredictor(
            prediction_length=dataset.prediction_length,
            # 动态计算上下文长度，确保不超过模型最大容量
            context_length=(args.max_length - dataset.prediction_length) // patch_size
            - 1,
            batch_size=8,
        )

        logger.info("开始预测...")
        predictions = predictor.predict(list(dataset.test_data.input))
        # (num_total_instances, prediction_length, n_quantiles) -> TIME 要求的布局
        fc_quantiles = np.transpose(predictions, [0, 2, 1])

        season_length = get_seasonality(dataset.freq)
        ds_config: str = f"{ds_name}/{term}"
        save_window_predictions(
            dataset=dataset,
            fc_quantiles=fc_quantiles,
            ds_config=ds_config,
            output_base_dir=output_base_dir,
            seasonality=season_length,
            model_hyperparams={
                "model_name": model_name,
                "patch_size": patch_size,
                "max_length": args.max_length,
            },
        )

        metrics_npz = np.load(os.path.join(output_base_dir, ds_config, "metrics.npz"))
        result_dict: Dict[str, Any] = {"dataset": ds_config, "model": model_name}
        for metric_key in metrics_npz.files:
            result_dict[f"eval_metrics/{metric_key}"] = float(
                np.nanmean(metrics_npz[metric_key])
            )
        all_results.append(result_dict)
        logger.info(f"数据集 {ds_config} 评估完成")

        del dataset, temp_dataset, predictor, predictions, fc_quantiles
        gc.collect()

# 保存结果并统计平均值
logger.info("所有评估完成，正在保存结果...")
df_results: pd.DataFrame = pd.DataFrame(all_results)
df_results.to_csv(csv_file_path, index=False)
logger.info(f"结果文件: {csv_file_path}")
logger.info(f"数据集-周期组合总数: {len(all_results)}")

numeric_columns = [col for col in df_results.columns if col.startswith("eval_metrics/")]

average_results = {col: df_results[col].mean() for col in numeric_columns}
df_average = pd.DataFrame([average_results])
average_csv_path = os.path.join(output_base_dir, "average_results.csv")
df_average.to_csv(average_csv_path, index=False)

logger.info("各指标平均值:")
for metric, value in average_results.items():
    logger.info(f"{metric:<50}: {value:.6f}")
logger.info(f"平均值结果已保存到: {average_csv_path}")
logger.info(
    "如需与官方榜单对齐（含 Seasonal Naive 相对分），可将结果目录复制或软链到 "
    "TIME/output/results/ 后运行 TIME/scripts/compute_local_leaderboard.py"
)
