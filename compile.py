import argparse
import hls4ml
import numpy as np
import h5py
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras.models import Model
from sklearn.metrics import auc, roc_curve
from utils import IsReadableDir, IsValidFile
from typing import List, Optional


def get_dataset(source) -> (np.array, np.array):
    with h5py.File(source, "r") as f:
        X_data = f["deposits"][:]
        y_data = f["labels"][:] + 0.0
    _, X_test, _, y_test = train_test_split(
        X_data, y_data, test_size=0.2, random_state=42, shuffle=True, stratify=y_data
    )
    return X_test.reshape(-1, 9, 1) / 1024., y_test


def plot_roc(tprs: List[np.array], fprs:List[ np.array], destination: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(
        fprs[0],
        tprs[0],        
        color="b",
        label=r"Keras ROC (AUC = %0.2f)" % (auc(fprs[0], tprs[0])),
        lw=2,
        alpha=0.8,
    )
    ax.plot(
        fprs[1],
        tprs[1],        
        color="r",
        label=r"hls4ml ROC (AUC = %0.2f)" % (auc(fprs[1], tprs[1])),
        lw=2,
        alpha=0.8,
    )
    ax.set(
        xlim=[-0.05, 1.05],
        ylim=[-0.05, 1.05],
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
    )
    plt.plot([0, 1], [0, 1], "k--")
    ax.axis("square")
    ax.legend(loc="lower right")
    plt.savefig(f"{destination}/roc-comparison.png", bbox_inches="tight")
    plt.close()


def run_compilation(datasetpath: Path, modelpath: Path, destination: Path) -> None:
    model = keras.models.load_model(modelpath)
    # Pop sigmoid
    model = Model(inputs=model.input, outputs=model.layers[-2].output)
    X_data, y_data = get_dataset(datasetpath)
    hls4ml.model.flow.flow.update_flow(
        "convert", remove_optimizers=["qkeras_factorize_alpha"]
    )
    hls_config = hls4ml.utils.config_from_keras_model(
        model, granularity="name", default_precision="ap_fixed<16, 2>"
    )
    hls4ml.model.optimizer.get_optimizer("output_rounding_saturation_mode").configure(
        layers=["relu0", "relu1", "relu2", "relu3"],
        rounding_mode="AP_RND",
        saturation_mode="AP_SAT",
        saturation_bits="AP_SAT",
    )
    hls_config["Model"]["Strategy"] = "Resource"
    for layer in hls_config["LayerName"].keys():
        hls_config["LayerName"][layer]["ReuseFactor"] = 1
        hls_config["LayerName"][layer]["Strategy"] = "Latency"
        if "conv" in layer:
            hls_config["LayerName"][layer]["ParallelizationFactor"] = 9
    hls4ml.model.optimizer.get_optimizer("output_rounding_saturation_mode").configure(
        layers=[]
    )
    hls_model = hls4ml.converters.convert_from_keras_model(
        model,
        clock_period=6.25,
        backend="Vitis",
        hls_config=hls_config,
        io_type="io_parallel",
        output_dir="boostedjets",
        part="xc7vx690tffg1927-2",
        project_name="boostedjets",
    )
    hls_model.compile()

    y_pred_keras = model.predict(X_data, verbose=0).ravel()
    y_pred_hls4ml = hls_model.predict(X_data).ravel()

    fpr_k, tpr_k, _ = roc_curve(y_data, y_pred_keras)
    fpr_h, tpr_h, _ = roc_curve(y_data, y_pred_hls4ml)
    plot_roc([tpr_k, tpr_h], [fpr_k, fpr_h], destination)


def main(args_in: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        """Evaluate CNN on CMS Calorimeter Layer-1 Trigger region energy deposits"""
    )
    parser.add_argument(
        "modelpath", action=IsValidFile, help="Input model file to evaluate", type=Path
    )
    parser.add_argument(
        "datasetpath", action=IsValidFile, help="Input HDF5 file with data", type=Path
    )
    parser.add_argument(
        "savepath", action=IsReadableDir, help="ROC curve output directory", type=Path
    )
    args = parser.parse_args(args_in)
    run_compilation(args.datasetpath, args.modelpath, args.savepath)


if __name__ == "__main__":
    main()
