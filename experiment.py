import argparse
import h5py
import numpy as np
import os
import matplotlib.pyplot as plt
import shutil

from heapq import nlargest
from tensorflow import keras
from keras import layers
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import auc, roc_curve
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import StratifiedKFold
from typing import List, Optional
from utils import IsReadableDir, IsValidFile

from clr import CyclicLR
from qkeras import QConv2D, QDense, QActivation


def get_dataset(source: Path) -> (np.array, np.array):
    with h5py.File(source, "r") as f:
        X_data = f["deposits"][:]
        y_data = f["labels"][:] + 0.0

    print(
        f"Percentage of the set that is boosted: {100*sum(y_data) / len(y_data):.2f}%."
    )
    print(f"The total number of samples in the set is: {len(y_data)}.")
    print(
        f"Taining samples: {len(y_data)*.8:.0f}; Testing samples: {len(y_data)*.2:.0f}."
    )

    return X_data.reshape(-1, 9, 1) / 1024., y_data


def build_model(depth: int, width: int) -> keras.Model:
    loss = keras.losses.BinaryFocalCrossentropy(
        apply_class_balancing=False, alpha=0.5, gamma=5.0, label_smoothing=0.05
    )
    adam = keras.optimizers.Adam(learning_rate=0.001)

    model = keras.Sequential(name="ml-boosted-jets")
    model.add(layers.Input(shape=(9, 1), name="inputs_"))
    model.add(layers.Reshape((3, 3, 1), name="reshape", input_shape=(9, 1)))
    for d in range(depth):
        model.add(
            QConv2D(
                width,
                (3, 3),
                use_bias=True,
                kernel_quantizer="quantized_bits(8, 1, 1, alpha=1)",
                bias_quantizer="quantized_bits(8, 1, 1, alpha=1)",
                input_shape=(3, 3, 1),
                padding="same",
                name=f"qconv{d}",
            )
        )
        model.add(QActivation("quantized_relu(8, 1)", name=f"relu{d}"))
    model.add(
        QConv2D(
            width,
            (3, 3),
            use_bias=True,
            kernel_quantizer="quantized_bits(8, 1, 1, alpha=1)",
            bias_quantizer="quantized_bits(8, 1, 1, alpha=1)",
            input_shape=(3, 3, 1),
            name=f"qconv{depth}",
        )
    )
    model.add(layers.Flatten(name="flatten"))
    model.add(
        QDense(
            1,
            use_bias=True,
            kernel_quantizer="quantized_bits(8, 1, 1, alpha=1)",
            bias_quantizer="quantized_bits(8, 1, 1, alpha=1)",
            name="output",
        )
    )
    model.add(layers.Activation("sigmoid", name="sigmoid"))
    model.compile(optimizer=adam, loss=loss, metrics=["accuracy"])
    return model


def plot_roc(
    tprs: np.array,
    mean_fpr: np.array,
    aucs: List[float],
    destination: Path,
    depth: int,
    width: int,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(
        mean_fpr,
        mean_tpr,
        color="b",
        label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
        lw=2,
        alpha=0.8,
    )

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(
        mean_fpr,
        tprs_lower,
        tprs_upper,
        color="grey",
        alpha=0.2,
        label=r"$\pm$ 1 std. dev.",
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
    plt.savefig(f"{destination}/roc-curve-d{depth}-w{width}.png", bbox_inches="tight")
    plt.close()


def plot_results(results: np.array, destination: Path) -> None:
    plt.imshow(results)
    for x_val in range(results.shape[1]):
        for y_val in range(results.shape[0]):
            c = "{0:.2}".format(results[y_val, x_val])
            plt.text(x_val, y_val, c, va="center", ha="center")
    plt.xlabel("No Channels")
    plt.ylabel("No Layers")
    plt.yticks(np.arange(results.shape[0]), np.arange(1, results.shape[0]+1))
    plt.xticks(np.arange(results.shape[1]), np.arange(1, results.shape[1]+1))
    plt.savefig(f"{destination}/results.png", bbox_inches="tight")
    plt.close()


def run_experiment(
    source: Path,
    destination: Path,
    epochs: int,
    cv: int,
    bs: int,
    width: int,
    depth: int,
) -> None:
    X_data, y_data = get_dataset(source)

    X_train, X_test, y_train, y_test = train_test_split(
        X_data, y_data, test_size=0.2, random_state=42, shuffle=True, stratify=y_data
    )

    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

    results = np.zeros((depth, width))

    for d in range(depth):
        for w in range(1, width + 1):
            tprs = []
            aucs = []
            mean_fpr = np.linspace(0, 1, 100)

            for i, (idx_train, idx_val) in enumerate(skf.split(X_train, y_train)):
                class_weight = compute_class_weight(
                    "balanced",
                    classes=np.unique(y_train[idx_train]),
                    y=y_train[idx_train],
                )

                model = build_model(d, w)

                clr = CyclicLR(base_lr=0.001, max_lr=0.006, step_size=50.0)

                checkpoint_filepath = "./tmp/checkpoint"

                model_checkpoint_callback = ModelCheckpoint(
                    filepath=checkpoint_filepath,
                    save_weights_only=True,
                    monitor="val_loss",
                    mode="min",
                    save_best_only=True,
                )

                hist = model.fit(
                    X_train[idx_train],
                    y_train[idx_train],
                    batch_size=bs,
                    epochs=epochs,
                    class_weight={0: class_weight[0], 1: class_weight[1]},
                    validation_data=(X_train[idx_val], y_train[idx_val]),
                    shuffle=True,
                    verbose=0,
                    callbacks=[clr, model_checkpoint_callback],
                )

                model.load_weights(checkpoint_filepath)
                y_pred = model.predict(X_test, verbose=0).ravel()
                fpr, tpr, _ = roc_curve(y_test, y_pred)
                roc_auc = auc(fpr, tpr)
                interp_tpr = np.interp(mean_fpr, fpr, tpr)
                interp_tpr[0] = 0.0
                tprs.append(interp_tpr)
                # Save the best model
                if not aucs or roc_auc > max(aucs):
                    model.save(f"{destination}/model-d{d}-w{w}")
                aucs.append(roc_auc)

            # Pop bottom half of low AUCs: initialization is too important
            aucs, tprs = zip(*nlargest(cv - cv // 2, list(zip(aucs, tprs))))

            mean_tpr = np.mean(tprs, axis=0)
            mean_tpr[-1] = 1.0
            mean_auc = auc(mean_fpr, mean_tpr)

            results[d][w - 1] = mean_auc

            plot_roc(tprs, mean_fpr, aucs, destination, d, w)
    plot_results(results, destination)
    shutil.rmtree("./tmp")


def main(args_in: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        """Run CNN training to detect boosted jets on CMS Calorimeter Layer-1 Trigger region energy deposits"""
    )
    parser.add_argument(
        "dataset",
        action=IsValidFile,
        help="Input HDF5 dataset file",
        default="data/dataset.h5",
        type=Path,
    )
    parser.add_argument(
        "savepath",
        action=IsReadableDir,
        help="Output for plots",
        default="results",
        type=Path,
    )
    parser.add_argument(
        "-e",
        "--epochs",
        help="Number of training epochs",
        dest="epochs",
        default=100,
        type=int,
    )
    parser.add_argument(
        "-c",
        "--validation",
        help="Number of cross validation folds",
        dest="cv",
        default=4,
        type=int,
    )
    parser.add_argument(
        "-b",
        "--batch",
        help="Number of samples in a batch",
        dest="bs",
        default=2048,
        type=int,
    )
    parser.add_argument(
        "-w",
        "--width",
        help="Maximum width of the network",
        dest="width",
        default=6,
        type=int,
    )
    parser.add_argument(
        "-d",
        "--depth",
        help="Maximum depth of the network",
        dest="depth",
        default=6,
        type=int,
    )
    args = parser.parse_args(args_in)
    run_experiment(
        args.dataset,
        args.savepath,
        args.epochs,
        args.cv,
        args.bs,
        args.width,
        args.depth,
    )


if __name__ == "__main__":
    main()
