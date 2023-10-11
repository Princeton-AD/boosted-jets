import argparse
import h5py
import numpy as np
import ROOT as root
import os

from itertools import compress
from pathlib import Path
from typing import List, Optional
from tensorflow import keras
from utils import IsReadableDir, IsValidFile


def get_dataset(
    source: Path,
) -> (np.array, np.array, np.array, np.array, np.array, np.array, np.array):
    with h5py.File(source, "r") as f:
        X_data = f["deposits"][:]
        l1_jets_deltas = f["l1_jets_deltas"][:]
        l1_jets_pts = f["l1_jets_pts"][:]
        l1_pt = f["l1_pt"][:]
        reco_pt = f["reco_pt"][:]
        reco_eta = f["reco_eta"][:]
        jets_per_event = f["jets_per_event"][:]

    return (
        X_data.reshape(-1, 9, 1) / 1024.,
        l1_jets_deltas,
        l1_jets_pts,
        l1_pt,
        reco_pt,
        jets_per_event,
        reco_eta,
    )


def load_model(source: Path) -> keras.Model:
    return keras.models.load_model(source)


def draw_efficeincy(reco_Pt: np.array, cnn_l1_pt: np.array, bp_l1_pt: np.array) -> None:
    cnvs = root.TCanvas("cnvs", "canvas", 1000, 1000)
    total_hist = root.TH1F("total_hist", "Pt: total; Events; Pt ", 40, 0, 600)
    passed_hist_cnn = root.TH1F(
        "passed_hist_cnn", "Pt with the 100 min cut  ; Events; Pt ", 40, 0, 600
    )
    passed_hist_bp = root.TH1F(
        "passed_hist_bp", "Pt with the 100 min cut  ; Events; Pt ", 40, 0, 600
    )

    for pt in reco_Pt[reco_Pt > 0]:
        total_hist.Fill(pt)

    for pt in reco_Pt[cnn_l1_pt > 100]:
        passed_hist_cnn.Fill(pt)

    for pt in reco_Pt[bp_l1_pt > 100]:
        passed_hist_bp.Fill(pt)

    eff_cnn = root.TEfficiency(passed_hist_cnn, total_hist)
    eff_cnn.SetLineColor(4)
    eff_cnn.SetLineStyle(1)
    eff_cnn.Draw()

    eff_bp = root.TEfficiency(passed_hist_bp, total_hist)
    eff_bp.SetLineColor(3)
    eff_bp.SetLineStyle(1)
    eff_bp.Draw("same")

    legend = root.TLegend(0.9, 0.2, 0.7, 0.3)
    legend.SetHeader("Efficiencies", "C")
    legend.AddEntry(eff_cnn, "CNN", "l")
    legend.AddEntry(eff_bp, "Bit Pattern ", "l")
    legend.Draw()
    cnvs.Draw()
    cnvs.SaveAs(f"results/efficiency.png")


def run_evaluation(source: Path, modelpath: Path) -> None:
    model = load_model(modelpath)
    (
        X_data,
        l1_jets_deltas,
        l1_jets_pts,
        l1_pt,
        reco_pt,
        jets_per_event,
        reco_eta,
    ) = get_dataset(source)
    scores = model.predict(X_data, verbose=0).ravel()
    results = [(d, p) for d, p in zip(l1_jets_deltas, l1_jets_pts)]
    cnn_l1_pt = []
    prev = 0
    for reco_index, ele in enumerate(np.cumsum(jets_per_event)):
        pt = 0
        if reco_eta[reco_index] > -5:
            mask = scores[prev:ele] > 0.5
            result = results[prev:ele]
            result = list(compress(result, mask))
            if result:
                _, pt = min(result)
        cnn_l1_pt.append(pt)
        prev = ele
    cnn_l1_pt = np.array(cnn_l1_pt)
    draw_efficeincy(reco_pt, cnn_l1_pt, l1_pt)


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
    args = parser.parse_args(args_in)
    run_evaluation(args.datasetpath, args.modelpath)


if __name__ == "__main__":
    main()
