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
) -> (
    np.array,
    np.array,
    np.array,
    np.array,
    np.array,
    np.array,
    np.array,
    np.array,
    np.array,
    np.array,
    np.array,
    np.array,
    np.array,
    np.array,
):
    with h5py.File(source, "r") as f:
        X_data = f["deposits"][:]  # 1
        l1_jets_deltas = f["l1_jets_deltas"][:]  # 2
        l1_jets_pts = f["l1_jets_pt"][:]  # 3
        l1_pt = f["l1_pt"][:]  # 4
        reco_pt = f["reco_pt"][:]  # 5
        reco_eta = f["reco_eta"][:]  # 6
        jets_per_event = f["jets_per_event"][:]  # 7
        jets_phi = f["jets_phi"][:]  # 8
        jets_eta = f["jets_eta"][:]  # 9
        l1_reco_deltaR = f["l1_reco_deltaR"][:]  # 10
        bit_pt_resolution = f["bit_pt_resolution"][:]  # 11
        jets_pt_res = f["jets_pt_res"][:]  # 12
        l1_phi = f["l1_phi"][:]  # 13
        l1_eta = f["l1_eta"][:]  # 14
        bit_multiplicity = f["bit_multiplicity"][:]  # 15

    return (
        X_data.reshape(-1, 9, 1) / 1024.0,  # 1
        l1_jets_deltas,  # 2
        l1_jets_pts,  # 3
        l1_pt,  # 4
        reco_pt,  # 5
        jets_per_event,  # 6
        reco_eta,  # 7
        jets_eta,  # 8
        jets_phi,  # 9
        l1_reco_deltaR,  # 10
        bit_pt_resolution,  # 11
        jets_pt_res,  # 12
        l1_eta,  # 13
        l1_phi,  # 14
        bit_multiplicity,  # 15
    )


def load_model(source: Path) -> keras.Model:
    return keras.models.load_model(source)


def draw_efficiency(reco_Pt: np.array, cnn_l1_pt: np.array, bp_l1_pt: np.array) -> None:
    cnvs = root.TCanvas("cnvs", "canvas", 1000, 1000)
    total_hist = root.TH1F("total_hist", " ; Offline p_{T} ; Efficiency ", 40, 0, 600)
    passed_hist_cnn = root.TH1F(
        "passed_hist_cnn", " ; Offline p_{T} ; Efficiency ", 40, 0, 600
    )
    passed_hist_bp = root.TH1F(
        "passed_hist_bp", " ; Offline p_{T} ; Efficiency ", 40, 0, 600
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
    cnvs.SaveAs(f"results/histograms/efficiency.png")
    cnvs.Delete()


def draw_deltaR_histograms(l1_reco_deltaR: np.array, jets_deltaR: np.array) -> None:
    # this needs the cut for default -99 values in the etas and phi's. This should prob be done somewhere else.

    dR_hist_l1reco = root.TH1F("deltaR_hist_l1reco", " ; #Delta R; #events  ", 40, 0, 1)
    jets_deltaR_hist = root.TH1F("delta R ", ";  #Delta R ;  #events", 40, 0, 1)
    dR_canvas = root.TCanvas("dR_canvas", "", 1000, 1000)

    dR_hist_l1reco, jets_deltaR_hist, legend = draw_comparison_histogram(
        l1_reco_deltaR, jets_deltaR, dR_hist_l1reco, jets_deltaR_hist, cut=None
    )

    jets_deltaR_hist.Draw("same")
    dR_hist_l1reco.Draw("same")
    legend.Draw()

    # gStyle.SetOptStats(0)
    dR_canvas.SetRightMargin(0.15)
    dR_canvas.SetBottomMargin(0.15)
    dR_canvas.SetLeftMargin(0.15)

    dR_canvas.Draw()
    dR_canvas.SaveAs(f"results/histograms/deltaR_histogram.png")
    dR_canvas.Delete()


def draw_pt_histograms(bit_pt: np.array, cnn_pt: np.array) -> None:
    pt_canvas = root.TCanvas("Tcanvas", "Canvas", 800, 800)
    cnn_pt_hist = root.TH1F("Pt_comparison", " ; p_{T} [GeV]; #events", 40, 0, 600)
    bit_pt_hist = root.TH1F("Pt_comparison", " ; p_{T} [GeV]; #events", 40, 0, 600)

    bit_pt, cnn_pt, legend = draw_comparison_histogram(
        bit_pt, cnn_pt, bit_pt_hist, cnn_pt_hist, cut=None
    )

    bit_pt_hist.Draw()
    cnn_pt_hist.Draw("same")
    legend.Draw()

    pt_canvas.SetRightMargin(0.15)
    pt_canvas.SetBottomMargin(0.15)
    pt_canvas.SetLeftMargin(0.15)

    pt_canvas.Draw()
    pt_canvas.SaveAs(f"results/histograms/pt_histograms.png")


def draw_pt_resolution_hist(bit_pt_resolution: np.array, cnn_pt_resolution: np.array):
    ptResolution_canvas = root.TCanvas(
        "ptResolution_canvas", "pt resolution", 1000, 1000
    )
    pt_resolution_cnn = root.TH1F(
        "", ";  p_{T} resolution [GeV]; #events", 40, -600, 600
    )
    pt_resolution_l1 = root.TH1F(
        "pt_resolution",
        "Pt resolution comparison ; Events; (reco -l1)pt ",
        40,
        -600,
        600,
    )

    bit_pt_resolution, cnn_pt_resolution, legend = draw_comparison_histogram(
        bit_pt_resolution,
        cnn_pt_resolution,
        pt_resolution_l1,
        pt_resolution_cnn,
        cut=None,
    )

    pt_resolution_cnn.Draw()
    pt_resolution_l1.Draw("same")
    legend.Draw()
    ptResolution_canvas.SetRightMargin(0.15)
    ptResolution_canvas.SetBottomMargin(0.15)
    ptResolution_canvas.SetLeftMargin(0.15)

    ptResolution_canvas.Draw()
    ptResolution_canvas.SaveAs(f"results/histograms/pt_resolution_histograms.png")
    ptResolution_canvas.Delete()


def draw_eta_histograms(bit_eta: np.array, cnn_eta: np.array):
    bit_eta_hist = root.TH1F("", " ; #eta;  #events", 40, -100, 100)
    eta_canvas = root.TCanvas("Tcanvas", "Canvas", 1000, 1000)
    cnn_eta_hist = root.TH1F("", " ; #eta;  #events", 40, -100, 100)

    bit_eta_hist, cnn_eta_hist, legend = draw_comparison_histogram(
        bit_eta, cnn_eta, bit_eta_hist, cnn_eta_hist, cut=None
    )
    eta_canvas.SetRightMargin(0.15)
    eta_canvas.SetBottomMargin(0.15)
    eta_canvas.SetLeftMargin(0.15)
    cnn_eta_hist.Draw()
    bit_eta_hist.Draw("same")

    legend.Draw()

    eta_canvas.Draw()
    eta_canvas.SaveAs(f"results/histograms/eta_histograms.png")
    eta_canvas.Delete()


def draw_phi_histograms(bit_phi: np.array, cnn_phi: np.array):
    phi_canvas = root.TCanvas("Tcanvas", "Canvas", 1000, 1000)
    bit_phi_hist = root.TH1F("phi_Comparison", " ; #phi;  #events", 40, -3, 3)
    cnn_phi_hist = root.TH1F("phi_Comparison", " ; #phi;  #events", 40, -3, 3)

    bit_phi_hist, cnn_phi_hist, legend = draw_comparison_histogram(
        bit_phi, cnn_phi, bit_phi_hist, cnn_phi_hist, cut=None
    )

    cnn_phi_hist.Draw()
    bit_phi_hist.Draw("same")
    legend.Draw()

    phi_canvas.SetRightMargin(0.15)
    phi_canvas.SetBottomMargin(0.15)
    phi_canvas.SetLeftMargin(0.15)

    phi_canvas.Draw()
    phi_canvas.SaveAs(f"results/histograms/phi_histograms.png")
    phi_canvas.Delete()


def draw_multiplicity_histograms(
    bit_multiplicity: np.array, cnn_multiplicity: np.array
) -> ():
    multiplicity_canvas = root.TCanvas("Tcanvas", "Canvas", 1000, 1000)

    cnn_multi_hist = root.TH1F("njet", " ; N_{jet} ; #events", 15, 0, 15)
    bit_multi_hist = root.TH1F("njet", " ; N_{jet} ; #events", 15, 0, 15)

    bit_multi_hist, cnn_multi_hist, legend = draw_comparison_histogram(
        bit_multiplicity, cnn_multiplicity, bit_multi_hist, cnn_multi_hist, cut=None
    )

    cnn_multi_hist.Draw("same")
    bit_multi_hist.Draw("same")
    legend.Draw()

    multiplicity_canvas.SetRightMargin(0.15)
    multiplicity_canvas.SetBottomMargin(0.15)
    multiplicity_canvas.SetLeftMargin(0.15)

    multiplicity_canvas.Draw()
    multiplicity_canvas.SaveAs(f"results/histograms/multiplicity_histograms.png")
    multiplicity_canvas.Delete()


def draw_comparison_histogram(
    arr1: np.array, arr2: np.array, hist1: root.TH1F, hist2: root.TH1F, cut=None
) -> (root.TH1F, root.TH1F, root.TLegend):
    if cut is not None:
        for ele in arr1:
            if ele > cut:
                hist1.Fill(ele)

        for ele in arr2:
            if ele > cut:
                hist2.Fill(ele)

    else:
        for ele in arr1:
            hist1.Fill(ele)

        for ele in arr2:
            hist2.Fill(ele)

    hist1.SetFillColorAlpha(1, 0.1)
    hist1.SetLineColor(1)
    hist2.SetLineColor(2)
    hist2.SetFillColorAlpha(2, 0.1)

    legend = root.TLegend(0.9, 0.2, 0.7, 0.3)

    legend.AddEntry(hist1, "bit pattern", "l")
    legend.AddEntry(hist2, "CNN", "l")

    return hist1, hist2, legend


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
        jets_phi,
        jets_eta,
        l1_reco_deltaR,
        bit_pt_resolution,
        jets_pt_res,
        l1_eta,
        l1_phi,
        bit_multiplicity,
    ) = get_dataset(source)
    scores = model.predict(X_data, verbose=0).ravel()

    results = [
        (d, p, e, ph, res)
        for d, p, e, ph, res in zip(
            l1_jets_deltas, l1_jets_pts, jets_eta, jets_phi, jets_pt_res
        )
    ]
    cnn_l1_pt = []
    cnn_l1_eta = []
    cnn_l1_phi = []
    cnn_l1_deltaR = []
    cnn_pt_resolution = []
    cnn_jet_multiplicity = []

    prev = 0
    for reco_index, ele in enumerate(np.cumsum(jets_per_event)):
        pt = 0
        eta = 0
        deltaR = 0
        phi = 0
        pt_resolution = 0

        mask_for_multiplicity = scores[prev:ele] > 0.5
        cnn_jet_multiplicity.append(np.sum(mask_for_multiplicity))

        if reco_eta[reco_index] > -5:
            mask = scores[prev:ele] > 0.5
            result = results[prev:ele]
            result = list(compress(result, mask))
            if result:
                deltaR, pt, eta, phi, pt_resolution = min(result)
        cnn_l1_pt.append(pt)
        cnn_l1_eta.append(eta)
        cnn_l1_phi.append(phi)
        cnn_l1_deltaR.append(deltaR)
        cnn_pt_resolution.append(pt_resolution)
        prev = ele
    cnn_l1_pt = np.array(cnn_l1_pt)

    draw_deltaR_histograms(l1_reco_deltaR, cnn_l1_deltaR)
    draw_pt_histograms(l1_pt, cnn_l1_pt)
    draw_pt_resolution_hist(bit_pt_resolution, cnn_pt_resolution)
    draw_eta_histograms(l1_eta, cnn_l1_eta)
    draw_phi_histograms(l1_phi, cnn_l1_phi)
    draw_multiplicity_histograms(bit_multiplicity, cnn_jet_multiplicity)
    draw_efficiency(reco_pt, cnn_l1_pt, l1_pt)

    # Instructions for another histogram:  "It should be finding the jet with the highest pt out of all the jets you select with the score > x requirement,
    # then plot that pt in a histogram. This should be a single number per event. Then you compare that to the jet with the highest pt out of the l1Jets vector stored in the root file -> another histogram"


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
