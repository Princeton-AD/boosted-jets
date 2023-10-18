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
) -> (np.array, np.array, np.array, np.array, np.array, np.array, np.array, np.array, np.array, np.array, np.array, np.array, np.array, np.array):
    with h5py.File(source, "r") as f:
        X_data = f["deposits"][:] #1 
        l1_jets_deltas = f["l1_jets_deltas"][:] #2
        l1_jets_pts = f["l1_jets_pt"][:] #3 
        l1_pt = f["l1_pt"][:]#4
        reco_pt = f["reco_pt"][:] #5 
        reco_eta = f["reco_eta"][:] #6
        jets_per_event = f["jets_per_event"][:] #7 
        jets_phi = f ["jets_phi"][:] #8
        jets_eta =f["jets_eta"][:] #9
        l1_reco_deltaR = f["l1_reco_deltaR"][:] #10
        bit_pt_resolution = f["bit_pt_resolution"][:] #11
        jets_pt_res = f["jets_pt_res"][:] #12 
        l1_phi =  f["l1_phi"][:] #13 
        l1_eta = f["l1_eta"][:] #14 
        bit_multiplicity = f["bit_multiplicity"] [:] #15 
        #print(len(jets_pt_res))


    return (
        X_data.reshape(-1, 9, 1) / 1024.,#1 
        l1_jets_deltas, #2
        l1_jets_pts, #3
        l1_pt, #4
        reco_pt,#5 
        jets_per_event, #6
        reco_eta,#7
        jets_eta, #8 
        jets_phi,#9
        l1_reco_deltaR,#10 
        bit_pt_resolution, #11
        jets_pt_res, #12
        l1_eta, #13 
        l1_phi,#14
        bit_multiplicity #15
    )


def load_model(source: Path) -> keras.Model:
    return keras.models.load_model(source)

def draw_efficiency(reco_Pt: np.array, cnn_l1_pt: np.array, bp_l1_pt: np.array) -> None:
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
    cnvs.SaveAs(f"results/histograms/efficiency.png")
    cnvs.Delete()

def draw_deltaR_histograms(l1_reco_deltaR , jets_deltaR) -> None: 
    
    #find the delta R from the ht file, fill the deltaR into a hist 
    #this needs the cut for default -99 values in the etas and phi's. This should prob be done somewhere else. 
    
    dR_hist_l1reco = root.TH1F("deltaR_hist_l1reco", " ; #Delta R; #events  " , 40, 0 ,1)

    for deltaR in l1_reco_deltaR: 

        dR_hist_l1reco.Fill(deltaR)

    jets_deltaR_hist = root.TH1F("delta R ", ";  #Delta R ;  #events", 40, 0, 1) 
    
    for deltaR in jets_deltaR : 

        jets_deltaR_hist.Fill(deltaR)

    dR_hist_l1reco.SetLineColor(1)
    dR_hist_l1reco.SetFillColorAlpha(1,0.1)
    jets_deltaR_hist.SetLineColor(2)
    jets_deltaR_hist.SetFillColorAlpha(2, 0.1)
    
    
    dR_canvas = root.TCanvas("dR_canvas", "" ,  1000, 1000)


    jets_deltaR_hist.Draw("same")
    dR_hist_l1reco.Draw("same")

    legend = root.TLegend(0.9,0.2,0.7,0.3);

    legend.AddEntry(dR_hist_l1reco,"bit pattern","l")
    legend.AddEntry(jets_deltaR_hist,"CNN","l")
    legend.Draw()

    #gStyle.SetOptStats(0)
    dR_canvas.Draw()
    dR_canvas.SaveAs(f"results/histograms/deltaR_histogram.png")
    dR_canvas.Delete()

#def draw_eta_histograms () -> None: 


def draw_pt_histograms (bit_pt , cnn_pt) -> None: 

    cnn_pt_hist = root.TH1F("new_L1Pt", "Pt from the chosen deltaR^2_min ; Events: with filter predict score > 0,5 ; Pt " , 40, 0 ,600) 

    for pt in cnn_pt:
        if pt > 0: 
            #print(pt)
            cnn_pt_hist.Fill(pt) 
    #pt_histArray.append(new_L1Pt_hist)

     
    cnn_pt_hist.SetLineColor(2) 
    cnn_pt_hist.SetFillColorAlpha(2,0.1)
    cnn_pt_hist.SetFillStyle(2)

    bit_pt_hist = root.TH1F("Pt_comparison" , " ; p_{T} [GeV]; #events", 40, 0 ,600)  

    for pt in bit_pt: 
        if pt > 0:
            #print(pt) 
            bit_pt_hist.Fill(pt)

    bit_pt_hist.SetFillColorAlpha(1,0.1)
    bit_pt_hist.SetLineColor(1)
    pt_canvas = root.TCanvas("Tcanvas", "Canvas", 800, 800)
    bit_pt_hist.Draw()
    cnn_pt_hist.Draw("same")

    legend = root.TLegend(0.9,0.2,0.7,0.3);

    legend.AddEntry(bit_pt_hist,"bit pattern","l")
    legend.AddEntry(cnn_pt_hist,"CNN","l")
    legend.Draw()

    pt_canvas.Draw()
    pt_canvas.SaveAs(f"results/histograms/pt_histograms.png") 

def draw_pt_resolution_hist(bit_pt_resolution, cnn_pt_resolution): 

    pt_resolution_cnn  = root.TH1F ("", ";  p_{T} resolution [GeV]; #events" , 40, -600,600) 
    pt_resolution_l1 =  root.TH1F ("pt_resolution", "Pt resolution comparison ; Events; (reco -l1)pt " , 40, -600, 600) 

        #I might have to add some cuts for the pt in this dont want those crazy -99's 
    for ele in bit_pt_resolution: 

        pt_resolution_l1.Fill(ele)
    
    for ele in cnn_pt_resolution :

        pt_resolution_cnn.Fill(ele)

    ptResolution_canvas = root.TCanvas("ptResolution_canvas", "pt resolution", 1000, 1000)     

        #ptResolution_histArrays.append(pt_resolution_cnn) this is for the comparison among the different scores cuts. 

    pt_resolution_cnn.Draw()
    pt_resolution_cnn.SetFillColorAlpha(2,0.1)
    pt_resolution_cnn.SetLineColor(2) 

    pt_resolution_l1.Draw("same")
    pt_resolution_l1.SetFillColorAlpha(1, 0.1 )
    pt_resolution_l1.SetLineColor(1)

    legend = root.TLegend(0.9,0.2,0.7,0.3)
    legend.AddEntry(pt_resolution_l1,"bit pattern","l")
    legend.AddEntry(pt_resolution_cnn,"CNN","l")
    legend.Draw()

    ptResolution_canvas.Draw()
    ptResolution_canvas.SaveAs(f"results/histograms/pt_resolution_histograms.png")
    ptResolution_canvas.Delete()       

def draw_eta_histograms(bit_eta, cnn_eta ) :

    bit_eta_hist = root.TH1F("eta", "eta comparison; Events ; Eta " , 40, -3 ,3) 
    eta_canvas = root.TCanvas("Tcanvas", "Canvas", 800, 800)

    for eta in bit_eta: 

        if eta > -5 : 
          #print(eta)
          bit_eta_hist.Fill(eta)

    #eta_histArray.append(new_l1eta_hist)
    bit_eta_hist.Draw()

    cnn_eta_hist = root.TH1F("Eta_Comparison", " ; #eta;  #events" , 40 , -3 , 3 ) 

    for eta in cnn_eta: 

        if eta> -5: 

            cnn_eta_hist.Fill(eta) 

    bit_eta_hist.SetFillColorAlpha(1,0.1)
    bit_eta_hist.SetLineColor(1)
    cnn_eta_hist.SetLineColor(2) 
    cnn_eta_hist.SetFillColorAlpha(2,0.1)
   
    cnn_eta_hist.Draw("same")
    


    legend = root.TLegend(0.9,0.2,0.7,0.3);

    legend.AddEntry(cnn_eta_hist,"bit pattern","l")
    legend.AddEntry(bit_eta_hist,"CNN","l")
    legend.Draw()
                    
    eta_canvas.Draw() 
    eta_canvas.SaveAs(f"results/histograms/eta_histograms.png")
    eta_canvas.Delete() 



def draw_phi_histograms (bit_phi, cnn_phi): 

    bit_phi_hist = root.TH1F("phi", "phi comparison; Events ; phi " , 40, -3 ,3) 
    phi_canvas = root.TCanvas("Tcanvas", "Canvas", 800, 800)

    for phi in bit_phi: 

        if phi > -5 : 
          #print(eta)
          bit_phi_hist.Fill(phi)

    #eta_histArray.append(new_l1eta_hist)
    bit_phi_hist.Draw()

    cnn_phi_hist = root.TH1F("phi_Comparison", " ; #phi;  #events" , 40 , -3 , 3 ) 

    for phi in cnn_phi: 

        if phi> -5: 

            cnn_phi_hist.Fill(phi) 

    bit_phi_hist.SetFillColorAlpha(1,0.1)
    bit_phi_hist.SetLineColor(1)
    cnn_phi_hist.SetLineColor(2) 
    cnn_phi_hist.SetFillColorAlpha(2,0.1)
   
    cnn_phi_hist.Draw("same")
    


    legend = root.TLegend(0.9,0.2,0.7,0.3);

    legend.AddEntry(cnn_phi_hist,"bit pattern","l")
    legend.AddEntry(bit_phi_hist,"CNN","l")
    legend.Draw()
                    
    phi_canvas.Draw() 
    phi_canvas.SaveAs(f"results/histograms/phi_histograms.png")
    phi_canvas.Delete() 

def draw_multiplicity_histograms(bit_multiplicity:np.array ,cnn_multiplicity: np.array) -> () : 

    multiplicity_canvas = root.TCanvas("Tcanvas", "Canvas", 800, 800)

    cnn_multi_hist = root.TH1F("njet" , " ; N_{jet} ; #events" , 15, 0 ,15)  
    bit_multi_hist = root.TH1F("njet" , " ; N_{jet} ; #events" , 15, 0 ,15) 


    bit_multi_hist ,cnn_multi_hist, legend = draw_comparison_histogram(bit_multiplicity, cnn_multiplicity, bit_multi_hist, cnn_multi_hist, cut=None)

    cnn_multi_hist.Draw("same")
    bit_multi_hist.Draw("same")
    legend.Draw()
    multiplicity_canvas.Draw()
    multiplicity_canvas.SaveAs(f"results/histograms/multiplicity_histograms.png")
    multiplicity_canvas.Delete()




     



def draw_comparison_histogram(arr1 : np.array , arr2: np.array, hist1 : root.TH1F, hist2 : root.TH1F, cut=None) -> (root.TH1F, root.TH1F , root.TLegend):

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
    


    hist1.SetFillColorAlpha(1,0.1)
    hist1.SetLineColor(1)
    hist2.SetLineColor(2) 
    hist2.SetFillColorAlpha(2,0.1)
    
    legend = root.TLegend(0.9,0.2,0.7,0.3);

    legend.AddEntry(hist1 ,"bit pattern","l")
    legend.AddEntry(hist2 ,"CNN","l")
    
    return  hist1, hist2 , legend
    








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
        bit_multiplicity
        
    ) = get_dataset(source)
    scores = model.predict(X_data, verbose=0).ravel()
    # print(len(jets_pt_res))
    results = [(d, p, e, ph, res) for d, p, e, ph ,res in zip(l1_jets_deltas, l1_jets_pts, jets_eta, jets_phi, jets_pt_res)]
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
        deltaR= 0
        phi = 0
        pt_resolution = 0 

        mask_for_multiplicity = scores[prev:ele] > 0.5
        cnn_jet_multiplicity.append(np.sum(mask_for_multiplicity))
        
        if reco_eta[reco_index] > -5:
            mask = scores[prev:ele] > 0.5
            result = results[prev:ele]
            result = list(compress(result, mask))
            if result:
                deltaR , pt, eta, phi , pt_resolution = min(result)
        cnn_l1_pt.append(pt)
        cnn_l1_eta.append(eta)
        cnn_l1_phi.append(phi)
        cnn_l1_deltaR.append(deltaR) 
        cnn_pt_resolution.append(pt_resolution) 
        prev = ele
    cnn_l1_pt = np.array(cnn_l1_pt)
    # I have to bring the l1 phi and eta files from the convert file so that I can use it right here. 

    
    draw_deltaR_histograms(l1_reco_deltaR, cnn_l1_deltaR)
    draw_pt_histograms(l1_pt,cnn_l1_pt)
    draw_pt_resolution_hist(bit_pt_resolution, cnn_pt_resolution)
    draw_eta_histograms(l1_eta,cnn_l1_eta)
    draw_phi_histograms(l1_phi, cnn_l1_phi)
    draw_multiplicity_histograms(bit_multiplicity, cnn_jet_multiplicity)
    draw_efficiency(reco_pt, cnn_l1_pt, l1_pt)

    #How many histograms I got left? I need jet mulptiplicity and this one "It should be finding the jet with the highest pt out of all the jets you select with the score > x requirement, 
    #then plot that pt in a histogram. This should be a single number per event. Then you compare that to the jet with the highest pt out of the l1Jets vector stored in the root file -> another histogram"
    


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
