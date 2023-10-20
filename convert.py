import argparse
import awkward as ak
import h5py
import numpy as np
import uproot
import os
import vector

from pathlib import Path
from typing import List, Optional
from utils import IsReadableDir, IsValidFile


def parse_source(source: Path, target: Path) -> None:
    _file = uproot.open(source)
    _tree = _file["l1NtupleProducer/efficiencyTree"]

    # Process inputs
    deposits = _tree["jetRegionEt"].array()
    deposits = ak.flatten(deposits)

    # Process labels
    labels = _tree["allL1Signals"].array()
    labels = ak.flatten(labels, axis=None)

    # Read the reconstruction data
    reco_pt = _tree["recoPt_1"].array()
    reco_eta = _tree["recoEta_1"].array()
    reco_phi = _tree["recoPhi_1"].array()

    # Read the L1 jets data
    l1_pt = _tree["l1Pt_1"].array()
    l1_jets = _tree["allL1Jets"].array()
    jets_per_event = [len(_) for _ in l1_jets]
    l1_eta = _tree["l1Eta_1"].array().to_numpy()
    l1_phi = _tree["l1Phi_1"].array().to_numpy()
    
    l1_reco_deltaR =  (l1_eta - reco_eta)**2 + (l1_phi - reco_phi)**2

    lv_eta = ak.broadcast_arrays(reco_eta, l1_jets)[0]
    lv_phi = ak.broadcast_arrays(reco_phi, l1_jets)[0]
    recopt_broadcasted = ak.broadcast_arrays(reco_pt, l1_jets) [0]
    l1_jets = ak.flatten(l1_jets)
    lorenz_vectors = vector.arr(
        {
            "px": l1_jets[:]["fP"]["fX"],
            "py": l1_jets[:]["fP"]["fY"],
            "pz": l1_jets[:]["fP"]["fZ"],
            "pt": l1_jets[:]["fE"],
        }
    )

    jets_deltas = (
        (lorenz_vectors.eta - ak.flatten(lv_eta)) ** 2
        + (lorenz_vectors.phi - ak.flatten(lv_phi)) ** 2
    ) ** 0.5
    jets_pt = lorenz_vectors.pt
    jets_eta = lorenz_vectors.eta
    jets_phi = lorenz_vectors.phi

    #for the pt resolution: 
    bit_pt_resolution =  reco_pt - l1_pt
    jets_pt_resolution = ak.flatten(recopt_broadcasted) - jets_pt

    #for the multiplicity plots
    l1_jets_2 = _tree["l1Jets"].array() 
    bit_pattern_multiplicity = [len(_) for _ in l1_jets_2 ]

    #for the last plot they asked me for I dont know what the name of that is 
   # l1_jets_2 = ak.flatten(l1_jets_2)
    #lorentz_vectors_2 = vector.arr({

        #"px": l1_jets[:]["fP"]["fX"],
        #"py": l1_jets[:]["fP"]["fY"],
        #"pz": l1_jets[:]["fP"]["fZ"],
    #    "pt": l1_jets[:]["fE"],
                 
   # })
   # prev = 0 
    #for ele in np.cumsum(bit_pattern_multiplicity): 

        #l1_jets_2


    #python3 convert.py /Users/jorgehernandez/Documents/HEP_work/BoostedJetML/l1TNtuple-ggHBB_29Jul.root data/dataset.h5
        
    # Write all the arrays into the H5 file.
    with h5py.File(f"{target}/dataset.h5", "w") as f:
        f.create_dataset("deposits", data=deposits.to_numpy())
        f.create_dataset("labels", data=labels.to_numpy())
        f.create_dataset("reco_eta", data=reco_eta.to_numpy())
        f.create_dataset("reco_phi", data=reco_phi.to_numpy())
        f.create_dataset("reco_pt", data=reco_pt.to_numpy())
        f.create_dataset("l1_pt", data=l1_pt.to_numpy())
        f.create_dataset("l1_jets", data=l1_jets.to_numpy())
        f.create_dataset("l1_jets_deltas", data=jets_deltas.to_numpy())
        f.create_dataset("l1_jets_pt", data=jets_pt)
        f.create_dataset("jets_eta", data = jets_eta)
        f.create_dataset("jets_phi", data = jets_phi) 
        f.create_dataset("jets_per_event", data=jets_per_event)
        f.create_dataset("l1_reco_deltaR", data = l1_reco_deltaR.to_numpy())
        f.create_dataset("l1_phi" , data = l1_phi)
        f.create_dataset("l1_eta" , data = l1_eta)
        f.create_dataset("bit_pt_resolution" , data = bit_pt_resolution.to_numpy())
        f.create_dataset("jets_pt_res", data = jets_pt_resolution.to_numpy())
        f.create_dataset("bit_multiplicity" , data = bit_pattern_multiplicity)


def main(args_in: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        """Convert CMS Calorimeter Layer-1 Trigger region energy deposits from ROOT to HDF5 format"""
    )
    parser.add_argument(
        "filepath", action=IsValidFile, help="Input ROOT file", type=Path
    )
    parser.add_argument(
        "savepath", action=IsReadableDir, help="Output HDF5 file", type=Path
    )
    args = parser.parse_args(args_in)
    parse_source(args.filepath, args.savepath)


if __name__ == "__main__":
    main()
