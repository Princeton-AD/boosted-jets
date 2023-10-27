# Machine Learning Boosted Jets (MLBJ)

Machine learning based identification of boosted jets at CMS Calo-Layer-1 using regional deposits.

## Usage Instructions

Setup the environment, and install the requierments.
```
conda env create -f misc/environment.yml
conda activate solaris 
python3 -m pip install -r misc/requirements.txt
```

Generate `hdf5` dataset file based on the input `root` file:
```
mkdir -p data/histograms
python3 convert.py <input_root_file.root> data
```

Run experiments on the CNN, scanning for depth and width of the network:
```
mkdir results
python3 experiments.py data/dataset.h5 results
```

Generate the efficiency plots for one of the models:
```
python3 evaluation.py <path_to_model> data/dataset.h5
```

Compile the C++ model:
```
python3 compile.py <path_to_model> data/dataset.h5 <roc_comparison_output_path>
```
