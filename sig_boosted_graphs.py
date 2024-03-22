# -*- coding: utf-8 -*eu-
#========================================================================================
# sig_boosted_graphs.py --------------------------------------------------------------
#----------------------------------------------------------------------------------------
# Author(s): Brendan Regnery, Roberto Seidita -------------------------------------------
#----------------------------------------------------------------------------------------

# Imports
import torch
from JetGraphProducer import JetGraphProducer, mt
import numpy as np, awkward as ak
import uproot
from LundTreeUtilities import OnTheFlyNormalizer
from torch.utils.data import ConcatDataset

# Making the graphs
signal = JetGraphProducer(
    "root://cmseos.fnal.gov//store/user/lpcdarkqcd/boosted/brendanSig/",
    n_store_jets=2,
    use_lund_decomp=True,
    n_lund_vars=5,
    weights="xsec",
    extra_obs_to_load=["MET", "METPhi"],
    extra_obs_to_compute_per_event=[mt],
    input_format="TreeMaker2",
    jet_collection="JetsAK15",
    verbose=True,
    mask=True,
    max_events_to_process=3000,
    label=1.,
)

# Splitting into train and test datasets
signal_training = signal[::2][:int(len(signal)/2*0.8)]
signal_testing = signal[::2][int(len(signal)/2*0.8):]

weigths_signal_training = signal_training.w
weigths_signal_testing = signal_testing.w

means, stds = 0., 0.

for graph in signal_training:
    means += graph.x.sum(dim=0)*graph.w
for graph in background_training:
    means += graph.x.sum(dim=0)*graph.w
for graph in signal_training:
    stds += ((graph.x - means)**2).sum(dim=0)*graph.w
for graph in background_training:
    stds += ((graph.x - means)**2).sum(dim=0)*graph.w

stds /= (signal_training.w.sum()+background_training.w.sum())
stds = torch.sqrt(stds)

# Careful that the normalizer is applied only once: slices in torch_geometric are actually only masks,
# so _training and _testing objects share the same underlying tensor

normalizer = OnTheFlyNormalizer(["x"], means, stds)
normalizer(signal_training.data)
normalizer(background_training.data)

data_training = ConcatDataset((signal_training, background_training))
data_testing = ConcatDataset((signal_testing, background_testing))
weights = torch.cat((
    weigths_signal_training/weigths_signal_training.sum(),
    weigths_background_training/weigths_background_training.sum()
    ))
weights_testing = torch.cat((
    weigths_signal_testing/weigths_signal_testing.sum(),
    weigths_background_testing/weigths_background_testing.sum()
    ))
