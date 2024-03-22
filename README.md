# SVJ_boosted_lundnet
Adaptation of a Supervised Jet tagger exploiting Lund tree representation

## Benvenuto alla Repo per Boosted SVJ!

This repo is filled with hacks.... I mean incredibly useful features ;) for adapting the LundNetTagger to be used with the boosted SVJ search. 

*Spero che Cesare e Rob non saranno molto triste quando guardaranno a questi codici! Mi dispiace ragazzi*

How does this differ? Well, in a number of way:
1. The boosted search uses AK15 jets
    - Why? Because the Z' is recoiling against ISR. The Z' is light leading to a large jet containing most of the decay.
2. The boosted search is done on subleading jets
    - Why? The leading jet has the ISR the subleading has (almost always most of) the dark decay. This simplifies the search.
3. The tagger for the boosted search must be evaluated in mT windows +/- 100GeV of different sig mass hypothesis
    - Why? Because the SVJ structure changes immensely as the mass increases
4. The boosted search uses TreeMaker ntuples

The important things that we need to make sure work properly:
- Training and evaluating only on the subleading AK15
- Using AK15 jet constituents from the TreeMaker files
- Applying the preselection
- Splitting train, val, and test sets

Hacks.... I mean features:
- Works for TreeMaker files (**done** Rob)
- Preselection implemented as a mask stored in npz files (**done** Brendan)
    - Need to run it for background files, still having issues on HTCondor (**pending** Brendan)
- Preselection mask applied during the graph making step (**done** Rob)
- Implement and allow mT calculation using input ntuple and store it with the graphs (**done** Rob)
- Train, val, test mask applied during graph data set making (**todo** Brendan)

As more developments happen, I will include documentation about how to run this code including the new features

The steps: (**todo** Brendan)
- Make large train, val, test datasets 
- Train (hopefully it's not too difficult)
- Evaluate per mass point (rinv = 0.3 only)
- Make comparisons to BDT using AUCs at each point 

## Instructions

### Installation

First, the Lund Net tagger is needed, this must be done by asking the ETH Zurich group.

Second, a virtual environment is needed for running this code. I personally like conda, here are instructions for such a conda environment:
```bash
conda create -n boostLundenv python=3.10
conda activate boostLundenv  # Needed every time

conda install torch

pip install torch_geometric
pip install sklearn
pip install vector
pip install uproot
pip install awkward

pip install pandas
pip install requests
pip install numpy
pip install matplotlib

pip install torch
pip install fsspec-xrootd
pip install xrootd
```

