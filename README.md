# Building and running a GNN classifer
I'll try and give a whistle-stop tour how we can make a graph and a graph neural network with pytorch and torch_geometric, some of the files I've used, and an introductory notebook.

Python libraries:
	jupyter
	matplotlib
	networkx
	numpy
	pandas
	scikit-learn
	torch (see below)
	torch_geometric (see below)
	torchinfo
	uproot

plotly is not strictly necessary, but I like it for visualising space points interactively in 3D.
## torch and torch_geometric
I have a version of conda set up on the uboone gpvms and EAF, and I've found it convenient to set a custom location. 
Any python virtual environment should work as you can use pip to install pytorch and torch_geometric, but its best to follow their respective installation guides as it isn't quite as simple as "pip install torch" if you want to have cuda/GPU support.

Instructions for setting both libraries up can be found here:
torch: https://pytorch.org/get-started/locally/
torch_geometric: https://pytorch-geometric.readthedocs.io/en/2.6.1/install/installation.html

Torch provides the standard neural network building blocks and nice-to-haves, and torch_geometric has all your graph-specific layers such as graph_convolutions, and other useful tools such as a graph-specific dataloader.
## Minimally-viable GNN
### Space points 
I've been using wire cell space points, which can be found in wcpselection/T_spacepoints
There are a number of types of space points, made at different stages of clustering/reclustering/deconvolution. The recommended space point type is "T_recchargeblob_spacepoint_<x/y/z/q>" which are the space points after the final round of reclustering and give an excellent 3D reconstruction of charge within the detector. 

The minimally viable GNN would probably require just a reconstructed vertex and these 4D spacepoints, so thats what this example notebook will do. 

### Files
These were SURPRISE samples but I've slimmed them down to only include Wire Cell reconstruction and space points, so they are a lot smaller and faster to load/copy. 

**Signal: neutrino -> nu e+ e-** 	
`/exp/uboone/data/users/jbateman/workdir/DarkNews/Trident/SURPRISE/checkout_trident_SURPRISE_big_wirecell.root`
This should look like 1 shower, or 2 showers starting from a shared vertex

**Background: nu -> nu pi0 (pi0 -> gamma gamma )**
`/exp/uboone/data/users/jbateman/workdir/DarkNews/newSamplesWCepem/mcc910/checkout_MCC9.10_Run4b_v10_04_07_09_BNB_NC_pi0_overlay_surprise_reco2_hist.root`
This will produce  two showers that don't start from the same point, but could be traced back to a common point (where the pion decayed). This is because it takes a few cm for a photon to start showering, and until that point it is invisible.

### Making a graph (neural network)

#### Setup
I'll briefly walk through the GNN jupyter notebook I should have shared separately. A copy can also be found here: `/exp/uboone/app/users/jbateman/workdir/DarkNews/WCepem/wc_epem_ml/gnn_example.ipynb` 

This can be run on the MicroBooNE gpvms, or you can copy these slimmed ntuples onto the Imperial machines or onto your laptop:
`/exp/uboone/data/users/jbateman/workdir/DarkNews/Trident/SURPRISE/trident_SURPRISE_big_wirecell.root` - **660 MB**
`/exp/uboone/data/users/jbateman/workdir/DarkNews/newSamplesWCepem/mcc910/checkout_MCC9.10_Run4b_v10_04_07_09_BNB_NC_pi0_overlay_surprise_reco2_hist.root` - **2.7 GB**

Optional for validation: `/exp/uboone/data/users/jbateman/workdir/DarkNews/newSamplesWCepem/mcc910/checkout_MCC9.10_Run4b_v10_04_07_09_BNB_nu_overlay_surprise_reco2_hist.root` - **11 GB**

These **files only contain wire cell** reconstruction - If you want to explore LANTERN or Pandora, take a look at the samples listed on Redmine.

I used a clean python 3.13.13 environment to test this notebook, and installed the python libraries above via pip.

#### The Notebook
In the notebook, I first import the libraries that we'll be needing later. Then, I check whether a CUDA device is available (an nvidia GPU with CUDA set up). This should already be sorted on remote GPU nodes, but check what version of CUDA so it matches your torch install. If you have a modern macbook, you can use the ML accelerator built in by using the `mps` device. Otherwise you default to the CPU, which should be fine for this workload. 

Next, we locate the signal and background files, decide how many to events to load, then turn these root files into pandas dataframes that store the spacepoints as well as some useful truth and reconstructed information. 

Once those files are loaded, we visualise a signal and background event, and draw the associated K-nearest-neighbour graph for each event. If you have plotly installed, you can also plot in 3D rather than the XY and XZ plane views. 

Once you're happy with the graphs, we turn our pandas dataframe into a series of graphs and nodes, and store them in a torch_geometric "Data" object. Part of the processing involves giving the space point coordinates relative to the reconstructed neutrino vertex. This makes training the network a lot easier! We also split our signal+background sample into separate training, validation and testing sets at this stage.

Now we know the structure of our graphs, we can design a graph neural network to handle the data. I've defined a simple one with 3 layers of graph convolutions and a relatively low number of weights, but there is lots of room for tweaking the model!

With the model ready, its time to train it! There a number of hyper parameters to change here, but for our purposes they should be fine to leave as-is. 

Once training is done, we plot the loss and accuracy of the models. You may notice the validation is lower than training here, because training loss/accuracy is the average across the epoch, where the model is constantly improving, whereas validation is done at the end of the epoch. 

Finally we validate the model. I test two scenarios that should be basically identical:
1. We grab the weights from the epoch with lowest validation loss
2. We grab the weights from the epoch with highest validation accuracy

In both cases, we plot the ROC curve, look at the confusion matrix (raw and normalised)
and the score distribution itself.

One final validation stage I try is by running the model on nu overlay, to show that our model is good at rejecting a lot of backgrounds, not just NC Pi0s!

### Things to try
Once you're feeling comfortable with this model, a quick modification could be to turn the 3D representation into 2D. If the model works well on that 2D projection, it may be quickly transferable to the dark trident images.

# NTuples
## SURPRISE Samples
MicroBooNE has a number of versions of ntuples, the root files that contain all the reconstructed variables (such as the number of showers in an event, and each energy) as well as truth information where available. 

The latest version of ntuples are known as SURPRISE (**S**uper **U**nified **R**e-**P**rocessing **R**eally **I**mproving **S**election **E**fficiencies), called super unified because it contains 3 separate ways of reconstructing events in one file:
	pandora: reconstruct objects (showers, tracks) in separate 2D planes, then match objects across planes for 3D reconstruction
	wire cell: match clusters between planes to produce a 3D representation, then turn clusters/groups of clusters into reconstructed objects
	LANTERN: Deep-learning based reconstruction taking a similar 2D-to-3D approach to pandora. 

This unified reprocessing (Reco2) is quite computationally intensive so, as we discussed, you may want to stick to the 2D plane images, but I'll use them as our example.

[Redmine ](https://cdcvs.fnal.gov/redmine/projects/uboone-physics-analysis/wiki/MCC910_Samples) has a list of all completed SURPRISE samples, and they're split by:
1. BNB vs NuMI - data from diffferent beams, and simulations use different beam models
2. Overlay/DetVar/Data/Open Data
	1. Overlay: the collection of simulated samples used for background expectation. This may be "nu overlay" that allows for any neutrino interaction and should be representative of data overall, or a process-specific sample (just electron neutrinos (intrinsic nue) or NC Pi0s for example). These process-specific samples appear in "nu overlay", but with a dedicated sample we have a lot more statistics to play with.
	2. Det Var: these are modified overlay files, where different aspects of the detector simulation are varied (such as the attenuation of light within the detector) to give us systematics on our detector simulation.
	3. Data: this contains both beam on (our data) and beam off (used for getting a cosmic ray background prediction). Beam-on data is assumed blinded, so don't open them! Beam off is fair game as we need it for our background prediction.  
	4. Open Data: a small subset of the data taken by MicroBooNE, so that analysers can check that the simulation is behaving as expected. Its generally okay to look at open data.
3. Run number: MicroBooNE took data across 5 runs - each a period of around 9 months where both the BNB and NuMI beams were on  for MicroBooNE to take data. Right now most of Runs 4 and 5 is made in SURPRISE, and Runs 1, 2 and 3 are being processed.
4. uboonecode: Version of uboonecode used to do the reconstruction. Newer is better, but for our purposes it shouldn't have much of an effect. 
5. 
For any sample you're interested in, you'll want the to grab the path to a file on "persistent" `/pnfs/uboone/persistent/users/uboonepro/surprise/` or "data" (if available) `/exp/uboone/data/users/...`

Some files may say `wc_only` (only containing the wire cell tree, no pandora or lantern) or `wc processed` (wire cell only, with additional Wire Cell processing steps such as removing bad runs and duplicates, and getting additional BDT score predictions). These shouldn't be relevant to you. 
##  Looking at an ntuple
I generally use uproot to open and look at ntuples. VSCode has a root interpreter, and you can also open it up via the command line if you set up root in an sl7 container. 
The structure of an ntuple will be

```
nuselection;1 (pandora)
	NeutrinoSelectionFilter;1 
	SubRun;1 
shrreco3d;1 
	_energy_tree;1 
	_dedx_tree;1 
	_rcshr_tree;1 
wcpselection;1 (wire cell)
	T_eval;1 
	T_pot;1 
	T_PFeval;1 
	T_spacepoints;1 
	T_BDTvars;1 
	T_KINEvars;1 
lantern;1 (lantern!)
	EventTree;1 
	potTree;1
```
### Relevant branches
We'll just be looking in `wcpselection` for now, so it's here I'll focus. 
`
`T_eval` contains a lot of the core truth information such as interaction vertex, additional event weights, and run/subrun/event number.

`T_pot` has POT counting information for correctly normalising your histograms later.

`T_PFeval` contains much of the per-particle reconstructed and truth information, such as the energy of a shower 

`T_spacepoints` is perhaps self explanatory, containing all the space points from each stage of wire cell clustering. The highest level/final version space points are `Trecchargeblob_spacepoints_<x/y/z/q>` and its these we'll be using.

`T_BDTvars` contains a number of different BDT scores to classify an event as muon neutrino-like, electron neutrino-like, as well as pi0-related scores.

`T_KINEvars` contains the reconstructed high level kinematics of objects within the event. We can use branches from here to check if there are any reconstructed protons for example.

Each of these trees contain a number of useful branches. Those that we use in the notebook have a comment next to them to briefly explain what they mean.
# Graphs and GNNs

There are many great online resources for explaining what a graph *is*. A crash course for graphs relevant for GNNs can be found here:
[https://lightning.ai/docs/pytorch/stable/notebooks/course_UvA-DL/06-graph-neural-networks.html#Graph-Neural-Networks](https://lightning.ai/docs/pytorch/stable/notebooks/course_UvA-DL/06-graph-neural-networks.html#Graph-Neural-Networks)

Wikipedia is also a great starting point for the mathematical theory underpinning.
https://en.wikipedia.org/wiki/Graph_(discrete_mathematics)

Essentially we represent the connections between $N$ space points ("nodes") in a $N\times N$ matrix, from which we can make a list of "edges", which nodes are connected together. We then use this list of connections to decide what 
