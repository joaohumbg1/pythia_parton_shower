# pythia_parton_shower


This code is a sample of the work I did for my dissertation, "Machine Learning in QCD Jets". The goal was to look at a Monte Carlo event generator called Pythia, used to simulate events in particle physics, and compare the effects of two mechanisms (called Simple Shower and Vincia) on the jet substructure (jets a formal definition of the beams of particles which are formed when two particles collide) using neural networks. Ultimately, we would like to compare the jets created by these event generators with real particles, but this wasn't done due to time constraints.

In order to make this comparison, we did the following:
1) Generated the events with Pythia using the Vincia parton shower, as is done in folder jetReco. We stored relevant jet substructure observables in a ROOT file (ROOT is an object-oriented computer program and library developed by CERN, widely used in scientific analysis).
2) Generate events with similar settings, but using the Simple Shower, and store it in a different ROOT file.
3) We convert the .root files into .csv using the programs tree2csv.C and tree2csv_nn.C, in folder neural_network (the former stores only the most relevant observables, in terms of physical relevance and observable difference; the latter stores all variables, in order to plot them; writing these two programs was much easier than manipulating large Pandas DataFrames).
4) We visualize the relevant information by reading the .csv and making plots using the Jupyter notebook "quantities plotter.ipynb".
5) We train a neural network neural_network_classifier in the following way:
   i)   Read the .csv files and create a pandas DataFrame with our data
   ii)  Assign a label to the data: 0 for Simple Shower, 1 for Vincia
   iii) Train a neural network to attempt to predict the label of a given event based on the jet substructure observables.
6) We attempt to optimize hyperparamters using optuned.py.

The dissertation moves around attempting to find ideal physical settings and improving the neural network. This is the final set of settings we tried.
