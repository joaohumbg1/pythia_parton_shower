# pythia_parton_shower


This code is a sample of the work I did for my dissertation, "Machine Learning in QCD Jets". The goal was to look at a Monte Carlo event generator called Pythia, used to simulate events in particle physics, and compare the effects of two mechanisms (called Simple Shower and Vincia) on the jet substructure (jets a formal definition of the beams of particles which are formed when two particles collide) using neural networks. Ultimately, we would like to compare the jets created by these event generators with real particles, but this wasn't done due to time constraints.

In order to make this comparison, we did the following:
1) Generated the events with Pythia, as is done in folder jetReco. We stored relevant jet substructure observables in a ROOT file (ROOT is an object-oriented computer program and library developed by CERN, widely used in scientific analysis).
