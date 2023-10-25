//
//  test.cpp
//  
//

/*
 * This program is meant to run Monte Carlo event simulations,
 * with pythia
 * It reconstruct the jets and save its kinematic and subjets 
 * information
 */

#include <stdio.h>
#include <cstdlib> // for exit function
#include <math.h>
#include <complex>
#include <string.h>
#include <sstream>
#include <iostream>
#include <ctime>

#include "Pythia8/Pythia.h"
#include "fastjet/PseudoJet.hh"
#include "fastjet/JetDefinition.hh"
#include "fastjet/ClusterSequence.hh"
#include "fastjet/tools/Recluster.hh"

#include "TH1D.h"
#include <TTree.h>
#include "TFile.h"
#include <TStyle.h>
#include "TCanvas.h"

using namespace std;
using namespace fastjet;
using namespace Pythia8;

// -------------------------------------------------------
// MAIN ROUTINE
// -------------------------------------------------------
int main(int argc, char* argv[]) {

   // Counting compiling time
   time_t start_time = time(0);
	// Variables
	// -------------------------------------------------------
   // pythia variables
   Pythia pythia;
   pythia.readFile("input_pythia.cmnd");
   int nEvent = pythia.mode("Main:numberOfEvents");

   // Jet settings 
   double jet_radius = 1;
   double jet_ptmin = 30.;
   double jet_ptmin_leading = 1000.;
   double p = 0.5; // p is the power of the generalized kt algorithm.
   //double jet_ptmin = 800.;
   double jet_eta = 2.0;
   JetDefinition jet_def(antikt_algorithm, jet_radius);
   //JetDefinition new_jet_def(genkt_algorithm, jet_radius+0.5, p);
   JetDefinition new_jet_def(cambridge_algorithm, jet_radius+0.5);
   //JetDefinition new_jet_def(kt_algorithm, jet_radius+0.5);
   Recluster recluster(new_jet_def);
   

   int max_iterations=10;
   double zz;
   int event;
   int j2_multiplicity[max_iterations], j1_multiplicity[max_iterations];
   int iterations;
   int soft_drops;
   int soft_drops_to_decluster[max_iterations];
   double Delta[max_iterations], kt[max_iterations], k[max_iterations], z[max_iterations];
   double ln_Delta[max_iterations], ln_kt[max_iterations], ln_k[max_iterations];
   double pT_j1, pT_j2, eta_j1, eta_j2, phi_j1, phi_j2, delta_phi;
   double pT_j1_cut, pT_j2_cut, eta_j1_cut, eta_j2_cut, phi_j1_cut, phi_j2_cut, delta_phi_cut;

   // Root Variables
   TFile* file = new TFile("vincia_tree_CA_hadron_pt_hat800_100TeV.root", "recreate");
   TTree* T    = new TTree("T","event Tree");
   
   // Set up branches for useful quantities
   T->Branch("event",&event,"event/I");
   T->Branch("iterations",&iterations,"iterations/I");
   T->Branch("soft_drops",&soft_drops,"soft_drops/I");
   T->Branch("z",&z,"z[iterations]/D");
   T->Branch("Delta",&Delta,"Delta[iterations]/D");
   T->Branch("kt",&kt,"kt[iterations]/D");
   T->Branch("k" ,&k ,"k[iterations]/D");
   T->Branch("ln_Delta",&ln_Delta,"ln_Delta[iterations]/D");
   T->Branch("ln_kt",&ln_kt,"ln_kt[iterations]/D");
   T->Branch("ln_k" ,&ln_k, "ln_k[iterations]/D");
   T->Branch("soft_drops_to_decluster",&soft_drops_to_decluster,"soft_drops_to_decluster[iterations]/I");
   T->Branch("j1_multiplicity",&j1_multiplicity,"j1_multiplicity[iterations]/I");
   T->Branch("j2_multiplicity",&j2_multiplicity,"j2_multiplicity[iterations]/I");
   
   // Control structures. Ensure that cuts are being applied. These quantities refer to the leading jet
   T->Branch("pT_j1"         , &pT_j1         , "pT_j1/D");
   T->Branch("pT_j2"         , &pT_j2         , "pT_j2/D");
   T->Branch("eta_j1"        , &eta_j1        , "eta_j1/D");
   T->Branch("eta_j2"        , &eta_j2        , "eta_j2/D");
   T->Branch("phi_j1"        , &phi_j1        , "phi_j1/D");
   T->Branch("phi_j2"        , &phi_j2        , "phi_j2/D");
   T->Branch("delta_phi"     , &delta_phi     , "delta_phi/D");
   T->Branch("pT_j1_cut"     , &pT_j1_cut     , "pT_j1_cut/D");
   T->Branch("pT_j2_cut"     , &pT_j2_cut     , "pT_j2_cut/D");
   T->Branch("eta_j1_cut"    , &eta_j1_cut    , "eta_j1_cut/D");
   T->Branch("eta_j2_cut"    , &eta_j2_cut    , "eta_j2_cut/D");
   T->Branch("phi_j1_cut"    , &phi_j1_cut    , "phi_j1_cut/D");
   T->Branch("phi_j2_cut"    , &phi_j2_cut    , "phi_j2_cut/D");
   T->Branch("delta_phi_cut" , &delta_phi_cut , "delta_phi_cut/D");
   // Setting environment
   // Pythia
   pythia.init();
   
   // Output the type of algorithm used
   cout << "Original jets obtained with " << jet_def.description() << endl;
   cout << "Jets reclustered with " << new_jet_def.description() << "\n\n\n" << endl;
   
   
	// Analysis of the tree entries
	// -------------------------------------------------------
	cout << " ------ Start of program ------ " << endl;
      for (int iEvent = 0; iEvent < nEvent; ++iEvent){
      
      // Generate Pythia event
      if (!pythia.next() ) continue;
      //double weight = pythia.info.weight();
      
      // Increase the number of events by 1;
      event = iEvent;
      
      
      // Set the number of declusterings to zero
      iterations = 0;
      
      // Set initial value of soft drops to 0, and other quantities to -1 in order to allow debugging.
      for (int i = 0; i< max_iterations; ++i){
		soft_drops_to_decluster[i] = 0;
		z[i] = -1.;
		Delta[i] = -1.;
		kt[i] = -1.;
		k [i] = -1.;
		soft_drops += 1;
		}
			 

      // final state particles
      vector<PseudoJet> particles;
      for (int iPart = 0; iPart < pythia.event.size(); ++iPart){
         if (pythia.event[iPart].isFinal()){
            PseudoJet jetAux (pythia.event[iPart].px(), pythia.event[iPart].py(),
                              pythia.event[iPart].pz(), pythia.event[iPart].e());
            particles.push_back(jetAux);
         }
      }
    
      // Set these quantities like so, in order to exclude them later if we can't determine them because we ran out of parents. 


      // final list of jets
      Selector cuts = SelectorAbsEtaMax(jet_eta)*SelectorPtMin(jet_ptmin);
      ClusterSequence cs(particles, jet_def);
      vector<PseudoJet> jets = cs.inclusive_jets();
      vector<PseudoJet> jets_uncut = jets;
      jets = cuts(jets);
      jets = sorted_by_pt(jets);
      
      jets_uncut = sorted_by_pt(jets_uncut);
      pT_j1     = jets_uncut[0].pt();
      pT_j2     = jets_uncut[1].pt();
      eta_j1    = jets_uncut[0].eta();
      eta_j2    = jets_uncut[1].eta();
      phi_j1    = jets_uncut[0].phi();
      phi_j2    = jets_uncut[1].phi();
      delta_phi = fabs(phi_j1 - phi_j2);
      delta_phi = (delta_phi > M_PI ? 2*M_PI - delta_phi : delta_phi);
      
      double phi1_semicut;
      double phi2_semicut;
      double delta_phi_semicut;
      
      if (jets.size() > 1) {
      
      phi1_semicut = jets[0].phi();
      phi2_semicut = jets[1].phi();
      
      delta_phi_semicut = fabs(phi1_semicut - phi2_semicut);
      delta_phi_semicut = (delta_phi_semicut > M_PI ? 2*M_PI - delta_phi_semicut : delta_phi_semicut);
      }
      
      //delta_phi_semicut = fabs(phi1_semicut - phi2_semicut);

      // Get only leading jet 
      if ((jets.size() > 1) and (jets[0].pt() >= jet_ptmin_leading) and (delta_phi_semicut > 2./3. * M_PI) ){
      
         pT_j1_cut     = jets[0].pt();
         pT_j2_cut     = jets[1].pt();
         eta_j1_cut    = jets[0].eta();
         eta_j2_cut    = jets[1].eta();
         phi_j1_cut    = jets[0].phi();
         phi_j2_cut    = jets[1].phi();
         delta_phi_cut = fabs(phi_j1 - phi_j2);
         delta_phi_cut = (delta_phi > M_PI ? 2*M_PI - delta_phi : delta_phi);
         
	 PseudoJet jj = recluster(jets[1]);
         //Establish the jet jj, which is to be divided into 2 subjets, j1 and j2.
	fastjet::PseudoJet j1;  // subjet 1 (largest pt)
	fastjet::PseudoJet j2;  // subjet 2 (smaller pt)

   	
	int i = 0;
	

	 // Unclustering jet and recording jet history
	 // This is, here, for each jet, we remove the softest subjet if the soft drop condition isn't verified
	 while(jj.has_parents(j1,j2) and i<max_iterations){
	    
	    // put subjet1 as the one with largest pt
	    if(j1.perp() < j2.perp()) std::swap(j1,j2);
	    
	    // Soft Drop condition
	    zz = j2.perp()/(j1.perp() + j2.perp());
	    
	    Delta[i] = -1.;
	    
	    // Applies Soft drop condition and save kinematic information
	    if (zz >= 0.1){
	      
	      // Get the number of components of the subleading jet
	      vector<PseudoJet> j2_constituents = j2.constituents(); 
      	      j2_multiplicity[i] = j2_constituents.size();
      	      vector<PseudoJet> j1_constituents = j1.constituents();
      	      j1_multiplicity[i] = j1_constituents.size();
	      
	      if (i == 0) {
	      Delta[i] = -1;
	      }
	      
	      // Calculate and add relevant quantities to tree
	      z[i] = zz;
	      Delta[i] = j1.delta_R(j2);
	      kt[i] = j2.pt() * Delta[i];
	      k[i] = zz*Delta[i];
	      ln_Delta[i] = log(1/Delta[i]);
	      ln_kt[i] = log(kt[i]);
	      ln_k[i]  = log(k [i]);
	      iterations = i+1;
	      
      
	      
	      
	      //Decluster again now
	      i++;
	      
	      //cout << "i = " << iterations << ", z = " << z[i] << ", Delta = " << Delta[i] << ", kt = " << kt[i] << endl;
	      //cout << "found pair of subjets" << endl;
	      ;
	      
	      
	      }
	    
	    
	    else {
	      soft_drops_to_decluster[i] ++;
	      }
	    

	   // continue unclustering
	   jj=j1;
	   }
	   
      }

      particles.clear();
      jets.clear();
      T->Fill();
   }

   // Print out statistics and final rescalings
   pythia.stat();

   T->Print();
   T->Write();
   
   delete file;
   
// Info about program's running time.
time_t finish_time = time(0);
char* finish_time_str = ctime(&finish_time);
int elapsed_time = difftime(finish_time,start_time);

cout << "\n\n\n\n" << endl;
cout << "Program took " << elapsed_time << " seconds to compile" << endl;
cout << "Finishing at " << finish_time_str << endl;


cout << " ------ End of program ------ " << endl;

return 0;
}



