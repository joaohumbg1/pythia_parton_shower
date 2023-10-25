/// \file
/// \ingroup tutorial_tree
/// \notebook
/// TTreeReader simplest example.
///
/// Read data from hsimple.root (written by hsimple.C)
///
/// \macro_code
///
/// \author Anders Eie, 2013


#include <TFile.h>
#include <TH1.h>
#include <TTreeReader.h>
#include <TTreeReaderValue.h>
#include <TTreeReaderArray.h>
#include <vector>
#include <iostream>

using namespace std;


void tree2csv() {

   string parton_shower [] = {"vincia", "simple_shower"};
   
   // Do the procedure for both parton shower models
   for (int i = 0; i < *(&parton_shower + 1) - parton_shower ; i++) { // That weird text is just a way expressing the length of the "use" array
	   // Create the output file
	   ofstream FileEvent (parton_shower[i] + ".csv");

	   // Open the file containing the tree.
	   string o_file = parton_shower[i] + "_tree_CA_hadron_pt_hat500_100TeV.root";
	   const char * open_file = o_file.c_str();
	   
	   auto myFile = TFile::Open(open_file);
	   if (!myFile || myFile->IsZombie()) {
	      return;
	   }
	   
	   // Create a TTreeReader, reading the TTree T in the file we've opened
	   TTreeReader fReader("T",myFile);


	   // Read the values/arrays
	   TTreeReaderValue<Int_t> event = {fReader, "event"};
	   TTreeReaderValue<Int_t> iterations = {fReader, "iterations"};
	   
	   TTreeReaderArray<Double_t> Delta 	= {fReader, "Delta"};
	   TTreeReaderArray<Double_t> kt	= {fReader, "kt"};
	   TTreeReaderArray<Double_t> z 	= {fReader, "z"};
	   TTreeReaderArray<Double_t> k	= {fReader, "k"};
	   TTreeReaderArray<Int_t> soft_drops_to_decluster = {fReader, "soft_drops_to_decluster"};
	   TTreeReaderArray<Int_t> j2_multiplicity = {fReader, "j2_multiplicity"};
	   TTreeReaderArray<Int_t> j1_multiplicity = {fReader, "j1_multiplicity"};
	   
	   TTreeReaderValue<Double_t> pT_j1     = {fReader, "pT_j1"};   
	   TTreeReaderValue<Double_t> pT_j2     = {fReader, "pT_j2"};   
	   TTreeReaderValue<Double_t> eta_j1    = {fReader, "eta_j1"};   
	   TTreeReaderValue<Double_t> eta_j2    = {fReader, "eta_j2"};   
	   TTreeReaderValue<Double_t> phi_j1    = {fReader, "phi_j1"};   
	   TTreeReaderValue<Double_t> phi_j2    = {fReader, "phi_j2"};   
	   TTreeReaderValue<Double_t> delta_phi = {fReader, "delta_phi"};   
	   
	   TTreeReaderValue<Double_t> pT_j1_cut     = {fReader, "pT_j1_cut"};   
	   TTreeReaderValue<Double_t> pT_j2_cut     = {fReader, "pT_j2_cut"};   
	   TTreeReaderValue<Double_t> eta_j1_cut    = {fReader, "eta_j1_cut"};   
	   TTreeReaderValue<Double_t> eta_j2_cut    = {fReader, "eta_j2_cut"};   
	   TTreeReaderValue<Double_t> phi_j1_cut    = {fReader, "phi_j1_cut"};   
	   TTreeReaderValue<Double_t> phi_j2_cut    = {fReader, "phi_j2_cut"};   
	   TTreeReaderValue<Double_t> delta_phi_cut = {fReader, "delta_phi_cut"};   

	   // Select which variables to use. Be very careful with the consistency of these names further down.
	   //string use [] = { "ln_z1", "ln_Delta1", "ln_k1", "ln_z2", "ln_Delta2", "ln_k2","ln_z3", "ln_Delta3", "ln_k3" };
	   string use [] = { "ln_Delta1", "ln_k1", "ln_Delta2", "ln_k2", "ln_Delta3", "ln_k3", "ln_Delta4", "ln_k4", "pT_j1", "pT_j2", "eta_j1", "eta_j2", "phi_j1", "phi_j2", "delta_phi", "pT_j1_cut", "pT_j2_cut", "eta_j1_cut", "eta_j2_cut", "phi_j1_cut", "phi_j2_cut", "delta_phi_cut"};
	   
	   // Add strings to the header of the csv.
	   string header;
	   
	   for (int i = 0; i < *(&use + 1) - use ; i++) { // That weird text is just a way expressing the length of the "use" array
	   	if (i == 0) { header = header + use[i]; }
	   	else { header = header + ',' + use[i]; }
	   	}
	   FileEvent << header << endl;
	   
	   // Loop over all entries of the TTree or TChain.
	   
	   // Very careful when adding things here!!! They must be in order as the use[] string.
	   while (fReader.Next()) {
	   
	   	if (*iterations >= 4) {
	   	// Remove instances where the jet can't be declustered
	   		
	   		string newline;


	      		for (int i = 0; i < *(&use + 1) - use ; i++) {
	      			if (i > 0) {newline = newline + ',';}
	      			
	      			if (use[i] == "event") 		{ newline += to_string(*event)			;}
	      			if (use[i] == "*iterations")		{ newline += to_string(*iterations)		;}
	      			   
	      			// 1st declustering quantities  
	      			if (use[i] == "ln_z1")			{ newline += to_string(log (z[0]) )		;}      			
	      			if (use[i] == "ln_Delta1")		{ newline += to_string(log (1/Delta[0] ))	;}
	      			//if (use[i] == "ln_Delta1")		{ newline += to_string(-1/log (Delta[0] ))	;} 
	      			if (use[i] == "ln_k1")			{ newline += to_string(log (k[0]) )		;}
	      			if (use[i] == "j1_multiplicity1")	{ newline += to_string(j1_multiplicity[0])	;}
	      			if (use[i] == "j2_multiplicity1")	{ newline += to_string(j2_multiplicity[0])	;}

				// 2nd declustering quantities
	      			if (use[i] == "ln_z2")			{ newline += to_string(log (z[1]) )		;}
				if (use[i] == "ln_Delta2")		{ newline += to_string(log (1/Delta[1] ))	;}
				//if (use[i] == "ln_Delta2")		{ newline += to_string(-1/log(Delta[1] ))	;}
				if (use[i] == "ln_k2")			{ newline += to_string(log (k[1]) )		;}
				if (use[i] == "j1_multiplicity2")	{ newline += to_string(j1_multiplicity[1])	;}
				if (use[i] == "j2_multiplicity2")	{ newline += to_string(j2_multiplicity[1])	;}
				
				// 3rd declustering quantities
	      			if (use[i] == "ln_z3")			{ newline += to_string(log (z[2]) )		;}
				if (use[i] == "ln_Delta3")		{ newline += to_string(log (1/Delta[2] ))	;}
				//if (use[i] == "ln_Delta3")		{ newline += to_string(-1 / log (Delta[2] ))	;}
				if (use[i] == "ln_k3")			{ newline += to_string(log (k[2]))		;}
	      			if (use[i] == "j1_mutliplicity3")	{ newline += to_string(j1_multiplicity[2])	;}
	      			if (use[i] == "j2_mutliplicity3")	{ newline += to_string(j2_multiplicity[2])	;}      			
				
	      			// 4th declustering quantities
	      			if (use[i] == "ln_z4")			{ newline += to_string(log (z[3]) )		;}
				if (use[i] == "ln_Delta4")		{ newline += to_string(log (1/Delta[3] ))	;}
				//if (use[i] == "ln_Delta4")		{ newline += to_string(-1/log (Delta[3] ))	;}
				if (use[i] == "ln_k4")			{ newline += to_string(log (k[3]) ) 		;}
	      			if (use[i] == "j1_mutliplicity4")	{ newline += to_string(j1_multiplicity[3])	;}
	      			if (use[i] == "j2_mutliplicity4")	{ newline += to_string(j2_multiplicity[3])	;} 
	      			
	      			// Control quantiies
	      			if (use[i] == "pT_j1")			{ newline += to_string(*pT_j1)			;}
				if (use[i] == "pT_j2")			{ newline += to_string(*pT_j2)			;}
				if (use[i] == "eta_j1")			{ newline += to_string(*eta_j1)		;}
				if (use[i] == "eta_j2")			{ newline += to_string(*eta_j2)		;}
			 	if (use[i] == "phi_j1")			{ newline += to_string(*phi_j1)		;}
				if (use[i] == "phi_j2")			{ newline += to_string(*phi_j2)		;}
				if (use[i] == "delta_phi")		{ newline += to_string(*delta_phi)		;}
				if (use[i] == "pT_j1_cut")		{ newline += to_string(*pT_j1_cut)		;}
				if (use[i] == "pT_j2_cut")		{ newline += to_string(*pT_j2_cut)		;}
				if (use[i] == "eta_j1_cut")		{ newline += to_string(*eta_j1_cut)		;}
				if (use[i] == "eta_j2_cut")		{ newline += to_string(*eta_j2_cut)		;}
			 	if (use[i] == "phi_j1_cut")		{ newline += to_string(*phi_j1_cut)		;}
				if (use[i] == "phi_j2_cut")		{ newline += to_string(*phi_j2_cut)		;}
				if (use[i] == "delta_phi_cut")		{ newline += to_string(*delta_phi_cut)		;}

				
	      		}

	      	FileEvent << newline << endl;

		      
		      // Set all quantities to -1 (in case of doubles) or 0 (integers) so that, in the next event, if said quantities can't be calculated, they'll default to -1/0 and we'll see they can't be calculated.
		     // z[0] = -1., z[1] = -1., z[2] = -1., Delta[0] = -1., Delta[1] = -1., Delta[2] = -1., k[0] = -1., k[1] = -1, k[2] = -1.;
		      Delta[0] = -1., Delta[1] = -1., Delta[2] = -1., k[0] = -1., k[1] = -1, k[2] = -1.;
		      
		      }
	      }

	   //Close file
	   FileEvent.close();
   }
}
