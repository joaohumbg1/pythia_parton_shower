
################################################################
ROOTLDFLAGS := $(shell root-config --libs --ldflags)
ROOTCFLAGS := $(shell root-config --cflags)
ROOTGLIBS := $(shell root-config --glibs --libs)

FASTJETCFLAGS := $(shell fastjet-config --cxxflags )
FASTJETLIBS := $(shell fastjet-config --libs)

PYTHIACFLAGS := $(shell pythia8-config --cxxflags)
PYTHIALIBS := $(shell pythia8-config --libs)


################################################################
PLATFORM = $(shell root-config --platform)
$(info PLATFORM is $(PLATFORM)) 


################################################################
CFLAGS = $(ROOTCFLAGS) $(FASTJETCFLAGS) $(PYTHIACFLAGS) 
LDFLAGS = $(ROOTLDFLAGS) $(FASTJETLIBS) $(ROOTGLIBS) $(PYTHIALIBS) 
GCCFLAGS = -g -Wall -fPIC -std=c++11
CXX = g++
RM = rm -f


################################################################
SOURCES =$(wildcard *.cpp)
OBJECTS = $(SOURCES:.cc=.o)
EXECUTABLE = jetReco
CLEANOBJS = $(wildcard *.o)

################################################################
all: $(EXECUTABLE) 
run: $(EXECUTABLE) exe

.cc.o:
	@echo  "\nCompiling " $@ 
	$(CXX) -c $(GCCFLAGS) $(CFLAGS) $^ -o $@

$(EXECUTABLE): $(OBJECTS)
		@echo "\nMaking executable " $@ " ..."
		@$(CXX) $^ -o $@ $(GCCFLAGS) $(CFLAGS) $(LDFLAGS) 
		@echo "\ndone"

clean:          
		@echo "\nclean" 
		@rm -fv $(CLEANOBJS)

distclean:          
		@echo "\nclean" 
		@rm -fv $(EXECUTABLE) $(CLEANOBJS)

exe:
		@echo "\nrun" 
		@./$(EXECUTABLE)

