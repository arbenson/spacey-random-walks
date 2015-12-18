CXX = g++

OPT := -O3 -std=c++11
CFLAGS := $(OPT) $(DEBUG) $(INCLUDES) $(DEFINES) $(DEFINES)

SRC = common_srs.cpp

OBJECTS = $(SRC:.cpp=.o)

.PHONY : default
default : all

.PHONY : all

simulate: simulate_srs.cpp $(OBJECTS)
	$(CXX) $(CFLAGS) $< $(LDFLAGS) $(SNAP_OBJ) $(OBJECTS) $(LDLIBS) -o $@

learn: learn_srs.cpp $(OBJECTS)
	$(CXX) $(CFLAGS) $< $(LDFLAGS) $(SNAP_OBJ) $(OBJECTS) $(LDLIBS) -o $@

SCRIPTS = simulate \
          learn

all: $(SCRIPTS)

%.o: %.cpp
	$(CXX) -c $(CFLAGS) $<

.PHONY : clean
clean :
	rm -rf *.o *~ $(SCRIPTS)
