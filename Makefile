CXX = g++

OPT := -O3 -std=c++11
CFLAGS := $(OPT) $(DEBUG) $(INCLUDES) $(DEFINES) $(DEFINES)

SRC = common_srw.cpp

OBJECTS = $(SRC:.cpp=.o)

.PHONY : default
default : all

.PHONY : all

sim: simulate_srw.cpp $(OBJECTS)
	$(CXX) $(CFLAGS) $< $(LDFLAGS) $(SNAP_OBJ) $(OBJECTS) $(LDLIBS) -o $@

learn: learn.cpp $(OBJECTS)
	$(CXX) $(CFLAGS) $< $(LDFLAGS) $(SNAP_OBJ) $(OBJECTS) $(LDLIBS) -o $@

SCRIPTS = sim \
          learn

all: $(SCRIPTS)

%.o: %.cpp
	$(CXX) -c $(CFLAGS) $<

.PHONY : clean
clean :
	rm -rf *.o *~ $(SCRIPTS)
