BOOST_ROOT ?= /usr/include
CFLAGS ?= -Wall -Werror -Ofast -mtune=native
LDFLAGS ?= -lstdc++ -lm
CXXFLAGS ?= -std=c++11 -I$(BOOST_ROOT)

all: libmedial.so

%.o: %.c
	$(CC) $(CFLAGS) -fPIC $< -c -o $@

%.o: %.cpp %.h
	$(CXX) $(CXXFLAGS) $(CFLAGS) -fPIC $< -c -o $@

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(CFLAGS) -fPIC $< -c -o $@

libmedial.so: medial.o
	$(CC) $(CFLAGS) $^ $(LDFLAGS) -shared -o $@
	strip $@

main: main.c medial.o
	$(CC) $(CFLAGS) -I . main.c medial.o $(LDFLAGS) -o $@

clean:
	rm -f *.o

cleaner: clean

cleanest: cleaner
	rm -f liboverlaps.so
	rm -f main
