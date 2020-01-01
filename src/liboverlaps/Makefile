BOOST_ROOT ?= /usr/include
CFLAGS ?= -Wall -Werror -Ofast -mtune=native
LDFLAGS ?= -lstdc++
CXXFLAGS ?= -std=c++11 -I$(BOOST_ROOT)

all: liboverlaps.so

%.o: %.c
	$(CC) $(CFLAGS) -fPIC $< -c -o $@

%.o: %.cpp %.h
	$(CXX) $(CXXFLAGS) $(CFLAGS) -fPIC $< -c -o $@

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(CFLAGS) -fPIC $< -c -o $@

liboverlaps.so: trees.o
	$(CC) $(CFLAGS) $^ $(LDFLAGS) -shared -o $@
	strip $@

main: main.c trees.o
	$(CC) $(CFLAGS) main.c trees.o $(LDFLAGS) -I . -o $@

clean:
	rm -f *.o

cleaner: clean

cleanest: cleaner
	rm -f liboverlaps.so
	rm -f main
