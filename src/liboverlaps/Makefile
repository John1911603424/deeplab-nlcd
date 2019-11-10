BOOST_ROOT ?= /usr/include
CFLAGS ?= -Wall -Werror -Ofast -mtune=native
CXXFLAGS ?= -std=c++11 -I$(BOOST_ROOT)

all: liboverlaps.so

%.o: %.c
	$(CC) $(CFLAGS) -fPIC $< -c -o $@

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(CFLAGS) -fPIC $< -c -o $@

liboverlaps.so: trees.o
	$(CC) $(CFLAGS) $^ $(LDFLAGS) -shared -o $@
	strip $@

main: main.c trees.cpp trees.h
	$(CC) $(CFLAGS) -I . $^ -lstdc++ -o $@

clean:
	rm -f *.o

cleaner: clean

cleanest: cleaner
	rm -f liboverlaps.so
	rm -f main
