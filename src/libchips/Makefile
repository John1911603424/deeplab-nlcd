CFLAGS ?= -Wall -Werror -Os
LDFLAGS ?= $(shell pkg-config gdal --libs) -lstdc++ -lpthread
GDALCFLAGS ?= $(shell pkg-config gdal --cflags)

all: libchips.so

%.o: %.c
	$(CC) $(GDALCFLAGS) $(CFLAGS) -fPIC $< -c -o $@

%.o: %.cpp
	$(CXX) $(GDALCFLAGS) $(CXXFLAGS) $(CFLAGS) -fPIC $< -c -o $@

libchips.so: chips.o
	$(CC) $(CFLAGS) $^ $(LDFLAGS) -shared -o $@
	strip $@

main: main.c chips.h chips.c
	$(CC) $(GDALCFLAGS) $(CFLAGS) -I . $^ $(shell pkg-config gdal --libs) -lpthread -o $@

clean:
	rm -f *.o

cleaner: clean

cleanest: cleaner
	rm -f libchips.so
	rm -f main
