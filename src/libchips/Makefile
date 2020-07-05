CFLAGS ?= -Wall -Werror -Os
LDFLAGS ?= $(shell pkg-config gdal --libs) -lstdc++ -lpthread
GDALCFLAGS ?= $(shell pkg-config gdal --cflags)

all: libchips.so libchips_ce.so

%.o: %.c
	$(CC) $(GDALCFLAGS) $(CFLAGS) -fPIC $< -c -o $@

chips_ce.o: chips.c
	$(CC) $(GDALCFLAGS) $(CFLAGS) -DCHAMPION_EDITION=1 -fPIC $< -c -o $@

%.o: %.cpp
	$(CXX) $(GDALCFLAGS) $(CXXFLAGS) $(CFLAGS) -fPIC $< -c -o $@

libchips.so: chips.o
	$(CC) $(CFLAGS) $^ $(LDFLAGS) -shared -o $@
	strip $@

libchips_ce.so: chips_ce.o
	$(CC) $(CFLAGS) $^ $(LDFLAGS) -shared -o $@
	strip $@

main: main.c chips.c chips.h
	$(CC) $(GDALCFLAGS) $(CFLAGS) -I . main.c chips.c $(shell pkg-config gdal --libs) -lpthread -o $@

clean:
	rm -f *.o

cleaner: clean

cleanest: cleaner
	rm -f libchips*.so
	rm -f main
