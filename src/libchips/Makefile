CFLAGS ?= -Wall -Werror -Os
LDFLAGS ?= $(shell pkg-config gdal --libs) -lstdc++ -lpthread
GDALCFLAGS ?= $(shell pkg-config gdal --cflags)

all: libchips.so.1.1 libchips_ce.so.1.1

%.o: %.c
	$(CC) $(GDALCFLAGS) $(CFLAGS) -fPIC $< -c -o $@

chips_ce.o: chips.c
	$(CC) $(GDALCFLAGS) $(CFLAGS) -DCHAMPION_EDITION=1 -fPIC $< -c -o $@

%.o: %.cpp
	$(CXX) $(GDALCFLAGS) $(CXXFLAGS) $(CFLAGS) -fPIC $< -c -o $@

libchips.so.1.1: chips.o globals.o reader.o
	$(CC) $(CFLAGS) $^ $(LDFLAGS) -shared -o $@
	strip $@

libchips_ce.so.1.1: chips_ce.o globals.o reader.o
	$(CC) $(CFLAGS) $^ $(LDFLAGS) -shared -o $@
	strip $@

main: main.c chips.c globals.c reader.c
	$(CC) $(GDALCFLAGS) $(CFLAGS) -I . \
	main.c chips.c globals.c reader.c \
	$(shell pkg-config gdal --libs) -lpthread -o $@

clean:
	rm -f *.o

cleaner: clean

cleanest: cleaner
	rm -f libchips*.so*
	rm -f main
