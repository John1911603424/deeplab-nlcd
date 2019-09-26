CFLAGS ?= -Wall -Werror -O
LDFLAGS ?= $(shell pkg-config gdal --libs) -l:libstdc++.a -lpthread
GDALCFLAGS ?= $(shell pkg-config gdal --cflags)

all: libchips.so

%.o: %.c
	$(CC) $(GDALCFLAGS) $(CFLAGS) -fPIC $< -c -o $@

libchips.so: chips.o
	$(CC) $(CFLAGS) $^ $(LDFLAGS) -shared -o $@
	strip $@

clean:
	rm -f *.o

cleaner: clean

cleanest: cleaner
	rm -f libchips.so
