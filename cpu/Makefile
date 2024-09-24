CC=gcc
CFLAGS = -I.
DEPS = util.h

%.o : %.c $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

mergesort: mergesort.o util.o
	$(CC) -g -o mergesort mergesort.o util.o