CC=gcc
CFLAGS = -I.
DEPS = util.h
OBJ = mergesort.o util.o

%.o : %.c $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

mergesort: $(OBJ)
	$(CC) -g -o mergesort $(OBJ)

.PHONY: clean

clean:
	rm -f $(OBJ) mergesort