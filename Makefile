CC      := gcc
CFLAGS  ?= -O3 -march=native -fopenmp -Wall -Wextra -std=c11 -Iinclude
LDFLAGS ?= -fopenmp -lm

SRC := src/semiring.c src/ternary.c src/ternary_fast.c src/util.c
OBJ := $(SRC:.c=.o)

all: build/test build/bench

build:
	mkdir -p build

%.o: %.c include/aim.h
	$(CC) $(CFLAGS) -c $< -o $@

build/test: tests/test.c $(OBJ) | build
	$(CC) $(CFLAGS) $< $(OBJ) -o $@ $(LDFLAGS)

build/bench: bench/bench.c $(OBJ) | build
	$(CC) $(CFLAGS) $< $(OBJ) -o $@ $(LDFLAGS)

test: build/test
	./build/test

bench: build/bench
	./build/bench

clean:
	rm -rf build src/*.o

.PHONY: all test bench clean
