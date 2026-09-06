CC      := gcc
CFLAGS  ?= -O3 -march=native -fopenmp -Wall -Wextra -std=c11 -Iinclude
LDFLAGS ?= -fopenmp -lpthread -lm
ifeq ($(OS),Windows_NT)
LDFLAGS += -lsynchronization
endif

SRC := src/semiring.c src/ternary.c src/ternary_fast.c src/ternary_gemm.c src/int8.c src/model.c src/pool.c src/util.c
OBJ := $(SRC:.c=.o)

all: build/test build/bench build/run

build:
	mkdir -p build

%.o: %.c include/aim.h include/aim_model.h
	$(CC) $(CFLAGS) -c $< -o $@

build/test: tests/test.c $(OBJ) | build
	$(CC) $(CFLAGS) $< $(OBJ) -o $@ $(LDFLAGS)

build/bench: bench/bench.c $(OBJ) | build
	$(CC) $(CFLAGS) $< $(OBJ) -o $@ $(LDFLAGS)

build/run: app/run.c $(OBJ) | build
	$(CC) $(CFLAGS) $< $(OBJ) -o $@ $(LDFLAGS)

test: build/test
	./build/test

bench: build/bench
	./build/bench

clean:
	rm -rf build src/*.o

.PHONY: all test bench clean
