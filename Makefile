CFLAGS ?= -O3 -march=native -ffast-math -Wall -Wextra
LDLIBS = -lm -lpthread

teapot: teapot.c gpu.c kir.c rt_kernel.c gen75.c i915.c
	$(CC) $(CFLAGS) -o $@ $< $(LDLIBS)

clean:
	rm -f teapot *.ppm

.PHONY: clean
