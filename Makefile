# The teapot is written in lisc (https://github.com/Luca00casati/lisc).
#   make                 build ./teapot (kernel interfaces only: DRM/KMS or a PPM file)
#   make SDL=1           also a desktop window through SDL2 (teapot -x; the default under X11/Wayland)
#   make test            check the host side against the reference checksums
# developer switches, e.g. make clean; make DEFINES='-DTEAPOT_GPU_DEBUG -DTEAPOT_DUMP_KERNEL=k.bin'
#   TEAPOT_GPU_DEBUG       GPU renders a traversal heat map (red material, green nodes, blue triangles)
#   TEAPOT_DUMP_KERNEL=p   write the GPU machine code to file p
LISC ?= lisc
SRCS = $(wildcard src/*.lsp)

ifdef SDL
DEFINES += -DTEAPOT_SDL
LIBS += -lSDL2
endif

teapot: $(SRCS)
	$(LISC) src/main.lsp -o $@ $(DEFINES) $(LIBS)

tests/host: tests/host.lsp $(SRCS)
	$(LISC) tests/host.lsp -o $@

test: tests/host
	./tests/host

clean:
	rm -f teapot tests/host tests/kernel *.ppm

.PHONY: test clean
