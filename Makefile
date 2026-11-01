CXXFLAGS ?= -O3 -march=native -ffast-math -Wall -Wextra
CXXFLAGS += -std=c++11 -fno-exceptions -fno-rtti -MMD -MP $(DEFINES)
# developer switches, e.g. make clean; make DEFINES='-DTEAPOT_GPU_DEBUG -DTEAPOT_DUMP_KERNEL="\"k.bin\""'
#   TEAPOT_GPU_DEBUG       GPU renders a traversal heat map (red material, green nodes, blue triangles)
#   TEAPOT_DUMP_KERNEL=p   write the GPU machine code to file p
LDLIBS = -lpthread

SRCS = teapot.cpp scene.cpp temporal.cpp display.cpp \
       gpu/gpu.cpp gpu/rt_kernel.cpp gpu/gen75.cpp gpu/i915.cpp
OBJS = $(SRCS:.cpp=.o)

teapot: $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS) $(LDLIBS)

clean:
	rm -f teapot $(OBJS) $(OBJS:.o=.d) *.ppm

.PHONY: clean
-include $(OBJS:.o=.d)
