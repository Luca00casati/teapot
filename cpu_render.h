// cpu_render.h — the reference path tracer on all CPU cores.
#pragma once

#include <atomic>
#include <csignal>
#include <thread>
#include <vector>

#include "display.h"
#include "scene.h"

class CpuRender {
public:
    explicit CpuRender(const Scene &s) : scene_(s) {}
    ~CpuRender() { join(); }
    // trace `spp` stratified samples per pixel of t on `threads` threads; returns at once.
    // `frame` varies the sampling noise between animation frames; rows stop early on *quit
    void start(const Camera &cam, const Target &t, int spp, uint32_t frame, int threads,
               const volatile sig_atomic_t *quit);
    int rows_done() const { return rows_done_.load(); }
    void join();
    void render(const Camera &cam, const Target &t, int spp, uint32_t frame, int threads,
                const volatile sig_atomic_t *quit) {
        start(cam, t, spp, frame, threads, quit);
        join();
    }

private:
    void worker(int id);
    const Scene &scene_;
    Camera cam_{};
    Target tgt_;
    int spp_ = 1;
    uint32_t frame_ = 0;
    const volatile sig_atomic_t *quit_ = nullptr;
    std::atomic<int> next_row_{0}, rows_done_{0};
    std::vector<std::thread> threads_;
};
