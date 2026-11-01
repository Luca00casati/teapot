// minlib.cpp — the C++ runtime hooks the compiler emits calls to, without
// libstdc++/libsupc++: heap new/delete on malloc, and the pure-virtual trap.

#include "minlib.h"

void *operator new(size_t n) { return ml::xrealloc(nullptr, n); }
void *operator new[](size_t n) { return ml::xrealloc(nullptr, n); }
void operator delete(void *p) noexcept { free(p); }
void operator delete[](void *p) noexcept { free(p); }
void operator delete(void *p, size_t) noexcept { free(p); }
void operator delete[](void *p, size_t) noexcept { free(p); }

extern "C" void __cxa_pure_virtual() { abort(); }
