// minlib.h — the little of a C++ library this program needs, on top of libc,
// so it builds with -nostdinc++ and links with -nostdlib++:
//
//   Vec<T>      growable array of trivially copyable T (malloc/realloc)
//   Box<T>      owning pointer (delete on destruction), move-only
//   Arr<T, N>   fixed array that can be returned by value
//   FnRef<F>    non-owning reference to a callable (a lambda on the stack)
//   Str         heap string, built with Str::fmt (printf) and +=
//   move, min, max
//
// Allocation failure aborts; nothing here throws.
#pragma once

#include <stdarg.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

inline void *operator new(size_t, void *p) noexcept { return p; }  // placement new (no <new>)

namespace ml {

template <class T> struct remove_ref { typedef T type; };
template <class T> struct remove_ref<T &> { typedef T type; };
template <class T> struct remove_ref<T &&> { typedef T type; };
template <class T> typename remove_ref<T>::type &&move(T &&t) { return static_cast<typename remove_ref<T>::type &&>(t); }

template <class T> T min(T a, T b) { return b < a ? b : a; }
template <class T> T max(T a, T b) { return a < b ? b : a; }

inline void *xrealloc(void *p, size_t n) {
    p = realloc(p, n ? n : 1);
    if (!p) abort();
    return p;
}

template <class T> class Vec {
    static_assert(__is_trivially_copyable(T), "Vec moves elements with realloc");

public:
    Vec() = default;
    explicit Vec(size_t n) { resize(n); }
    Vec(Vec &&o) noexcept : p_(o.p_), n_(o.n_), cap_(o.cap_) { o.p_ = nullptr; o.n_ = o.cap_ = 0; }
    Vec &operator=(Vec &&o) noexcept {
        if (this != &o) { free(p_); p_ = o.p_; n_ = o.n_; cap_ = o.cap_; o.p_ = nullptr; o.n_ = o.cap_ = 0; }
        return *this;
    }
    Vec(const Vec &) = delete;
    Vec &operator=(const Vec &) = delete;
    ~Vec() { free(p_); }

    size_t size() const { return n_; }
    size_t capacity() const { return cap_; }
    bool empty() const { return n_ == 0; }
    T *data() { return p_; }
    const T *data() const { return p_; }
    T &operator[](size_t i) { return p_[i]; }
    const T &operator[](size_t i) const { return p_[i]; }
    T *begin() { return p_; }
    T *end() { return p_ + n_; }
    const T *begin() const { return p_; }
    const T *end() const { return p_ + n_; }

    void reserve(size_t c) {
        if (c > cap_) { p_ = (T *)xrealloc(p_, c * sizeof(T)); cap_ = c; }
    }
    void resize(size_t n) {                  // new elements are value-initialised
        if (n > cap_) reserve(n > 2 * cap_ ? n : 2 * cap_);
        for (size_t i = n_; i < n; i++) new (p_ + i) T();
        n_ = n;
    }
    void assign(size_t n, const T &v) {
        n_ = 0;
        reserve(n);
        for (size_t i = 0; i < n; i++) new (p_ + i) T(v);
        n_ = n;
    }
    void push_back(const T &v) {
        if (n_ == cap_) reserve(cap_ ? 2 * cap_ : 8);
        new (p_ + n_++) T(v);
    }
    void clear() { n_ = 0; }

private:
    T *p_ = nullptr;
    size_t n_ = 0, cap_ = 0;
};

template <class T> class Box {
public:
    Box() = default;
    Box(decltype(nullptr)) {}
    explicit Box(T *p) : p_(p) {}
    template <class U> Box(Box<U> &&o) : p_(o.release()) {}
    Box(Box &&o) : p_(o.release()) {}
    Box &operator=(Box &&o) { reset(o.release()); return *this; }
    Box(const Box &) = delete;
    Box &operator=(const Box &) = delete;
    ~Box() { delete p_; }

    T *get() const { return p_; }
    T *operator->() const { return p_; }
    T &operator*() const { return *p_; }
    explicit operator bool() const { return p_ != nullptr; }
    T *release() { T *p = p_; p_ = nullptr; return p; }
    void reset(T *p = nullptr) { T *old = p_; p_ = p; delete old; }

private:
    T *p_ = nullptr;
};

template <class T, size_t N> struct Arr {
    T v[N];
    T &operator[](size_t i) { return v[i]; }
    const T &operator[](size_t i) const { return v[i]; }
    T *data() { return v; }
    const T *data() const { return v; }
};

template <class F> class FnRef;
template <class R, class... A> class FnRef<R(A...)> {
public:
    template <class F> FnRef(F &f) : obj_(&f), call_([](void *o, A... a) -> R { return (*(F *)o)(a...); }) {}
    R operator()(A... a) const { return call_(obj_, a...); }

private:
    void *obj_;
    R (*call_)(void *, A...);
};

class Str {
public:
    Str() = default;
    Str(const char *s) { *this += s; }
    Str(const Str &o) { *this += o.c_str(); }
    Str &operator=(const Str &o) {
        if (this != &o) { b_.clear(); *this += o.c_str(); }
        return *this;
    }
    Str(Str &&) = default;
    Str &operator=(Str &&) = default;

    static Str fmt(const char *f, ...) __attribute__((format(printf, 1, 2)));
    Str &operator+=(const char *s) { return append(s, strlen(s)); }
    Str &append(const char *s, size_t n) {   // s must not point into this string
        size_t k = size();
        b_.resize(k + n + 1);                 // the new last byte is the zero terminator
        memcpy(b_.data() + k, s, n);
        return *this;
    }
    const char *c_str() const { return b_.empty() ? "" : b_.data(); }
    size_t size() const { return b_.empty() ? 0 : b_.size() - 1; }
    bool empty() const { return size() == 0; }

private:
    Vec<char> b_;   // the characters and a zero terminator; empty for ""
};

inline Str Str::fmt(const char *f, ...) {
    va_list ap;
    va_start(ap, f);
    int n = vsnprintf(nullptr, 0, f, ap);
    va_end(ap);
    Str s;
    if (n <= 0) return s;
    s.b_.resize(n + 1);
    va_start(ap, f);
    vsnprintf(s.b_.data(), n + 1, f, ap);
    va_end(ap);
    return s;
}

}  // namespace ml
