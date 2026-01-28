#pragma once
#include <vector_types.h>

inline float2 operator+(const float2 a, const float2 b) { return {.x = a.x + b.x, .y = a.y + b.y}; }
inline float2 operator-(const float2 a, const float2 b) { return {.x = a.x - b.x, .y = a.y - b.y}; }
inline float2 operator*(const float2 a, const float2 b) { return {.x = a.x * b.x, .y = a.y * b.y}; }
inline float2 operator*(const float2 a, const float b) { return {.x = a.x * b, .y = a.y * b}; }
inline float2 operator*(const float a, const float2 b) { return {.x = a * b.x, .y = a * b.y}; }
inline float2 operator/(const float2 a, const float2 b) { return {.x = a.x / b.x, .y = a.y / b.y}; }
inline float2 operator/(const float2 a, const float b) { return {.x = a.x / b, .y = a.y / b}; }

inline float2 operator+=(float2& a, const float2 b) { return a = a + b; }

inline float2 to_float2(const int2 v) { return {.x = static_cast<float>(v.x), .y = static_cast<float>(v.y)}; }

inline int2 to_int2(const float2 v) { return {.x = static_cast<int>(v.x), .y = static_cast<int>(v.y)}; }

constexpr bool is_zero(float f) { return f == 0.0f; }

constexpr bool is_zero(const float2 f) { return f.x == 0.0f && f.y == 0.0f; }
