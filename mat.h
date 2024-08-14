#pragma once
#include "vec.h"
#include <assert.h>

struct mat3
{
    mat3() : 
        xx(1.0f), xy(0.0f), xz(0.0f),
        yx(0.0f), yy(1.0f), yz(0.0f),
        zx(0.0f), zy(0.0f), zz(1.0f) {}
    
    mat3(
        float _xx, float _xy, float _xz, 
        float _yx, float _yy, float _yz, 
        float _zx, float _zy, float _zz) : 
        xx(_xx), xy(_xy), xz(_xz),
        yx(_yx), yy(_yy), yz(_yz),
        zx(_zx), zy(_zy), zz(_zz) {}
    
    mat3(vec3 r0, vec3 r1, vec3 r2) : 
        xx(r0.x), xy(r0.y), xz(r0.z),
        yx(r1.x), yy(r1.y), yz(r1.z),
        zx(r2.x), zy(r2.y), zz(r2.z) {}
    
    vec3 r0() const { return vec3(xx, xy, xz); }
    vec3 r1() const { return vec3(yx, yy, yz); }
    vec3 r2() const { return vec3(zx, zy, zz); }

    vec3 c0() const { return vec3(xx, yx, zx); }
    vec3 c1() const { return vec3(xy, yy, zy); }
    vec3 c2() const { return vec3(xz, yz, zz); }

    float xx, xy, xz,
          yx, yy, yz,
          zx, zy, zz;
};

static inline mat3 mat3_from_cols(vec3 c0, vec3 c1, vec3 c2)
{
    return mat3(
        c0.x, c1.x, c2.x,
        c0.y, c1.y, c2.y,
        c0.z, c1.z, c2.z);
}    

static inline mat3 operator+(mat3 m, mat3 n)
{
    return mat3(
        m.xx + n.xx, m.xy + n.xy, m.xz + n.xz,
        m.yx + n.yx, m.yy + n.yy, m.yz + n.yz,
        m.zx + n.zx, m.zy + n.zy, m.zz + n.zz);
}

static inline mat3 operator-(mat3 m, mat3 n)
{
    return mat3(
        m.xx - n.xx, m.xy - n.xy, m.xz - n.xz,
        m.yx - n.yx, m.yy - n.yy, m.yz - n.yz,
        m.zx - n.zx, m.zy - n.zy, m.zz - n.zz);
}

static inline mat3 operator/(mat3 m, float v)
{
    return mat3(
        m.xx / v, m.xy / v, m.xz / v,
        m.yx / v, m.yy / v, m.yz / v,
        m.zx / v, m.zy / v, m.zz / v);
}

static inline mat3 operator/(float v, mat3 m)
{
    return mat3(
        v / m.xx, v / m.xy, v / m.xz,
        v / m.yx, v / m.yy, v / m.yz,
        v / m.zx, v / m.zy, v / m.zz);
}

static inline mat3 operator*(float v, mat3 m)
{
    return mat3(
        v * m.xx, v * m.xy, v * m.xz,
        v * m.yx, v * m.yy, v * m.yz,
        v * m.zx, v * m.zy, v * m.zz);
}

static inline mat3 mat3_zero()
{
    return mat3(
        0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f);
}

static inline mat3 mat3_eye()
{
    return mat3(
        1.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 1.0f);
}

static inline mat3 mat3_transpose(mat3 m)
{
    return mat3(
        m.xx, m.yx, m.zx,
        m.xy, m.yy, m.zy,
        m.xz, m.yz, m.zz);
}

static inline mat3 mat3_abs(mat3 m)
{
    return mat3(
        fabsf(m.xx), fabsf(m.xy), fabsf(m.xz),
        fabsf(m.yx), fabsf(m.yy), fabsf(m.yz),
        fabsf(m.zx), fabsf(m.zy), fabsf(m.zz));
}


static inline mat3 mat3_mul(mat3 m, mat3 n)
{
  return mat3(
      dot(m.r0(), n.c0()), dot(m.r0(), n.c1()), dot(m.r0(), n.c2()),
      dot(m.r1(), n.c0()), dot(m.r1(), n.c1()), dot(m.r1(), n.c2()),
      dot(m.r2(), n.c0()), dot(m.r2(), n.c1()), dot(m.r2(), n.c2()));
}

static inline mat3 mat3_transpose_mul(mat3 m, mat3 n)
{
  return mat3(
      dot(m.c0(), n.c0()), dot(m.c0(), n.c1()), dot(m.c0(), n.c2()),
      dot(m.c1(), n.c0()), dot(m.c1(), n.c1()), dot(m.c1(), n.c2()),
      dot(m.c2(), n.c0()), dot(m.c2(), n.c1()), dot(m.c2(), n.c2()));
}

static inline vec3 mat3_mul_vec3(mat3 m, vec3 v)
{
    return vec3(
        dot(m.r0(), v),
        dot(m.r1(), v),
        dot(m.r2(), v));
}

static inline vec3 mat3_transpose_mul_vec3(mat3 m, vec3 v)
{
    return vec3(
        dot(m.c0(), v),
        dot(m.c1(), v),
        dot(m.c2(), v));
}

static inline mat3 mat3_from_angle_axis(float angle, vec3 axis)
{
    float a0 = axis.x, a1 = axis.y, a2 = axis.z; 
    float c = cosf(angle), s = sinf(angle), t = 1.0f - cosf(angle);
    
    return mat3(
        c+a0*a0*t, a0*a1*t-a2*s, a0*a2*t+a1*s,
        a0*a1*t+a2*s, c+a1*a1*t, a1*a2*t-a0*s,
        a0*a2*t-a1*s, a1*a2*t+a0*s, c+a2*a2*t);
}

static inline mat3 mat3_outer(vec3 v, vec3 w)
{
    return mat3(
        v.x * w.x, v.x * w.y, v.x * w.z,
        v.y * w.x, v.y * w.y, v.y * w.z,
        v.z * w.x, v.z * w.y, v.z * w.z);
}

static inline vec3 mat3_svd_dominant_eigen(
    const mat3 A, 
    const vec3 v0,
    const int iterations, 
    const float eps)
{
    // Initial Guess at Eigen Vector & Value
    vec3 v = v0;
    float ev = (mat3_mul_vec3(A, v) / v).x;
    
    for (int i = 0; i < iterations; i++)
    {
        // Power Iteration
        vec3 Av = mat3_mul_vec3(A, v);
        
        // Next Guess at Eigen Vector & Value
        vec3 v_new = normalize(Av);
        float ev_new = (mat3_mul_vec3(A, v_new) / v_new).x;
        
        // Break if converged
        if (fabsf(ev - ev_new) < eps)
        {
            break;
        }
        
        // Update best guess
        v = v_new;
        ev = ev_new;
    }
    
    return v;
}

// Note: This returns V^T rather than V
static inline void mat3_svd_piter(
    mat3& U,
    vec3& s,
    mat3& V,
    const mat3 A, 
    const int iterations = 64,
    const float eps = 1e-5f)
{
    // First Eigen Vector
    vec3 g0 = vec3(1, 0, 0);
    mat3 B0 = A;
    vec3 u0 = mat3_svd_dominant_eigen(B0, g0, iterations, eps);
    vec3 v0_unnormalized = mat3_transpose_mul_vec3(A, u0);
    float s0 = length(v0_unnormalized);
    vec3 v0 = s0 < eps ? g0 : normalize(v0_unnormalized);
    
    // Second Eigen Vector
    mat3 B1 = A;
    vec3 g1 = normalize(cross(vec3(0, 0, 1), v0));
    B1 = B1 - s0 * mat3_outer(u0, v0);
    vec3 u1 = mat3_svd_dominant_eigen(B1, g1, iterations, eps);
    vec3 v1_unnormalized = mat3_transpose_mul_vec3(A, u1);
    float s1 = length(v1_unnormalized);
    vec3 v1 = s1 < eps ? g1 : normalize(v1_unnormalized);
    
    // Third Eigen Vector
    mat3 B2 = A;
    vec3 v2 = normalize(cross(v0, v1));
    B2 = B2 - s0 * mat3_outer(u0, v0);
    B2 = B2 - s1 * mat3_outer(u1, v1);
    vec3 u2 = mat3_svd_dominant_eigen(B2, v2, iterations, eps);
    float s2 = length(mat3_transpose_mul_vec3(A, u2));
    
    // Done
    U = mat3(u0, u1, u2);
    s = vec3(s0, s1, s2);
    V = mat3(v0, v1, v2);
}

vec3 mat3_diag(const mat3 M)
{
    return vec3(M.xx, M.yy, M.zz);
}

float mat3_trace(const mat3 M)
{
    return M.xx + M.yy + M.zz;
}

float mat3_det(const mat3 M)
{
    return (M.xx * M.yy * M.zz) + 
           (M.xy * M.yz * M.zx) + 
           (M.xz * M.yx * M.zy) -
           (M.xz * M.yy * M.zx) - 
           (M.xy * M.yx * M.zz) - 
           (M.xx * M.yz * M.zy);
}

// Partial derivative of matrix determinant (adj(M)^T)
mat3 mat3_ddet(const mat3 M)
{
    vec3 da = cross(M.c1(), M.c2());
    vec3 db = cross(M.c2(), M.c0());
    vec3 dc = cross(M.c0(), M.c1());
    return mat3_from_cols(da, db, dc);
}

// The squared frobenius norm of a matrix (|| M ||^2)
float mat3_frobenius_squared(const mat3 M)
{
    return M.xx*M.xx + M.xy*M.xy + M.xz*M.xz +
           M.yx*M.yx + M.yy*M.yy + M.yz*M.yz +
           M.zx*M.zx + M.zy*M.zy + M.zz*M.zz;
}

// returns the real root with the largest magnitude which solves the cubic 
// equation of form x^3 + a*x^2 + b*x + c
static inline float cubic_max_abs_root(float a, float b, float c)
{
    float q = (a*a - 3.0f*b) / 9.0f;
    float r = (2.0f*a*a*a - 9.0f*a*b + 27.0f*c) / 54.0f;
    
    if (r*r < q*q*q) 
    {
        // Three Real Roots
        float t = acosf(clampf(r / sqrtf(q*q*q), -1.0f, 1.0f));
        float x0 = -2.0f * sqrtf(q) * cosf((t             ) / 3.0f) - a / 3.0f;
        float x1 = -2.0f * sqrtf(q) * cosf((t + 2.0f * PIf) / 3.0f) - a / 3.0f;
        float x2 = -2.0f * sqrtf(q) * cosf((t - 2.0f * PIf) / 3.0f) - a / 3.0f;
        return fabsf(x0) > fabsf(x1) && fabsf(x0) > fabsf(x2) ? x0 :
               fabsf(x1) > fabsf(x2) && fabsf(x1) > fabsf(x0) ? x1 : x2;
    }
    else
    {
        // One Real Root
        float e = powf(sqrtf(r*r - q*q*q) + fabsf(r), 1.0f / 3.0f);
        e = r > 0.0f ? -e : e;
        float f = e == 0.0f ? 0.0f : q / e;
        return (e + f) - a / 3.0f;
    }
}

// Computes tr(M^T R) and the singular values from A, B and C
float mat3_f_trace_cg(vec3& s, float A, float B, float C)
{
    // Compute polynomial coefficients
    float b = -2.0f * A;
    float c = -8.0f * C;
    float d = -A*A + 2.0f * B;
    
    // Find root with largest magnitude using cubic resolvent coefficients 
    float y = cubic_max_abs_root(-b, -4.0f*d, -c*c + 4.0f*b*d);

    // Find quadratics for each pair of quartic roots
    float q1, p1, q2, p2;
    
    float D = y*y - 4.0f*d;
    if (D < 1e-10f)
    {
        float D2 = maxf(-4.0f * (b - y), 0.0f);
        q1 = q2 = y * 0.5f;
        p1 = +sqrtf(D2) * 0.5f;            
        p2 = -sqrtf(D2) * 0.5f;            
    }
    else
    {
        q1 = (y + sqrtf(D)) * 0.5f;
        q2 = (y - sqrtf(D)) * 0.5f;
        p1 = (-c) / (q1 - q2);
        p2 = (+c) / (q1 - q2);
    }

    // Find first two roots
    float D01 = maxf(p1*p1 - 4.0f*q1, 0.0f);
    float x0 = (-p1 + sqrtf(D01)) * 0.5f;
    float x1 = (-p1 - sqrtf(D01)) * 0.5f;
    
    // Find second two roots
    float D23 = maxf(p2*p2 - 4.0f*q2, 0.0f);
    float x2 = (-p2 - sqrtf(D23)) * 0.5f;
    float x3 = (-p2 + sqrtf(D23)) * 0.5f;
    
    // Singular Values
    s.x = (x0 + x3) * 0.5f;
    s.y = (x1 + x3) * 0.5f;
    s.z = (x2 + x3) * 0.5f;
    
    // return trace root
    return x3;
}

// Computes the polar decomposition and singular values of a matrix M using the 
// closed-form solution
void mat3_polar(mat3& R, mat3& S, vec3& s, const mat3 M)
{
    float A = mat3_frobenius_squared(M);
    float B = mat3_frobenius_squared(mat3_transpose_mul(M, M));
    float C = mat3_det(M);
    
    float f = mat3_f_trace_cg(s, A, B, C);
    float denom = 4.0f*f*f*f - 4.0f*A*f - 8.0f*C;
    
    if (fabsf(denom) < 1e-10f)
    {
        R = mat3_eye();
        S = mat3_transpose_mul(R, M); 
        return;
    }        
    
    float dfdA = (2.0f*f*f + 2.0f*A) / denom;
    float dfdB = -2.0f / denom;
    float dfdC = (8.0f*f) / denom;

    mat3 dAdM = 2.0f * M;
    mat3 dBdM = 4.0f * mat3_mul(M, mat3_transpose_mul(M, M));
    mat3 dCdM = mat3_ddet(M);
    
    R = dfdA*dAdM + dfdB*dBdM + dfdC*dCdM;
    S = mat3_transpose_mul(R, M); 
}

// Returns true if a matrix is symmetric within some tolerance
bool mat3_is_sym(const mat3 M, const float tolerance = 1e-4f)
{
    return 
        fabsf(M.xy - M.yx) < tolerance &&
        fabsf(M.zx - M.xz) < tolerance &&
        fabsf(M.yz - M.zy) < tolerance;
}

// For a symmetric matrix returns the first eigen vector given the first eigen value
vec3 mat3_sym_evec0(const mat3 M, const float eval0)
{
    assert(mat3_is_sym(M));
    
    vec3 row0 = vec3(M.xx - eval0, M.xy, M.xz);
    vec3 row1 = vec3(M.yx, M.yy - eval0, M.yz);
    vec3 row2 = vec3(M.zx, M.zy, M.zz - eval0);
    
    vec3 r0xr1 = cross(row0, row1);
    vec3 r0xr2 = cross(row0, row2);
    vec3 r1xr2 = cross(row1, row2);
    
    float d0 = dot(r0xr1, r0xr1);
    float d1 = dot(r0xr2, r0xr2);
    float d2 = dot(r1xr2, r1xr2);
    
    if (d0 == 0.0f && d1 == 0.0f && d2 == 0.0f)
    {
        // No valid eigen vectors
        return vec3(1.0f, 0.0f, 0.0f);
    }
    else
    {
        // Return largest candidate
        return d0 >= d1 && d0 >= d2 ? r0xr1 / sqrtf(d0) :
               d1 >= d0 && d1 >= d2 ? r0xr2 / sqrtf(d1) :
                                      r1xr2 / sqrtf(d2); 
    }
}

// For a symmetric matrix returns the second eigen vector given the first eigen vector 
// and the second eigen value
vec3 mat3_sym_evec1(const mat3 M, const vec3 evec0, const float eval1)
{
    assert(mat3_is_sym(M));

    vec3 u = fabsf(evec0.x) > fabsf(evec0.y) ?
        vec3(-evec0.z, 0.0f, +evec0.x) / sqrtf(evec0.x * evec0.x + evec0.z * evec0.z) :
        vec3(0.0f, +evec0.z, -evec0.y) / sqrtf(evec0.y * evec0.y + evec0.z * evec0.z);
    
    vec3 v = cross(evec0, u);
    
    float m00 = dot(u, mat3_mul_vec3(M, u)) - eval1;
    float m01 = dot(u, mat3_mul_vec3(M, v));
    float m11 = dot(v, mat3_mul_vec3(M, v)) - eval1;

    if (fabsf(m00) >= fabsf(m11))
    {
        if (maxf(fabsf(m00), fabsf(m01)) <= 0.0f)
        {
            return u;
        }
        
        if (fabsf(m00) >= fabsf(m01))
        {
            m01 /= m00;
            m00 = 1.0f / sqrtf(1.0f + m01 * m01);
            return m01 * m00 * u - m00 * v;
        }
        else
        {
            m00 /= m01;
            m01 = 1.0f / sqrtf(1.0f + m00 * m00);
            return m01 * u - m00 * m01 * v;
        }        
    }
    else
    {
        if (maxf(fabsf(m00), fabsf(m01)) <= 0.0f)
        {
            return u;
        }
        
        if (fabsf(m11) >= fabsf(m01))
        {
            m01 /= m11;
            m11 = 1.0f / sqrtf(1.0f + m01 * m01);
            return m11 * u - m01 * m11 * v;
        }
        else
        {
            m11 /= m01;
            m01 = 1.0f / sqrtf(1.0f + m11 * m11);
            return m11 * m01 * u - m01 * v;
        }        
    }
}

// Find the eigen vectors from the eigen values of a symmetric 3x3 matrix
mat3 mat3_sym_evecs_from_evals(const mat3& M, const vec3 evals)
{    
    // Assert matrix is symmetric
    assert(mat3_is_sym(M));
    
    // Check matrix has some magnitude
    if (M.xy * M.xy + M.xz * M.xz + M.yz * M.yz <= 0.0f)
    {
        return mat3_eye();
    }
    
    // Compute Eigen Vectors from Eigen Values
    if (mat3_det(M) >= 0.0)
    {
        vec3 evec0 = mat3_sym_evec0(M, evals.x);
        vec3 evec1 = mat3_sym_evec1(M, evec0, evals.y);
        vec3 evec2 = cross(evec1, evec0);
        return mat3_from_cols(evec0, evec1, evec2);
    }
    else
    {
        vec3 evec2 = mat3_sym_evec0(M, evals.z);
        vec3 evec1 = mat3_sym_evec1(M, evec2, evals.y);
        vec3 evec0 = cross(evec2, evec1);
        return mat3_from_cols(evec0, evec1, evec2);
    }
}

// Compute SVD using polar decomposition and symmetric eigenvector computation
void mat3_svd(mat3& U, vec3& s, mat3& V, const mat3 M)
{
    mat3 R, S;
    mat3_polar(R, S, s, M);
    
    V = mat3_sym_evecs_from_evals(S, s);
    U = mat3_mul(R, V);
}

// Computes the largest absolute value in the matrix
float mat3_max_abs(const mat3 M)
{
    return maxf(maxf(maxf(maxf(maxf(maxf(maxf(maxf(
        fabsf(M.xx) , fabsf(M.xy)), fabsf(M.xy)), 
        fabsf(M.yx)), fabsf(M.yy)), fabsf(M.yz)), 
        fabsf(M.zx)), fabsf(M.zy)), fabsf(M.zz));
}

// More numerically stable version of mat3_polar
void mat3_polar_stable(mat3& R, mat3& S, vec3& s, const mat3 M)
{
    float scale = mat3_max_abs(M);
    if (scale == 0.0f)
    {
        R = mat3_eye();
        S = M;
        s = vec3();
        return;
    }
    
    mat3_polar(R, S, s, M / scale);
    S = scale * S;
    s = scale * s;
}

// More numerically stable version of mat3_svd
void mat3_svd_stable(mat3& U, vec3& s, mat3& V, const mat3 M)
{
    float scale = mat3_max_abs(M);
    if (scale == 0.0f)
    {
        U = mat3_eye();
        s = vec3();
        V = mat3_eye();
        return;
    }
    
    mat3 R, S;
    mat3_polar(R, S, s, M / scale);
    
    V = mat3_sym_evecs_from_evals(S, s);
    U = mat3_mul(R, V);
    s = scale * s;
}
