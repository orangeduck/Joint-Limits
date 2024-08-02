#pragma once
#include "vec.h"

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
        if (fabs(ev - ev_new) < eps)
        {
            break;
        }
        
        // Update best guess
        v = v_new;
        ev = ev_new;
    }
    
    return v;
}

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

float mat3_trace(const mat3 M)
{
    return M.xx + M.yy + M.zz;
}

float mat3_det(const mat3 M)
{
    return  M.xx * (M.yy * M.zz - M.zy * M.yz) -
            M.xy * (M.yx * M.zz - M.yz * M.zx) +
            M.xz * (M.yx * M.zy - M.yy * M.zx);
}

// solve cubic equation of form x^3 + a*x^2 + b*x + c
// outputs roots into x and returns the number of roots
int cubic(float x[3], float a, float b, float c)
{
    float q = (a*a - 3.0f*b) / 9.0f;
	float r = (2.0f*a*a*a - 9.0f*a*b + 27.0f*c) / 54.0f;
    float d = q*q*q - r*r;
    
    if (d > 0.0f) 
    {
        // Three Real Roots
        float t = acosf(clampf(r / sqrtf(q*q*q), -1.0f, 1.0f));
        x[0] = -2.0f * sqrtf(q) * cosf((t             ) / 3.0f) - a / 3.0f;
        x[1] = -2.0f * sqrtf(q) * cosf((t + 2.0f * PIf) / 3.0f) - a / 3.0f;
        x[2] = -2.0f * sqrtf(q) * cosf((t + 4.0f * PIf) / 3.0f) - a / 3.0f;
        return 3;
    }
    else
    {
        // One Real Root
        float e = powf(sqrtf(-d) + fabs(r), 1.0f / 3.0f);
        e = r > 0.0f ? -e : e;
        x[0] = (e + q / e) - a / 3.0f;
        return 1;
    }
}

float mat3_f_trace_cg(const float A, const float B, const float C, const float eps=1e-10f)
{
    // Compute polynomial coefficients
    float b = -2.0f * A;
    float c = -8.0f * C;
    float d = A*A - B;
    
    // Compute cubic resolvent coefficients 
    float a3 = -b;
	float b3 = -4.0f*d;
	float c3 = -c*c + 4.0f*b*d;
	float x3[3];
	int num = cubic(x3, a3, b3, c3);

    // Find root with largest magnitude
	float y = x3[0];
	if (num > 1 && fabs(x3[1]) > fabs(y)) { y = x3[1]; }
	if (num > 2 && fabs(x3[2]) > fabs(y)) { y = x3[2]; }

    // Find quadratic for trace root
    float q2, p2;
	float D = y*y - 4.0f*d;
    
	if (fabs(D) < eps)
	{
		D = -4.0f * (b - y);
		q2 = y * 0.5f;
        p2 = fabs(D) < eps ? 0.0f : (-sqrtf(D)) * 0.5f;
	}
	else
	{
		float q1 = (y + sqrtf(D)) * 0.5f;
		q2 = (y - sqrtf(D)) * 0.5f;
		p2 = c / (q1 - q2);
	}

    // Find trace root
	D = p2*p2 - 4.0f*q2;
    return D < 0.0f ? -p2 * 0.5f : (-p2 + sqrtf(D)) * 0.5f;
}

// Partial derivative of matrix determinant
mat3 mat3_ddet(const mat3 M)
{
    vec3 da = cross(M.c1(), M.c2());
    vec3 db = cross(M.c2(), M.c0());
    vec3 dc = cross(M.c0(), M.c1());
    return mat3_from_cols(da, db, dc);
}

// The squared frobenius norm
float mat3_frobenius_squared(const mat3 M)
{
    return M.xx*M.xx + M.xy*M.xy + M.xz*M.xz +
           M.yx*M.yx + M.xy*M.yy + M.yz*M.yz +
           M.zx*M.zx + M.xy*M.zy + M.zz*M.zz;
}

void mat3_polar(mat3& R, mat3& S, const mat3 M)
{
    float A = mat3_trace(mat3_transpose_mul(M, M));
    float B = mat3_frobenius_squared(mat3_transpose_mul(M, M));
    float C = mat3_det(M);
    
    mat3 dAdM = 2.0f * M;
    mat3 dBdM = 4.0f * mat3_mul(M, mat3_transpose_mul(M, M));
    mat3 dCdM = mat3_ddet(M);
    
    float f = mat3_f_trace_cg(A, B, C);
    float denom = 4.0f*f*f*f - 4.0f*A*f - 8.0f*C;
    
    float dfdA = (2.0f*f*f + 2.0f*A) / denom;
    float dfdB = -2.0f / denom;
    float dfdC = (8.0f*f) / denom;
    
    R = dfdA*dAdM + dfdB*dBdM + dfdC*dCdM;
    S = mat3_transpose_mul(R, M); 
}

void mat3_svd(mat3& U, vec3& s, mat3& V, const mat3 M)
{
    mat3 R, S;
    mat3_polar(R, S, M);
    
    V = mat3_transpose(R);
    mat3 E = mat3_mul(mat3_mul(R, S), V);
    s = vec3(E.xx, E.yy, E.zz);
    U = mat3_mul(mat3_mul(M, R), (1.0f / E));
}



