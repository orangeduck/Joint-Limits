extern "C"
{
#include "raylib.h"
#include "raymath.h"
#define RAYGUI_IMPLEMENTATION
#include "raygui.h"
}
#if defined(PLATFORM_WEB)
#include <emscripten/emscripten.h>
#endif

#include "common.h"
#include "vec.h"
#include "mat.h"
#include "quat.h"
#include "spring.h"
#include "array.h"
#include "character.h"
#include "database.h"

#include <initializer_list>
#include <vector>
#include <functional>

//--------------------------------------

static inline Vector3 to_Vector3(vec3 v)
{
    return (Vector3){ v.x, v.y, v.z };
}

//--------------------------------------

// Perform linear blend skinning and copy 
// result into mesh data. Update and upload 
// deformed vertex positions and normals to GPU
void deform_character_mesh(
  Mesh& mesh, 
  const character& c,
  const slice1d<vec3> bone_anim_positions,
  const slice1d<quat> bone_anim_rotations,
  const slice1d<int> bone_parents)
{
    linear_blend_skinning_positions(
        slice1d<vec3>(mesh.vertexCount, (vec3*)mesh.vertices),
        c.positions,
        c.bone_weights,
        c.bone_indices,
        c.bone_rest_positions,
        c.bone_rest_rotations,
        bone_anim_positions,
        bone_anim_rotations);
    
    linear_blend_skinning_normals(
        slice1d<vec3>(mesh.vertexCount, (vec3*)mesh.normals),
        c.normals,
        c.bone_weights,
        c.bone_indices,
        c.bone_rest_rotations,
        bone_anim_rotations);
    
    UpdateMeshBuffer(mesh, 0, mesh.vertices, mesh.vertexCount * 3 * sizeof(float), 0);
    UpdateMeshBuffer(mesh, 2, mesh.normals, mesh.vertexCount * 3 * sizeof(float), 0);
}

Mesh make_character_mesh(character& c)
{
    Mesh mesh = { 0 };
    
    mesh.vertexCount = c.positions.size;
    mesh.triangleCount = c.triangles.size / 3;
    mesh.vertices = (float*)MemAlloc(c.positions.size * 3 * sizeof(float));
    mesh.texcoords = (float*)MemAlloc(c.texcoords.size * 2 * sizeof(float));
    mesh.normals = (float*)MemAlloc(c.normals.size * 3 * sizeof(float));
    mesh.indices = (unsigned short*)MemAlloc(c.triangles.size * sizeof(unsigned short));
    
    memcpy(mesh.vertices, c.positions.data, c.positions.size * 3 * sizeof(float));
    memcpy(mesh.texcoords, c.texcoords.data, c.texcoords.size * 2 * sizeof(float));
    memcpy(mesh.normals, c.normals.data, c.normals.size * 3 * sizeof(float));
    memcpy(mesh.indices, c.triangles.data, c.triangles.size * sizeof(unsigned short));
    
    UploadMesh(&mesh, true);
    
    return mesh;
}

//--------------------------------------

float orbit_camera_update_azimuth(
    const float azimuth, 
    const float mouse_dx,
    const float dt)
{
    return azimuth + 1.0f * dt * -mouse_dx;
}

float orbit_camera_update_altitude(
    const float altitude, 
    const float mouse_dy,
    const float dt)
{
    return clampf(altitude + 1.0f * dt * mouse_dy, 0.0, 0.4f * PIf);
}

float orbit_camera_update_distance(
    const float distance, 
    const float dt)
{
    return clampf(distance +  20.0f * dt * -GetMouseWheelMove(), 0.1f, 100.0f);
}

// Updates the camera using the orbit cam controls
void orbit_camera_update(
    Camera3D& cam, 
    float& camera_azimuth,
    float& camera_altitude,
    float& camera_distance,
    const vec3 target,
    const float mouse_dx,
    const float mouse_dy,
    const float dt)
{
    camera_azimuth = orbit_camera_update_azimuth(camera_azimuth, mouse_dx, dt);
    camera_altitude = orbit_camera_update_altitude(camera_altitude, mouse_dy, dt);
    camera_distance = orbit_camera_update_distance(camera_distance, dt);
    
    quat rotation_azimuth = quat_from_angle_axis(camera_azimuth, vec3(0, 1, 0));
    vec3 position = quat_mul_vec3(rotation_azimuth, vec3(0, 0, camera_distance));
    vec3 axis = normalize(cross(position, vec3(0, 1, 0)));
    
    quat rotation_altitude = quat_from_angle_axis(camera_altitude, axis);
    
    vec3 eye = target + quat_mul_vec3(rotation_altitude, position);

    cam.target = (Vector3){ target.x, target.y, target.z };
    cam.position = (Vector3){ eye.x, eye.y, eye.z };
    
    UpdateCamera(&cam);
}

//--------------------------------------

static inline void quat_unroll_ranges_inplace(
    slice2d<quat> rotations,
    const slice1d<int> range_starts,
    const slice1d<int> range_stops)
{
    for (int r = 0; r < range_starts.size; r++)
    {
        for (int j = 0; j < rotations.cols; j++)
        {
            rotations(range_starts(r), j) = quat_abs(rotations(range_starts(r), j));
        }
        
        for (int i = range_starts(r) + 1; i < range_stops(r); i++)
        {
            for (int j = 0; j < rotations.cols; j++)
            {
                if (quat_dot(rotations(i, j), rotations(i - 1, j)) < 0.0f)
                {
                    rotations(i, j) = -rotations(i, j);
                }
            }
        }
    }
}

static inline void compute_twist_axes(
    slice1d<vec3> twist_axes,
    const slice1d<vec3> reference_positions,
    const slice1d<quat> reference_rotations,
    const slice1d<int> bone_parents,
    const vec3 default_twist_axis = vec3(1, 0, 0),
    const float eps = 1e-8f)
{
    twist_axes.zero();
    
    for (int i = 0; i < bone_parents.size; i++)
    {
        for (int j = 0; j < bone_parents.size; j++)
        {
            if (bone_parents(j) == i)
            {
                twist_axes(i) = twist_axes(i) + reference_positions(j);
            }
        }

        if (length(twist_axes(i)) > eps)
        {
            twist_axes(i) = normalize(twist_axes(i));
        }
        else
        {
            twist_axes(i) = default_twist_axis;
        }
    }
}

static inline int subsample_limit_space_rotations(
    slice1d<vec3> subsampled_limit_space_rotations,
    slice1d<vec3> limit_space_rotations,
    const float threshold = 0.05f)
{
    int count = 0;
    subsampled_limit_space_rotations(count) = limit_space_rotations(count);
    
    for (int i = 1; i < limit_space_rotations.size; i++)
    {
        bool include = true;
        for (int j = 0; j < count; j++)
        {
            if (length(subsampled_limit_space_rotations(j) - limit_space_rotations(i)) < threshold)
            {
                include = false;
                break;
            }
        }
        
        if (include)
        {
            subsampled_limit_space_rotations(count) = limit_space_rotations(i);
            count++;
        }
    }
    
    return count;
}

//--------------------------------------

static inline void fit_limit_orientations(
    vec3& limit_position,
    mat3& limit_rotation,
    const slice1d<vec3> limit_space_rotations)
{
    limit_position = vec3();
  
    // Compute Average Position
    for (int i = 0; i < limit_space_rotations.size; i++)
    {
        limit_position = limit_position + 
            limit_space_rotations(i) / limit_space_rotations.size;
    }
    
    // Compute Inner Product
    mat3 inner_product = mat3_zero();
    
    for (int i = 0; i < limit_space_rotations.size; i++)
    {
        vec3 v = limit_space_rotations(i) - limit_position;
      
        inner_product = inner_product + mat3(
            v.x * v.x, v.x * v.y, v.x * v.z,
            v.y * v.x, v.y * v.y, v.y * v.z,
            v.z * v.x, v.z * v.y, v.z * v.z) / limit_space_rotations.size;
    }
    
    // Perform SVD to extract rotation
    vec3 s;
    mat3 U, V;
    mat3_svd_piter(U, s, V, inner_product);
    
    limit_rotation = mat3_transpose(V);
    
    // Recompute position as midpoint
    vec3 limit_min = vec3(FLT_MAX, FLT_MAX, FLT_MAX);
    vec3 limit_max = vec3(FLT_MIN, FLT_MIN, FLT_MIN);
    
    for (int i = 0; i < limit_space_rotations.size; i++)
    {
        vec3 limit_point = mat3_transpose_mul_vec3(
            limit_rotation,
            limit_space_rotations(i) - limit_position);
      
        limit_min = min(limit_min, limit_point);
        limit_max = max(limit_max, limit_point);
    }
    
    // Re-project midpoint into non-rotated space
    vec3 limit_midpoint = limit_min + (limit_max - limit_min) / 2.0f;
    limit_position = mat3_mul_vec3(limit_rotation, limit_midpoint) + limit_position;
}

//--------------------------------------

static inline void fit_rectangular_limits(
    vec3& limit_min,
    vec3& limit_max,
    const vec3 limit_position,
    const mat3 limit_rotation,
    const slice1d<vec3> limit_space_rotations,
    const float padding = 0.05f)
{
    limit_min = vec3(FLT_MAX, FLT_MAX, FLT_MAX);
    limit_max = vec3(FLT_MIN, FLT_MIN, FLT_MIN);
  
    for (int i = 0; i < limit_space_rotations.size; i++)
    {
        vec3 limit_point = mat3_transpose_mul_vec3(
            limit_rotation,
            limit_space_rotations(i) - limit_position);
      
        limit_min = min(limit_min, limit_point);
        limit_max = max(limit_max, limit_point);
    }
    
    limit_min = limit_min - vec3(padding, padding, padding);
    limit_max = limit_max + vec3(padding, padding, padding);
}

static inline void fit_ellipsoid_limits(
    vec3& limit_scale,
    const vec3 limit_position,
    const mat3 limit_rotation,
    const slice1d<vec3> limit_space_rotations,
    const float padding = 0.05f)
{   
    // Estimate Scales
    limit_scale = vec3(padding, padding, padding);
  
    for (int i = 0; i < limit_space_rotations.size; i++)
    {
        vec3 limit_point = mat3_transpose_mul_vec3(
            limit_rotation,
            limit_space_rotations(i) - limit_position);
      
        limit_scale = max(limit_scale, abs(limit_point));
    }
    
    // Compute required radius
    float radius = 0.0f;
    for (int i = 0; i < limit_space_rotations.size; i++)
    {
        vec3 limit_point = mat3_transpose_mul_vec3(
            limit_rotation,
            limit_space_rotations(i) - limit_position);
      
        radius = maxf(radius, length(limit_point / limit_scale));
    }
    
    // Scale by required radius
    limit_scale = max(radius * limit_scale, vec3(padding, padding, padding));
}

static inline void fit_kdop_limits(
    slice1d<float> limit_mins,
    slice1d<float> limit_maxs,
    const vec3 limit_position,
    const mat3 limit_rotation,
    const slice1d<vec3> limit_space_rotations,
    const slice1d<vec3> limit_kdop_axes,
    const float padding = 0.05f)
{
    limit_mins.set(FLT_MAX);
    limit_maxs.set(FLT_MIN);
  
    for (int i = 0; i < limit_space_rotations.size; i++)
    {
        vec3 limit_point = mat3_transpose_mul_vec3(
            limit_rotation,
            limit_space_rotations(i) - limit_position);
        
        for (int k = 0; k < limit_kdop_axes.size; k++)
        {
            float limit_point_proj = dot(limit_kdop_axes(k), limit_point);
            limit_mins(k) = minf(limit_mins(k), limit_point_proj);
            limit_maxs(k) = maxf(limit_maxs(k), limit_point_proj);
        }
    }
    
    for (int k = 0; k < limit_kdop_axes.size; k++)
    {
        limit_mins(k) -= padding;
        limit_maxs(k) += padding;
    }
}

//--------------------------------------

static inline vec3 apply_rectangular_limit(
    const vec3 limit_space_rotation,
    const vec3 limit_min,
    const vec3 limit_max,
    const vec3 limit_position,
    const mat3 limit_rotation)
{
    vec3 limit_point = mat3_transpose_mul_vec3(
        limit_rotation,
        limit_space_rotation - limit_position);
        
    limit_point = clamp(limit_point, limit_min, limit_max);
    
    return mat3_mul_vec3(limit_rotation, limit_point) + limit_position;
}

static inline vec3 apply_ellipsoid_limit(
    const vec3 limit_space_rotation,
    const vec3 limit_scale,
    const vec3 limit_position,
    const mat3 limit_rotation,
    const int iterations = 8,
    const float eps = 1e-5f)
{
    vec3 limit_point = mat3_transpose_mul_vec3(
        limit_rotation,
        limit_space_rotation - limit_position);
    
    if (length(limit_point / limit_scale) <= 1.0f)
    {
        return limit_space_rotation;
    }
    
    vec3 ss = limit_scale * limit_scale + eps;

    float ss_mid = (ss.y < ss.x) ? (ss.z < ss.x ? ss.x : ss.z) : (ss.z < ss.y ? ss.y : ss.z);
    
    float hmin = sqrtf(dot(limit_point * limit_point, ss * ss) / ss_mid) - ss_mid;
    hmin = maxf(hmin, (fabs(limit_point.x) - limit_scale.x) * limit_scale.x);
    hmin = maxf(hmin, (fabs(limit_point.y) - limit_scale.y) * limit_scale.y);
    hmin = maxf(hmin, (fabs(limit_point.z) - limit_scale.z) * limit_scale.z);

    if (dot(limit_point, limit_point / ss.x) > 1.0f && hmin < 0.0f)
    {
        hmin = 0;
    }

    float h = hmin;
    float hprev;

    for (int i = 0; i < iterations; i++)
    {
        vec3 wa = limit_point / (ss + h);
        vec3 pp = wa * wa * ss;

        hprev = h;
        h = h - (1.0f - sum(pp)) / (2.0f * sum(pp / (ss + h)));

        if (h < hmin)
        {
            h = 0.5f * (hprev + hmin);
            continue;
        }
        
        if (h <= hprev)
        {
            break;
        }
    }

    limit_point = limit_point * ss / (ss + h);
    
    return mat3_mul_vec3(limit_rotation, limit_point) + limit_position;
}

static inline vec3 apply_kdop_limit(
    const vec3 limit_space_rotation,
    const slice1d<float> limit_mins,
    const slice1d<float> limit_maxs,
    const vec3 limit_position,
    const mat3 limit_rotation,
    const slice1d<vec3> kdop_axes)
{
    vec3 limit_point = mat3_transpose_mul_vec3(
        limit_rotation,
        limit_space_rotation - limit_position);
        
    for (int k = 0; k < kdop_axes.size; k++)
    {
        vec3 t0 = limit_point - limit_mins(k) * kdop_axes(k);
        vec3 t1 = limit_point - limit_maxs(k) * kdop_axes(k);
        limit_point = limit_point - minf(dot(t0, kdop_axes(k)), 0.0f) * kdop_axes(k);
        limit_point = limit_point - maxf(dot(t1, kdop_axes(k)), 0.0f) * kdop_axes(k);
    }

    return mat3_mul_vec3(limit_rotation, limit_point) + limit_position;
}

static inline vec3 projection_soften(
    const vec3 original_position,
    const vec3 projected_position,
    const float falloff = 0.1f,
    const float radius = 0.1f,
    const float eps = 1e-5f)
{
    float distance = length(projected_position - original_position);
    
    if (distance > eps)
    {
        float softening = tanhf(distance / (falloff + eps)) * radius;
        return projected_position + normalize(original_position - projected_position) * softening;
    }
    else
    {
        return projected_position;
    }
}

//--------------------------------------

// Rotate a joint to look toward some 
// given target position
void ik_look_at(
    quat& bone_rotation,
    const quat global_parent_rotation,
    const quat global_rotation,
    const vec3 global_position,
    const vec3 child_position,
    const vec3 target_position,
    const float eps = 1e-5f)
{
    vec3 curr_dir = normalize(child_position - global_position);
    vec3 targ_dir = normalize(target_position - global_position);

    if (fabs(1.0f - dot(curr_dir, targ_dir) > eps))
    {
        bone_rotation = quat_inv_mul(global_parent_rotation, 
            quat_mul(quat_between(curr_dir, targ_dir), global_rotation));
    }
}

// Basic two-joint IK in the style of https://theorangeduck.com/page/simple-two-joint
// Here I add a basic "forward vector" which acts like a kind of pole-vetor
// to control the bending direction
void ik_two_bone(
    quat& bone_root_lr, 
    quat& bone_mid_lr,
    const vec3 bone_root, 
    const vec3 bone_mid, 
    const vec3 bone_end, 
    const vec3 target, 
    const vec3 fwd,
    const quat bone_root_gr, 
    const quat bone_mid_gr,
    const quat bone_par_gr,
    const float max_length_buffer) {
    
    float max_extension = 
        length(bone_root - bone_mid) + 
        length(bone_mid - bone_end) - 
        max_length_buffer;
    
    vec3 target_clamp = target;
    if (length(target - bone_root) > max_extension)
    {
        target_clamp = bone_root + max_extension * normalize(target - bone_root);
    }
    
    vec3 axis_dwn = normalize(bone_end - bone_root);
    vec3 axis_rot = normalize(cross(axis_dwn, fwd));

    vec3 a = bone_root;
    vec3 b = bone_mid;
    vec3 c = bone_end;
    vec3 t = target_clamp;
    
    float lab = length(b - a);
    float lcb = length(b - c);
    float lat = length(t - a);

    float ac_ab_0 = acosf(clampf(dot(normalize(c - a), normalize(b - a)), -1.0f, 1.0f));
    float ba_bc_0 = acosf(clampf(dot(normalize(a - b), normalize(c - b)), -1.0f, 1.0f));

    float ac_ab_1 = acosf(clampf((lab * lab + lat * lat - lcb * lcb) / (2.0f * lab * lat), -1.0f, 1.0f));
    float ba_bc_1 = acosf(clampf((lab * lab + lcb * lcb - lat * lat) / (2.0f * lab * lcb), -1.0f, 1.0f));

    quat r0 = quat_from_angle_axis(ac_ab_1 - ac_ab_0, axis_rot);
    quat r1 = quat_from_angle_axis(ba_bc_1 - ba_bc_0, axis_rot);

    vec3 c_a = normalize(bone_end - bone_root);
    vec3 t_a = normalize(target_clamp - bone_root);

    quat r2 = quat_from_angle_axis(
        acosf(clampf(dot(c_a, t_a), -1.0f, 1.0f)),
        normalize(cross(c_a, t_a)));
    
    bone_root_lr = quat_inv_mul(bone_par_gr, quat_mul(r2, quat_mul(r0, bone_root_gr)));
    bone_mid_lr = quat_inv_mul(bone_root_gr, quat_mul(r1, bone_mid_gr));
}

//--------------------------------------

void draw_axis(const vec3 pos, const quat rot, const float scale = 1.0f)
{
    vec3 axis0 = pos + quat_mul_vec3(rot, scale * vec3(1.0f, 0.0f, 0.0f));
    vec3 axis1 = pos + quat_mul_vec3(rot, scale * vec3(0.0f, 1.0f, 0.0f));
    vec3 axis2 = pos + quat_mul_vec3(rot, scale * vec3(0.0f, 0.0f, 1.0f));
    
    DrawLine3D(to_Vector3(pos), to_Vector3(axis0), RED);
    DrawLine3D(to_Vector3(pos), to_Vector3(axis1), GREEN);
    DrawLine3D(to_Vector3(pos), to_Vector3(axis2), BLUE);
}

void draw_axis(const vec3 pos, const mat3 rot, const float scale = 1.0f)
{
    vec3 axis0 = pos + mat3_mul_vec3(rot, scale * vec3(1.0f, 0.0f, 0.0f));
    vec3 axis1 = pos + mat3_mul_vec3(rot, scale * vec3(0.0f, 1.0f, 0.0f));
    vec3 axis2 = pos + mat3_mul_vec3(rot, scale * vec3(0.0f, 0.0f, 1.0f));
    
    DrawLine3D(to_Vector3(pos), to_Vector3(axis0), RED);
    DrawLine3D(to_Vector3(pos), to_Vector3(axis1), GREEN);
    DrawLine3D(to_Vector3(pos), to_Vector3(axis2), BLUE);
}

void draw_skeleton(
    const slice1d<vec3> bone_positions,
    const slice1d<quat> bone_rotations,
    const slice1d<int> bone_parents,
    const int joint_selected = -1)
{
    for (int i = 1; i < bone_positions.size; i++)
    {
        DrawSphereWires(
            to_Vector3(bone_positions(i)), 
            i == joint_selected ? 0.025 : 0.01f, 4, 8, i == joint_selected ? PINK : MAROON);
    }
    
    for (int i = 2; i < bone_positions.size; i++)
    {
        DrawLine3D(
            to_Vector3(bone_positions(i)),
            to_Vector3(bone_positions(bone_parents(i))),
            bone_parents(i) == joint_selected ? PINK : MAROON);
    }      
  
}

int generate_sphere_line_count(int rings, int slices)
{
    return (rings + 2) * slices * 3;
}

void generate_sphere_lines(
    slice1d<vec3> line_starts,
    slice1d<vec3> line_stops,
    int rings, 
    int slices)
{   
    int count = generate_sphere_line_count(rings, slices);
    int index = 0;

    for (int i = 0; i < (rings + 2); i++)
    {
        for (int j = 0; j < slices; j++)
        {
            line_starts(index) = vec3(
                cosf(DEG2RAD*(270 + (180.0f/(rings + 1))*i))*sinf(DEG2RAD*(360.0f*j/slices)),
                sinf(DEG2RAD*(270 + (180.0f/(rings + 1))*i)),
                cosf(DEG2RAD*(270 + (180.0f/(rings + 1))*i))*cosf(DEG2RAD*(360.0f*j/slices)));
            
            line_stops(index) = vec3(
                cosf(DEG2RAD*(270 + (180.0f/(rings + 1))*(i + 1)))*sinf(DEG2RAD*(360.0f*(j + 1)/slices)),
                sinf(DEG2RAD*(270 + (180.0f/(rings + 1))*(i + 1))),
                cosf(DEG2RAD*(270 + (180.0f/(rings + 1))*(i + 1)))*cosf(DEG2RAD*(360.0f*(j + 1)/slices)));
            
            index++;

            line_starts(index) = vec3(
                cosf(DEG2RAD*(270 + (180.0f/(rings + 1))*(i + 1)))*sinf(DEG2RAD*(360.0f*(j + 1)/slices)),
                sinf(DEG2RAD*(270 + (180.0f/(rings + 1))*(i + 1))),
                cosf(DEG2RAD*(270 + (180.0f/(rings + 1))*(i + 1)))*cosf(DEG2RAD*(360.0f*(j + 1)/slices)));
                
              
            line_stops(index) = vec3(
                cosf(DEG2RAD*(270 + (180.0f/(rings + 1))*(i + 1)))*sinf(DEG2RAD*(360.0f*j/slices)),
                sinf(DEG2RAD*(270 + (180.0f/(rings + 1))*(i + 1))),
                cosf(DEG2RAD*(270 + (180.0f/(rings + 1))*(i + 1)))*cosf(DEG2RAD*(360.0f*j/slices)));
            
            index++;
            
            line_starts(index) = vec3(
                cosf(DEG2RAD*(270 + (180.0f/(rings + 1))*(i + 1)))*sinf(DEG2RAD*(360.0f*j/slices)),
                sinf(DEG2RAD*(270 + (180.0f/(rings + 1))*(i + 1))),
                cosf(DEG2RAD*(270 + (180.0f/(rings + 1))*(i + 1)))*cosf(DEG2RAD*(360.0f*j/slices)));
            
            line_stops(index) = vec3(
                cosf(DEG2RAD*(270 + (180.0f/(rings + 1))*i))*sinf(DEG2RAD*(360.0f*j/slices)),
                sinf(DEG2RAD*(270 + (180.0f/(rings + 1))*i)),
                cosf(DEG2RAD*(270 + (180.0f/(rings + 1))*i))*cosf(DEG2RAD*(360.0f*j/slices)));
                
            index++;
        }
    }
    
    assert(index == count);
}

void draw_limit_samples(
    const slice1d<vec3> subsampled_rotations,
    const vec3 limit_position,
    const mat3 limit_rotation,
    const vec3 space_offset, 
    const float scale)
{
    DrawCubeWires(to_Vector3(space_offset), scale * 2.0f * PIf, scale * 2.0f * PIf, scale * 2.0f * PIf, GRAY);
    
    draw_axis(space_offset + scale * 
        limit_position, 
        limit_rotation, 0.5f * scale * PIf);
    
    for (int i = 0; i < subsampled_rotations.size; i ++)
    {
        DrawSphereWires(
            to_Vector3(space_offset + scale * subsampled_rotations(i)),
            0.005f,
            4, 6,
            (Color){135, 60, 190, 50});        
    }
}

void draw_rectangular_limit_bounds(
    const vec3 limit_position,
    const mat3 limit_rotation,
    const vec3 rectangular_limit_min,
    const vec3 rectangular_limit_max,
    const vec3 space_offset, 
    const float scale,
    const int sphere_rings = 32,
    const int sphere_slices = 64)
{
    int line_count = generate_sphere_line_count(sphere_rings, sphere_slices);
    array1d<vec3> line_starts(line_count);
    array1d<vec3> line_stops(line_count);
    generate_sphere_lines(line_starts, line_stops, sphere_rings, sphere_slices);
    
    for (int i = 0; i < line_count; i++)
    {   
        vec3 line_start = apply_rectangular_limit(
            2.0f * PIf * line_starts(i),
            rectangular_limit_min,
            rectangular_limit_max,
            limit_position,
            limit_rotation);
            
        vec3 line_stop = apply_rectangular_limit(
            2.0f * PIf * line_stops(i),
            rectangular_limit_min,
            rectangular_limit_max,
            limit_position,
            limit_rotation);
            
        DrawLine3D(
            to_Vector3(space_offset + scale * line_start), 
            to_Vector3(space_offset + scale * line_stop), 
            (Color){35, 190, 60, 64});
    }
}

void draw_ellipsoid_limit_bounds(
    const vec3 limit_position,
    const mat3 limit_rotation,
    const vec3 ellipsoid_limit_scale,
    const vec3 space_offset, 
    const float scale,
    const int sphere_rings = 32,
    const int sphere_slices = 64)
{
    int line_count = generate_sphere_line_count(sphere_rings, sphere_slices);
    array1d<vec3> line_starts(line_count);
    array1d<vec3> line_stops(line_count);
    generate_sphere_lines(line_starts, line_stops, sphere_rings, sphere_slices);
    
    for (int i = 0; i < line_count; i++)
    {   
        vec3 line_start = apply_ellipsoid_limit(
            2.0f * PIf * line_starts(i),
            ellipsoid_limit_scale,
            limit_position,
            limit_rotation);
            
        vec3 line_stop = apply_ellipsoid_limit(
            2.0f * PIf * line_stops(i),
            ellipsoid_limit_scale,
            limit_position,
            limit_rotation);
                
        DrawLine3D(
            to_Vector3(space_offset + scale * line_start), 
            to_Vector3(space_offset + scale * line_stop), 
            (Color){35, 190, 60, 64});
    }
}

void draw_kdop_limit_bounds(
    const vec3 limit_position,
    const mat3 limit_rotation,
    const slice1d<float> kdop_limit_mins,
    const slice1d<float> kdop_limit_maxs,
    const slice1d<vec3> kdop_axes,
    const vec3 space_offset, 
    const float scale,
    const int sphere_rings = 32,
    const int sphere_slices = 64)
{
    int line_count = generate_sphere_line_count(sphere_rings, sphere_slices);
    array1d<vec3> line_starts(line_count);
    array1d<vec3> line_stops(line_count);
    generate_sphere_lines(line_starts, line_stops, sphere_rings, sphere_slices);
    
    for (int i = 0; i < line_count; i++)
    {   
        vec3 line_start = apply_kdop_limit(
            2.0f * PIf * line_starts(i),
            kdop_limit_mins,
            kdop_limit_maxs,
            limit_position,
            limit_rotation,
            kdop_axes);
            
        vec3 line_stop = apply_kdop_limit(
            2.0f * PIf * line_stops(i),
            kdop_limit_mins,
            kdop_limit_maxs,
            limit_position,
            limit_rotation,
            kdop_axes);
                
        DrawLine3D(
            to_Vector3(space_offset + scale * line_start), 
            to_Vector3(space_offset + scale * line_stop), 
            (Color){35, 190, 60, 64});
    }
}

void draw_current_limit(
    const vec3 projected_limit_space_rotation,
    const vec3 limit_space_rotation,
    const vec3 space_offset, 
    const float scale)
{
    DrawSphereWires(
        to_Vector3(space_offset + scale * limit_space_rotation),
        0.025f, 4, 8, PINK);
    
    DrawSphereWires(
        to_Vector3(space_offset + scale * projected_limit_space_rotation),
        0.025f, 4, 8, PINK);
    
    DrawLine3D(
        to_Vector3(space_offset + scale * limit_space_rotation), 
        to_Vector3(space_offset + scale * projected_limit_space_rotation), 
        PINK);
}


//--------------------------------------

enum
{
    LIMIT_TYPE_RECTANGULAR = 0,
    LIMIT_TYPE_ELLIPSOID = 1,
    LIMIT_TYPE_KDOP = 2,
};

void update_callback(void* args)
{
    ((std::function<void()>*)args)->operator()();
}

int main(void)
{
    // Init Window
    
    const int screen_width = 1280;
    const int screen_height = 720;
    
    SetConfigFlags(FLAG_VSYNC_HINT);
    SetConfigFlags(FLAG_MSAA_4X_HINT);
    InitWindow(screen_width, screen_height, "raylib [joint limits]");
    SetTargetFPS(60);
    
    // Camera

    Camera3D camera = { 0 };
    camera.position = (Vector3){ 2.0f, 3.0f, 5.0f };
    camera.target = (Vector3){ 1.5f, 1.0f, 0.0f };
    camera.up = (Vector3){ 0.0f, 1.0f, 0.0f };
    camera.fovy = 45.0f;
    camera.projection = CAMERA_PERSPECTIVE;
    
    float camera_azimuth = 0.0f;
    float camera_altitude = 0.4f;
    float camera_distance = 4.0f;
    
    // Ground Plane
    
    Shader ground_plane_shader = LoadShader("./resources/checkerboard.vs", "./resources/checkerboard.fs");
    Mesh ground_plane_mesh = GenMeshPlane(20.0f, 20.0f, 10, 10);
    Model ground_plane_model = LoadModelFromMesh(ground_plane_mesh);
    ground_plane_model.materials[0].shader = ground_plane_shader;
    
    // Character
    
    character character_data;
    character_load(character_data, "./resources/character.bin");
    
    Shader character_shader = LoadShader("./resources/character.vs", "./resources/character.fs");
    Mesh character_mesh = make_character_mesh(character_data);
    Model character_model = LoadModelFromMesh(character_mesh);
    character_model.materials[0].shader = character_shader;
    
    // Load Animation Data and build Matching Database
    
    database db;
    database_load(db, "./resources/database.bin");
    
    // Controls
    
    bool reference_pose = true;
    int frame_index = db.range_starts(0);
    int joint_index = 1;

    bool limit_swing_twist = false;
    bool limit_type_edit = false;
    int limit_type = LIMIT_TYPE_RECTANGULAR;

    float rotation_x = 0.0f;
    float rotation_y = 0.0f;
    float rotation_z = 0.0f;

    bool ik_enabled = false;
    float ik_max_length_buffer = 0.015f;
    float ik_foot_height = 0.03f;
    float ik_toe_length = 0.15f;    
    vec3 ik_target = vec3(-0.5f, 0.1f, 0.0f);

    bool lookat_enabled = true;
    vec3 lookat_target = vec3(0.0f, 1.0f, 1.0f);

    bool projection_enabled = true;
    float projection_soften_falloff = 0.1f;
    float projection_soften_radius = 0.1f;
    
    // Pose Data
    
    array1d<vec3> reference_positions(db.nbones());
    array1d<quat> reference_rotations(db.nbones());
    
    backward_kinematics_full(
        reference_positions,
        reference_rotations,
        character_data.bone_rest_positions,
        character_data.bone_rest_rotations,
        db.bone_parents);
    
    array1d<vec3> global_bone_positions(db.nbones());
    array1d<quat> global_bone_rotations(db.nbones());
    array1d<bool> global_bone_computed(db.nbones());
    
    array1d<vec3> adjusted_bone_positions = db.bone_positions(frame_index);
    array1d<quat> adjusted_bone_rotations = db.bone_rotations(frame_index);
    
    // Twist Axes
    
    array1d<vec3> twist_axes(db.nbones());
    
    compute_twist_axes(
        twist_axes,
        reference_positions,
        reference_rotations,
        db.bone_parents);
    
    // Reference Space 
    
    array2d<quat> reference_space_rotations(db.nframes(), db.nbones());
    array2d<quat> reference_space_rotations_swing(db.nframes(), db.nbones());
    array2d<quat> reference_space_rotations_twist(db.nframes(), db.nbones());
    
    for (int i = 0; i < db.nframes(); i++)
    {
        for (int j = 0; j < db.nbones(); j++)
        {
            reference_space_rotations(i, j) = quat_inv_mul(
                reference_rotations(j), db.bone_rotations(i, j));
                
            quat_swing_twist(
                reference_space_rotations_swing(i, j),
                reference_space_rotations_twist(i, j), 
                reference_space_rotations(i, j), twist_axes(j));
        }
    }
    
    quat_unroll_ranges_inplace(
        reference_space_rotations,
        db.range_starts, 
        db.range_stops);
        
    quat_unroll_ranges_inplace(
        reference_space_rotations_swing,
        db.range_starts, 
        db.range_stops);
        
    quat_unroll_ranges_inplace(
        reference_space_rotations_twist,
        db.range_starts, 
        db.range_stops);
    
    // Limit Space
    
    array2d<vec3> limit_space_rotations(db.nframes(), db.nbones());
    array2d<vec3> limit_space_rotations_swing(db.nframes(), db.nbones());
    array2d<vec3> limit_space_rotations_twist(db.nframes(), db.nbones());
    
    for (int i = 0; i < db.nframes(); i++)
    {
        for (int j = 0; j < db.nbones(); j++)
        {   
            limit_space_rotations(i, j) = quat_to_scaled_angle_axis(reference_space_rotations(i, j));
            limit_space_rotations_swing(i, j) = quat_to_scaled_angle_axis(reference_space_rotations_swing(i, j));
            limit_space_rotations_twist(i, j) = quat_to_scaled_angle_axis(reference_space_rotations_twist(i, j));
        }
    }
    
    // Sub-sample Limit Space Samples
    
    array2d<vec3> limit_space_rotations_transpose(db.nbones(), db.nframes());
    array2d<vec3> limit_space_rotations_transpose_swing(db.nbones(), db.nframes());
    array2d<vec3> limit_space_rotations_transpose_twist(db.nbones(), db.nframes());
    array2d_transpose(limit_space_rotations_transpose, limit_space_rotations);
    array2d_transpose(limit_space_rotations_transpose_swing, limit_space_rotations_swing);
    array2d_transpose(limit_space_rotations_transpose_twist, limit_space_rotations_twist);
  
    std::vector<array1d<vec3>> subsampled_limit_space_rotations(db.nbones());
    std::vector<array1d<vec3>> subsampled_limit_space_rotations_swing(db.nbones());
    std::vector<array1d<vec3>> subsampled_limit_space_rotations_twist(db.nbones());
    
    for (int j = 0; j < db.nbones(); j++)
    {
        subsampled_limit_space_rotations[j].resize(db.nframes());
        
        int count = subsample_limit_space_rotations(
            subsampled_limit_space_rotations[j],
            limit_space_rotations_transpose(j));
            
        subsampled_limit_space_rotations[j].resize(count);
        
        subsampled_limit_space_rotations_swing[j].resize(db.nframes());
        
        int count_swing = subsample_limit_space_rotations(
            subsampled_limit_space_rotations_swing[j],
            limit_space_rotations_transpose_swing(j));
            
        subsampled_limit_space_rotations_swing[j].resize(count_swing);
        
        subsampled_limit_space_rotations_twist[j].resize(db.nframes());
        
        int count_twist = subsample_limit_space_rotations(
            subsampled_limit_space_rotations_twist[j],
            limit_space_rotations_transpose_twist(j));
            
        subsampled_limit_space_rotations_twist[j].resize(count_twist);
    }
    
    // Fit Limits
    
    array1d<vec3> limit_positions(db.nbones());
    array1d<mat3> limit_rotations(db.nbones());
    array1d<vec3> limit_positions_swing(db.nbones());
    array1d<mat3> limit_rotations_swing(db.nbones());
    array1d<vec3> limit_positions_twist(db.nbones());
    array1d<mat3> limit_rotations_twist(db.nbones());
    
    bool orient_limits = true;
    
    if (orient_limits)
    {   
        for (int j = 0; j < db.nbones(); j++)
        {
            fit_limit_orientations(
                limit_positions(j),
                limit_rotations(j),
                subsampled_limit_space_rotations[j]);
        }

        for (int j = 0; j < db.nbones(); j++)
        {
            fit_limit_orientations(
                limit_positions_swing(j),
                limit_rotations_swing(j),
                subsampled_limit_space_rotations_swing[j]);
        }
        
        for (int j = 0; j < db.nbones(); j++)
        {
            fit_limit_orientations(
                limit_positions_twist(j),
                limit_rotations_twist(j),
                subsampled_limit_space_rotations_twist[j]);
        }
    }
    else
    {
        limit_positions.zero();
        limit_rotations.set(mat3());
        limit_positions_swing.zero();
        limit_rotations_swing.set(mat3());
        limit_positions_twist.zero();
        limit_rotations_twist.set(mat3());
    }
    
    // Rectangular Limits

    array1d<vec3> rectangular_limit_mins(db.nbones());
    array1d<vec3> rectangular_limit_maxs(db.nbones());    
    array1d<vec3> rectangular_limit_mins_swing(db.nbones());
    array1d<vec3> rectangular_limit_maxs_swing(db.nbones());   
    array1d<vec3> rectangular_limit_mins_twist(db.nbones());
    array1d<vec3> rectangular_limit_maxs_twist(db.nbones());   

    for (int j = 0; j < db.nbones(); j++)
    {
        fit_rectangular_limits(
            rectangular_limit_mins(j),
            rectangular_limit_maxs(j),
            limit_positions(j),
            limit_rotations(j),
            subsampled_limit_space_rotations[j]);
    }

    for (int j = 0; j < db.nbones(); j++)
    {
        fit_rectangular_limits(
            rectangular_limit_mins_swing(j),
            rectangular_limit_maxs_swing(j),
            limit_positions_swing(j),
            limit_rotations_swing(j),
            subsampled_limit_space_rotations_swing[j]);
    }
    
    for (int j = 0; j < db.nbones(); j++)
    {
        fit_rectangular_limits(
            rectangular_limit_mins_twist(j),
            rectangular_limit_maxs_twist(j),
            limit_positions_twist(j),
            limit_rotations_twist(j),
            subsampled_limit_space_rotations_twist[j]);
    }

    // Ellipsoid Limits
    
    array1d<vec3> ellipsoid_limit_scales(db.nbones());
    array1d<vec3> ellipsoid_limit_scales_swing(db.nbones());
    array1d<vec3> ellipsoid_limit_scales_twist(db.nbones());

    for (int j = 0; j < db.nbones(); j++)
    {
        fit_ellipsoid_limits(
            ellipsoid_limit_scales(j),
            limit_positions(j),
            limit_rotations(j),
            subsampled_limit_space_rotations[j]);
    }
    
    for (int j = 0; j < db.nbones(); j++)
    {
        fit_ellipsoid_limits(
            ellipsoid_limit_scales_swing(j),
            limit_positions_swing(j),
            limit_rotations_swing(j),
            subsampled_limit_space_rotations_swing[j]);
    }
    
    for (int j = 0; j < db.nbones(); j++)
    {
        fit_ellipsoid_limits(
            ellipsoid_limit_scales_twist(j),
            limit_positions_twist(j),
            limit_rotations_twist(j),
            subsampled_limit_space_rotations_twist[j]);
    }

    // KDop Limits
    
    array1d<vec3> kdop_axes(13);
    kdop_axes( 0) = normalize(vec3( 1.0f, 0.0f, 0.0f));
    kdop_axes( 1) = normalize(vec3( 0.0f, 1.0f, 0.0f));
    kdop_axes( 2) = normalize(vec3( 0.0f, 0.0f, 1.0f));
    kdop_axes( 3) = normalize(vec3( 1.0f, 1.0f, 1.0f));
    kdop_axes( 4) = normalize(vec3(-1.0f, 1.0f, 1.0f));
    kdop_axes( 5) = normalize(vec3(-1.0f,-1.0f, 1.0f));
    kdop_axes( 6) = normalize(vec3( 1.0f,-1.0f, 1.0f));
    kdop_axes( 7) = normalize(vec3( 0.0f, 1.0f, 1.0f));
    kdop_axes( 8) = normalize(vec3( 0.0f,-1.0f, 1.0f));
    kdop_axes( 9) = normalize(vec3( 1.0f, 0.0f, 1.0f));
    kdop_axes(10) = normalize(vec3(-1.0f, 0.0f, 1.0f));
    kdop_axes(11) = normalize(vec3( 1.0f, 1.0f, 0.0f));
    kdop_axes(12) = normalize(vec3(-1.0f, 1.0f, 0.0f));
    
    array2d<float> kdop_limit_mins(db.nbones(), kdop_axes.size);
    array2d<float> kdop_limit_maxs(db.nbones(), kdop_axes.size);
    array2d<float> kdop_limit_mins_swing(db.nbones(), kdop_axes.size);
    array2d<float> kdop_limit_maxs_swing(db.nbones(), kdop_axes.size);
    array2d<float> kdop_limit_mins_twist(db.nbones(), kdop_axes.size);
    array2d<float> kdop_limit_maxs_twist(db.nbones(), kdop_axes.size);

    for (int j = 0; j < db.nbones(); j++)
    {
        fit_kdop_limits(
            kdop_limit_mins(j),
            kdop_limit_maxs(j),
            limit_positions(j),
            limit_rotations(j),
            subsampled_limit_space_rotations[j],
            kdop_axes);
    }
    
    for (int j = 0; j < db.nbones(); j++)
    {
        fit_kdop_limits(
            kdop_limit_mins_swing(j),
            kdop_limit_maxs_swing(j),
            limit_positions_swing(j),
            limit_rotations_swing(j),
            subsampled_limit_space_rotations_swing[j],
            kdop_axes);
    }
    
    for (int j = 0; j < db.nbones(); j++)
    {
        fit_kdop_limits(
            kdop_limit_mins_twist(j),
            kdop_limit_maxs_twist(j),
            limit_positions_twist(j),
            limit_rotations_twist(j),
            subsampled_limit_space_rotations_twist[j],
            kdop_axes);
    }
    
    // Local Adjustment Data
  
    array1d<quat> pose_reference_space_rotations = reference_space_rotations(frame_index);
    array1d<quat> pose_reference_space_rotations_swing = reference_space_rotations_swing(frame_index);
    array1d<quat> pose_reference_space_rotations_twist = reference_space_rotations_twist(frame_index);
    
    array1d<vec3> pose_limit_space_rotations(db.nbones());
    array1d<vec3> pose_limit_space_rotations_swing(db.nbones());
    array1d<vec3> pose_limit_space_rotations_twist(db.nbones());
    
    for (int i = 0; i < db.nbones(); i++)
    {
        pose_limit_space_rotations(i) = quat_to_scaled_angle_axis(pose_reference_space_rotations(i));
        pose_limit_space_rotations_swing(i) = quat_to_scaled_angle_axis(pose_reference_space_rotations_swing(i));
        pose_limit_space_rotations_twist(i) = quat_to_scaled_angle_axis(pose_reference_space_rotations_twist(i));
    }
    
    array1d<vec3> pose_limit_space_rotations_projected = pose_limit_space_rotations;
    array1d<vec3> pose_limit_space_rotations_projected_swing = pose_limit_space_rotations_swing;
    array1d<vec3> pose_limit_space_rotations_projected_twist = pose_limit_space_rotations_twist;
    
    float dt = 1.0f / 60.0f;
    
    
    // Go
    
    auto update_func = [&]()
    {

        orbit_camera_update(
            camera, 
            camera_azimuth,
            camera_altitude,
            camera_distance,
            vec3(1.5, 1, 0),
            (IsKeyDown(KEY_LEFT_CONTROL) && IsMouseButtonDown(0)) ? GetMouseDelta().x : 0.0f,
            (IsKeyDown(KEY_LEFT_CONTROL) && IsMouseButtonDown(0)) ? GetMouseDelta().y : 0.0f,
            dt);
  
        //frame_index = (frame_index + 1) % db.range_stops(0);
        
        if (reference_pose)
        {
            adjusted_bone_positions = reference_positions;
            adjusted_bone_rotations = reference_rotations;
        }
        else
        {
            adjusted_bone_positions = db.bone_positions(frame_index);
            adjusted_bone_rotations = db.bone_rotations(frame_index);
        }
        
        if (ik_enabled)
        {
            if (!IsKeyDown(KEY_LEFT_CONTROL) && IsMouseButtonDown(1))
            {
                quat rotation_azimuth = quat_from_angle_axis(camera_azimuth, vec3(0, 1, 0));
                ik_target = ik_target + dt * 0.1f * quat_mul_vec3(rotation_azimuth, vec3(GetMouseDelta().x, -GetMouseDelta().y, 0.0f));
            }
          
            global_bone_computed.zero();
          
            // Compute toe, heel, knee, hip, and root bone positions
            for (int bone : {Bone_Hips, Bone_RightUpLeg, Bone_RightLeg, Bone_RightFoot, Bone_RightToe})
            {
                forward_kinematics_partial(
                    global_bone_positions,
                    global_bone_rotations,
                    global_bone_computed,
                    adjusted_bone_positions,
                    adjusted_bone_rotations,
                    db.bone_parents,
                    bone);
            }
            
            // Perform simple two-joint IK to place heel
            ik_two_bone(
                adjusted_bone_rotations(Bone_RightUpLeg),
                adjusted_bone_rotations(Bone_RightLeg),
                global_bone_positions(Bone_RightUpLeg),
                global_bone_positions(Bone_RightLeg),
                global_bone_positions(Bone_RightFoot),
                ik_target + (global_bone_positions(Bone_RightFoot) - global_bone_positions(Bone_RightToe)),
                quat_mul_vec3(global_bone_rotations(Bone_RightLeg), vec3(0.0f, 1.0f, 0.0f)),
                global_bone_rotations(Bone_RightUpLeg),
                global_bone_rotations(Bone_RightLeg),
                global_bone_rotations(Bone_Hips),
                ik_max_length_buffer);
            
            // Re-compute toe, heel, and knee positions 
            global_bone_computed.zero();
            
            for (int bone : {Bone_RightToe, Bone_RightFoot, Bone_RightLeg})
            {
                forward_kinematics_partial(
                    global_bone_positions,
                    global_bone_rotations,
                    global_bone_computed,
                    adjusted_bone_positions,
                    adjusted_bone_rotations,
                    db.bone_parents,
                    bone);
            }
            
            // Rotate heel so toe is facing toward contact point
            ik_look_at(
                adjusted_bone_rotations(Bone_RightFoot),
                global_bone_rotations(Bone_RightLeg),
                global_bone_rotations(Bone_RightFoot),
                global_bone_positions(Bone_RightFoot),
                global_bone_positions(Bone_RightToe),
                ik_target);
            
            // Re-compute toe and heel positions
            global_bone_computed.zero();
            
            for (int bone : {Bone_RightToe, Bone_RightFoot})
            {
                forward_kinematics_partial(
                    global_bone_positions,
                    global_bone_rotations,
                    global_bone_computed,
                    adjusted_bone_positions,
                    adjusted_bone_rotations,
                    db.bone_parents,
                    bone);
            }
            
            // Rotate toe bone so that the end of the toe 
            // does not intersect with the ground
            vec3 toe_end_curr = quat_mul_vec3(
                global_bone_rotations(Bone_RightToe), vec3(ik_toe_length, 0.0f, 0.0f)) + 
                global_bone_positions(Bone_RightToe);
                
            vec3 toe_end_targ = toe_end_curr;
            toe_end_targ.y = maxf(toe_end_targ.y, ik_foot_height);
            
            ik_look_at(
                adjusted_bone_rotations(Bone_RightToe),
                global_bone_rotations(Bone_RightFoot),
                global_bone_rotations(Bone_RightToe),
                global_bone_positions(Bone_RightToe),
                toe_end_curr,
                toe_end_targ);
        }
        else if (lookat_enabled)
        {
            if (!IsKeyDown(KEY_LEFT_CONTROL) && IsMouseButtonDown(1))
            {
                quat rotation_azimuth = quat_from_angle_axis(camera_azimuth, vec3(0, 1, 0));
                lookat_target = lookat_target + dt * 0.1f * quat_mul_vec3(rotation_azimuth, vec3(GetMouseDelta().x, -GetMouseDelta().y, 0.0f));
            }
            
            global_bone_computed.zero();
            
            forward_kinematics_partial(
                global_bone_positions,
                global_bone_rotations,
                global_bone_computed,
                adjusted_bone_positions,
                adjusted_bone_rotations,
                db.bone_parents,
                Bone_Head);
            
            vec3 lookat_direction = normalize(lookat_target - global_bone_positions(Bone_Head));
            
            float spine_scale = 2.0f;
            float look_angle_azimuth = spine_scale * lookat_direction.x;
            float look_angle_altitude = spine_scale * -lookat_direction.y;
            
            adjusted_bone_rotations(Bone_Spine) = quat_mul(
                adjusted_bone_rotations(Bone_Spine),
                quat_from_angle_axis(0.1f * look_angle_azimuth, vec3(1, 0, 0)));
            
            adjusted_bone_rotations(Bone_Spine1) = quat_mul(
                adjusted_bone_rotations(Bone_Spine1),
                quat_from_angle_axis(0.2f * look_angle_azimuth, vec3(1, 0, 0)));
            
            adjusted_bone_rotations(Bone_Spine2) = quat_mul(
                adjusted_bone_rotations(Bone_Spine2),
                quat_from_angle_axis(0.3f * look_angle_azimuth, vec3(1, 0, 0)));
            
            adjusted_bone_rotations(Bone_Neck) = quat_mul(
                adjusted_bone_rotations(Bone_Neck),
                quat_from_angle_axis(0.4f * look_angle_azimuth, vec3(1, 0, 0)));
            
            adjusted_bone_rotations(Bone_Spine) = quat_mul(
                adjusted_bone_rotations(Bone_Spine),
                quat_from_angle_axis(0.1f * look_angle_altitude, vec3(0, 0, 1)));
            
            adjusted_bone_rotations(Bone_Spine1) = quat_mul(
                adjusted_bone_rotations(Bone_Spine1),
                quat_from_angle_axis(0.2f * look_angle_altitude, vec3(0, 0, 1)));
            
            adjusted_bone_rotations(Bone_Spine2) = quat_mul(
                adjusted_bone_rotations(Bone_Spine2),
                quat_from_angle_axis(0.3f * look_angle_altitude, vec3(0, 0, 1)));
            
            adjusted_bone_rotations(Bone_Neck) = quat_mul(
                adjusted_bone_rotations(Bone_Neck),
                quat_from_angle_axis(0.4f * look_angle_altitude, vec3(0, 0, 1)));
            
            global_bone_computed.zero();
            
            for (int bone : {Bone_Head, Bone_Neck})
            {
                forward_kinematics_partial(
                    global_bone_positions,
                    global_bone_rotations,
                    global_bone_computed,
                    adjusted_bone_positions,
                    adjusted_bone_rotations,
                    db.bone_parents,
                    bone);
            }
            
            vec3 head_lookat_curr = quat_mul_vec3(
                global_bone_rotations(Bone_Head), vec3(0.0f, 1.0f, 0.0f)) + 
                global_bone_positions(Bone_Head);
                
            vec3 head_lookat_targ = normalize(lookat_target - global_bone_positions(Bone_Head)) + 
                global_bone_positions(Bone_Head);
            
            ik_look_at(
                adjusted_bone_rotations(Bone_Head),
                global_bone_rotations(Bone_Neck),
                global_bone_rotations(Bone_Head),
                global_bone_positions(Bone_Head),
                head_lookat_curr,
                head_lookat_targ);
        }
        else
        {
            adjusted_bone_rotations(joint_index) = quat_mul(
                adjusted_bone_rotations(joint_index),
                quat_from_euler_xyz(rotation_x, rotation_y, rotation_z));
        }
  
        // Reference Space
        
        for (int i = 0; i < pose_reference_space_rotations.size; i++)
        {
            pose_reference_space_rotations(i) = quat_inv_mul(
                reference_rotations(i), 
                adjusted_bone_rotations(i));
                
            quat_swing_twist(
                pose_reference_space_rotations_swing(i), 
                pose_reference_space_rotations_twist(i),
                pose_reference_space_rotations(i),
                twist_axes(i));
        }

        // Unroll
        
        for (int i = 0; i < pose_reference_space_rotations.size; i++)
        {
            pose_reference_space_rotations(i) = quat_abs(pose_reference_space_rotations(i));
            pose_reference_space_rotations_swing(i) = quat_abs(pose_reference_space_rotations_swing(i));
            pose_reference_space_rotations_twist(i) = quat_abs(pose_reference_space_rotations_twist(i));
        }
        
        // Limit Space
        
        for (int i = 0; i < pose_reference_space_rotations.size; i++)
        {
            pose_limit_space_rotations(i) = quat_to_scaled_angle_axis(pose_reference_space_rotations(i));
            pose_limit_space_rotations_swing(i) = quat_to_scaled_angle_axis(pose_reference_space_rotations_swing(i));
            pose_limit_space_rotations_twist(i) = quat_to_scaled_angle_axis(pose_reference_space_rotations_twist(i));
        }
        
        // Project
        
        if (limit_swing_twist)
        {
            if (limit_type == LIMIT_TYPE_RECTANGULAR)
            {
                for (int i = 0; i < db.nbones(); i++)
                {
                    pose_limit_space_rotations_projected_swing(i) = apply_rectangular_limit(
                        pose_limit_space_rotations_swing(i),
                        rectangular_limit_mins_swing(i),
                        rectangular_limit_maxs_swing(i),
                        limit_positions_swing(i),
                        limit_rotations_swing(i));
                }
                
                for (int i = 0; i < db.nbones(); i++)
                {
                    pose_limit_space_rotations_projected_twist(i) = apply_rectangular_limit(
                        pose_limit_space_rotations_twist(i),
                        rectangular_limit_mins_twist(i),
                        rectangular_limit_maxs_twist(i),
                        limit_positions_twist(i),
                        limit_rotations_twist(i));
                }
            }
            else if (limit_type == LIMIT_TYPE_ELLIPSOID)
            {
                for (int i = 0; i < db.nbones(); i++)
                {
                    pose_limit_space_rotations_projected_swing(i) = apply_ellipsoid_limit(
                        pose_limit_space_rotations_swing(i),
                        ellipsoid_limit_scales_swing(i),
                        limit_positions_swing(i),
                        limit_rotations_swing(i));
                }
                
                for (int i = 0; i < db.nbones(); i++)
                {
                    pose_limit_space_rotations_projected_twist(i) = apply_ellipsoid_limit(
                        pose_limit_space_rotations_twist(i),
                        ellipsoid_limit_scales_twist(i),
                        limit_positions_twist(i),
                        limit_rotations_twist(i));
                }
            }
            else if (limit_type == LIMIT_TYPE_KDOP)
            {
                for (int i = 0; i < db.nbones(); i++)
                {
                    pose_limit_space_rotations_projected_swing(i) = apply_kdop_limit(
                        pose_limit_space_rotations_swing(i),
                        kdop_limit_mins_swing(i),
                        kdop_limit_maxs_swing(i),
                        limit_positions_swing(i),
                        limit_rotations_swing(i),
                        kdop_axes);
                }
                
                for (int i = 0; i < db.nbones(); i++)
                {
                    pose_limit_space_rotations_projected_twist(i) = apply_kdop_limit(
                        pose_limit_space_rotations_twist(i),
                        kdop_limit_mins_twist(i),
                        kdop_limit_maxs_twist(i),
                        limit_positions_twist(i),
                        limit_rotations_twist(i),
                        kdop_axes);
                }
            }
            
            for (int i = 0; i < db.nbones(); i++)
            {
                pose_limit_space_rotations_projected_swing(i) = projection_soften(
                    pose_limit_space_rotations_swing(i),
                    pose_limit_space_rotations_projected_swing(i),
                    projection_soften_falloff,
                    projection_soften_radius);
              
                pose_limit_space_rotations_projected_twist(i) = projection_soften(
                    pose_limit_space_rotations_twist(i),
                    pose_limit_space_rotations_projected_twist(i),
                    projection_soften_falloff,
                    projection_soften_radius);
            }
        }
        else
        {
            if (limit_type == LIMIT_TYPE_RECTANGULAR)
            {
                for (int i = 0; i < db.nbones(); i++)
                {
                    pose_limit_space_rotations_projected(i) = apply_rectangular_limit(
                        pose_limit_space_rotations(i),
                        rectangular_limit_mins(i),
                        rectangular_limit_maxs(i),
                        limit_positions(i),
                        limit_rotations(i));
                }
            }
            else if (limit_type == LIMIT_TYPE_ELLIPSOID)
            {
                for (int i = 0; i < db.nbones(); i++)
                {
                    pose_limit_space_rotations_projected(i) = apply_ellipsoid_limit(
                        pose_limit_space_rotations(i),
                        ellipsoid_limit_scales(i),
                        limit_positions(i),
                        limit_rotations(i));
                }
            }
            else if (limit_type == LIMIT_TYPE_KDOP)
            {
                for (int i = 0; i < db.nbones(); i++)
                {
                    pose_limit_space_rotations_projected(i) = apply_kdop_limit(
                        pose_limit_space_rotations(i),
                        kdop_limit_mins(i),
                        kdop_limit_maxs(i),
                        limit_positions(i),
                        limit_rotations(i),
                        kdop_axes);
                }
            }
            
            for (int i = 0; i < db.nbones(); i++)
            {
                pose_limit_space_rotations_projected(i) = projection_soften(
                    pose_limit_space_rotations(i),
                    pose_limit_space_rotations_projected(i),
                    projection_soften_falloff,
                    projection_soften_radius);
            }
        }
        
        // De-project
        
        if (projection_enabled)
        {   
            if (limit_swing_twist)
            {
                // Don't adjust root joint
                for (int i = 1; i < db.nbones(); i++)
                {
                    adjusted_bone_rotations(i) = quat_mul(
                        reference_rotations(i), 
                        quat_mul(
                            quat_from_scaled_angle_axis(pose_limit_space_rotations_projected_swing(i)),
                            quat_from_scaled_angle_axis(pose_limit_space_rotations_projected_twist(i))));
                }
            }
            else
            {
                // Don't adjust root joint
                for (int i = 1; i < db.nbones(); i++)
                {
                    adjusted_bone_rotations(i) = quat_mul(
                        reference_rotations(i), 
                        quat_from_scaled_angle_axis(pose_limit_space_rotations_projected(i)));
                }
            }
        }
        
        // Done!
        
        forward_kinematics_full(
            global_bone_positions,
            global_bone_rotations,
            adjusted_bone_positions,
            adjusted_bone_rotations,
            db.bone_parents);

        // Render
        
        BeginDrawing();
        ClearBackground(RAYWHITE);
        
        BeginMode3D(camera);
        
        // Draw Joint Limit Data
        
        if (ik_enabled)
        {
            DrawSphereWires(
                to_Vector3(ik_target),
                0.025, 4, 8,
                VIOLET);
        }
        else if (lookat_enabled)
        {
            DrawLine3D(
                to_Vector3(global_bone_positions(Bone_Head)),
                to_Vector3(lookat_target),
                VIOLET);
          
            DrawSphereWires(
                to_Vector3(lookat_target),
                0.025, 4, 8,
                VIOLET);
        }
        
        float scale = 0.25f;
        
        if (limit_swing_twist)
        {
            vec3 space_offset_swing = vec3(1.25f, 1.0f, 0.0f);
            vec3 space_offset_twist = vec3(3.0f, 1.0f, 0.0f);
    
            draw_current_limit(
                pose_limit_space_rotations_projected_swing(joint_index),
                pose_limit_space_rotations_swing(joint_index),
                space_offset_swing, 
                scale);
              
            draw_current_limit(
                pose_limit_space_rotations_projected_twist(joint_index),
                pose_limit_space_rotations_twist(joint_index),
                space_offset_twist, 
                scale);
              
            draw_limit_samples(
                subsampled_limit_space_rotations_swing[joint_index],
                limit_positions_swing(joint_index),
                limit_rotations_swing(joint_index),
                space_offset_swing, 
                scale);
                
            draw_limit_samples(
                subsampled_limit_space_rotations_twist[joint_index],
                limit_positions_twist(joint_index),
                limit_rotations_twist(joint_index),
                space_offset_twist, 
                scale);
              
            if (limit_type == LIMIT_TYPE_RECTANGULAR)
            {
                draw_rectangular_limit_bounds(
                    limit_positions_swing(joint_index),
                    limit_rotations_swing(joint_index),
                    rectangular_limit_mins_swing(joint_index),
                    rectangular_limit_maxs_swing(joint_index),
                    space_offset_swing, 
                    scale);
                    
                draw_rectangular_limit_bounds(
                    limit_positions_twist(joint_index),
                    limit_rotations_twist(joint_index),
                    rectangular_limit_mins_twist(joint_index),
                    rectangular_limit_maxs_twist(joint_index),
                    space_offset_twist, 
                    scale);
            }
            else if (limit_type == LIMIT_TYPE_ELLIPSOID)
            {
                draw_ellipsoid_limit_bounds(
                    limit_positions_swing(joint_index),
                    limit_rotations_swing(joint_index),
                    ellipsoid_limit_scales_swing(joint_index),
                    space_offset_swing, 
                    scale);
                    
                  draw_ellipsoid_limit_bounds(
                    limit_positions_twist(joint_index),
                    limit_rotations_twist(joint_index),
                    ellipsoid_limit_scales_twist(joint_index),
                    space_offset_twist, 
                    scale);
            }
            else if (limit_type == LIMIT_TYPE_KDOP)
            {
                draw_kdop_limit_bounds(
                    limit_positions_swing(joint_index),
                    limit_rotations_swing(joint_index),
                    kdop_limit_mins_swing(joint_index),
                    kdop_limit_maxs_swing(joint_index),
                    kdop_axes,
                    space_offset_swing, 
                    scale);
                    
                draw_kdop_limit_bounds(
                    limit_positions_twist(joint_index),
                    limit_rotations_twist(joint_index),
                    kdop_limit_mins_twist(joint_index),
                    kdop_limit_maxs_twist(joint_index),
                    kdop_axes,
                    space_offset_twist, 
                    scale);
            }
            else
            {
                assert(false);
            }
              
        }
        else
        {
            vec3 space_offset = vec3(1.5f, 1.0f, 0.0f);
            
            draw_current_limit(
                pose_limit_space_rotations_projected(joint_index),
                pose_limit_space_rotations(joint_index),
                space_offset, 
                scale);
            
            draw_limit_samples(
                subsampled_limit_space_rotations[joint_index],
                limit_positions(joint_index),
                limit_rotations(joint_index),
                space_offset, 
                scale);
              
            if (limit_type == LIMIT_TYPE_RECTANGULAR)
            {
                draw_rectangular_limit_bounds(
                    limit_positions(joint_index),
                    limit_rotations(joint_index),
                    rectangular_limit_mins(joint_index),
                    rectangular_limit_maxs(joint_index),
                    space_offset, 
                    scale);
            }
            else if (limit_type == LIMIT_TYPE_ELLIPSOID)
            {
                draw_ellipsoid_limit_bounds(
                    limit_positions(joint_index),
                    limit_rotations(joint_index),
                    ellipsoid_limit_scales(joint_index),
                    space_offset, 
                    scale);
            }
            else if (limit_type == LIMIT_TYPE_KDOP)
            {
                draw_kdop_limit_bounds(
                    limit_positions(joint_index),
                    limit_rotations(joint_index),
                    kdop_limit_mins(joint_index),
                    kdop_limit_maxs(joint_index),
                    kdop_axes,
                    space_offset, 
                    scale);
            }
            else
            {
                assert(false);
            }
        }

        // Draw Character
        
        deform_character_mesh(
            character_mesh, 
            character_data, 
            global_bone_positions, 
            global_bone_rotations,
            db.bone_parents);
        
        DrawModel(character_model, (Vector3){0.0f, 0.0f, 0.0f}, 1.0f, RAYWHITE);
        
        draw_skeleton(
            global_bone_positions,
            global_bone_rotations,
            db.bone_parents,
            joint_index);
        
        // Draw Ground Plane
        
        DrawModel(ground_plane_model, (Vector3){0.0f, -0.01f, 0.0f}, 1.0f, WHITE);
        DrawGrid(20, 1.0f);
        draw_axis(vec3(), quat());
        
        EndMode3D();

        // UI
        
        int frame_index_prev = frame_index;
        int joint_index_prev = joint_index;

        frame_index = (int)GuiSliderBar(
            (Rectangle){ 100, 20, 1000, 20 }, 
            "frame index", 
            TextFormat("%4i", frame_index),
            frame_index,
            0, db.range_stops(0));
                
        joint_index = (int)GuiSliderBar(
            (Rectangle){ 100, 50, 300, 20 }, 
            "joint index", 
            TextFormat("%s", BoneNames[joint_index]),
            joint_index,
            1, db.nbones() - 1);

        rotation_x = GuiSliderBar(
            (Rectangle){ 100, 80, 200, 20 }, 
            "rotation x", 
            TextFormat("%3.2f", rotation_x),
            rotation_x,
            -PIf, PIf);  

        rotation_y = GuiSliderBar(
            (Rectangle){ 100, 110, 200, 20 }, 
            "rotation y", 
            TextFormat("%3.2f", rotation_y),
            rotation_y,
            -PIf, PIf);  
            
        rotation_z = GuiSliderBar(
            (Rectangle){ 100, 140, 200, 20 }, 
            "rotation z", 
            TextFormat("%3.2f", rotation_z),
            rotation_z,
            -PIf, PIf);  
        
        reference_pose = GuiCheckBox(
            (Rectangle){ 100, 170, 20, 20 }, 
            "reference pose toggle", 
            reference_pose);
        
        projection_enabled = GuiCheckBox(
            (Rectangle){ 100, 200, 20, 20 }, 
            "projection enabled", 
            projection_enabled);
        
        projection_soften_falloff = GuiSliderBar(
            (Rectangle){ 100, 230, 200, 20 }, 
            "soften falloff", 
            TextFormat("%3.2f", projection_soften_falloff),
            projection_soften_falloff,
            0.0f, 1.0f);  
        
        projection_soften_radius = GuiSliderBar(
            (Rectangle){ 100, 230, 200, 20 }, 
            "soften radius", 
            TextFormat("%3.2f", projection_soften_radius),
            projection_soften_radius,
            0.0f, 1.0f);  
        
        if (GuiDropdownBox(
            (Rectangle){ 100, 260, 100, 20 }, 
            "Rectangular;Ellipsoid;KDop",
            &limit_type,
            limit_type_edit))
        {
            limit_type_edit = !limit_type_edit;
        }
        
        limit_swing_twist = GuiCheckBox(
            (Rectangle){ 220, 260, 20, 20 }, 
            "swing twist decomposition", 
            limit_swing_twist);
      
        EndDrawing();

        if (joint_index != joint_index_prev
        ||  frame_index != frame_index_prev)
        {
            rotation_x = 0.0f;
            rotation_y = 0.0f;
            rotation_z = 0.0f;
        }

    };
    
#if defined(PLATFORM_WEB)
    std::function<void()> u{update_func};
    emscripten_set_main_loop_arg(update_callback, &u, 0, 1);
#else
    while (!WindowShouldClose())
    {
        update_func();
    }
#endif

    // Unload stuff and finish
    UnloadModel(character_model);
    UnloadModel(ground_plane_model);
    UnloadShader(character_shader);
    UnloadShader(ground_plane_shader);

    CloseWindow();

    return 0;
}