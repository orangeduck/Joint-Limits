#pragma once

#include "common.h"
#include "vec.h"
#include "quat.h"
#include "array.h"

#include <assert.h>
#include <float.h>
#include <stdio.h>
#include <math.h>

//--------------------------------------

struct database
{
    array2d<vec3> bone_positions;
    array2d<quat> bone_rotations;
    array1d<int> bone_parents;
    
    array1d<int> range_starts;
    array1d<int> range_stops;
    
    int nframes() const { return bone_positions.rows; }
    int nbones() const { return bone_positions.cols; }
    int nranges() const { return range_starts.size; }
};

void database_load(database& db, const char* filename)
{
    FILE* f = fopen(filename, "rb");
    assert(f != NULL);
    
    array2d_read(db.bone_positions, f);
    array2d_read(db.bone_rotations, f);
    array1d_read(db.bone_parents, f);
    
    array1d_read(db.range_starts, f);
    array1d_read(db.range_stops, f);
    
    fclose(f);
}

//--------------------------------------

// Here I am using a simple recursive version of forward kinematics
void forward_kinematics(
    vec3& bone_position,
    quat& bone_rotation,
    const slice1d<vec3> bone_positions,
    const slice1d<quat> bone_rotations,
    const slice1d<int> bone_parents,
    const int bone)
{
    if (bone_parents(bone) != -1)
    {
        vec3 parent_position;
        quat parent_rotation;
        
        forward_kinematics(
            parent_position,
            parent_rotation,
            bone_positions,
            bone_rotations,
            bone_parents,
            bone_parents(bone));
        
        bone_position = quat_mul_vec3(parent_rotation, bone_positions(bone)) + parent_position;
        bone_rotation = quat_mul(parent_rotation, bone_rotations(bone));
    }
    else
    {
        bone_position = bone_positions(bone);
        bone_rotation = bone_rotations(bone); 
    }
}

// Compute forward kinematics of just some joints using a
// mask to indicate which joints are already computed
void forward_kinematics_partial(
    slice1d<vec3> global_bone_positions,
    slice1d<quat> global_bone_rotations,
    slice1d<bool> global_bone_computed,
    const slice1d<vec3> local_bone_positions,
    const slice1d<quat> local_bone_rotations,
    const slice1d<int> bone_parents,
    int bone)
{
    if (global_bone_computed(bone))
    {
        return;
    }
  
    if (bone_parents(bone) == -1)
    {
        global_bone_positions(bone) = local_bone_positions(bone);
        global_bone_rotations(bone) = local_bone_rotations(bone);
        global_bone_computed(bone) = true;
        return;
    }
    
    if (!global_bone_computed(bone_parents(bone)))
    {
        forward_kinematics_partial(
            global_bone_positions,
            global_bone_rotations,
            global_bone_computed,
            local_bone_positions,
            local_bone_rotations,
            bone_parents,
            bone_parents(bone));
    }
    
    vec3 parent_position = global_bone_positions(bone_parents(bone));
    quat parent_rotation = global_bone_rotations(bone_parents(bone));
    global_bone_positions(bone) = quat_mul_vec3(parent_rotation, local_bone_positions(bone)) + parent_position;
    global_bone_rotations(bone) = quat_mul(parent_rotation, local_bone_rotations(bone));
    global_bone_computed(bone) = true;
}

// Compute forward kinematics for all joints
void forward_kinematics_full(
    slice1d<vec3> global_bone_positions,
    slice1d<quat> global_bone_rotations,
    const slice1d<vec3> local_bone_positions,
    const slice1d<quat> local_bone_rotations,
    const slice1d<int> bone_parents)
{
    for (int i = 0; i < bone_parents.size; i++)
    {
        // Assumes bones are always sorted from root onwards
        assert(bone_parents(i) < i);
        
        if (bone_parents(i) == -1)
        {
            global_bone_positions(i) = local_bone_positions(i);
            global_bone_rotations(i) = local_bone_rotations(i);
        }
        else
        {
            vec3 parent_position = global_bone_positions(bone_parents(i));
            quat parent_rotation = global_bone_rotations(bone_parents(i));
            global_bone_positions(i) = quat_mul_vec3(parent_rotation, local_bone_positions(i)) + parent_position;
            global_bone_rotations(i) = quat_mul(parent_rotation, local_bone_rotations(i));
        }
    }
}

// Compute backward kinematics for all joints
void backward_kinematics_full(
    slice1d<vec3> local_bone_positions,
    slice1d<quat> local_bone_rotations,
    const slice1d<vec3> global_bone_positions,
    const slice1d<quat> global_bone_rotations,
    const slice1d<int> bone_parents)
{
    for (int i = 0; i < bone_parents.size; i++)
    {
        if (bone_parents(i) == -1)
        {
            local_bone_positions(i) = global_bone_positions(i);
            local_bone_rotations(i) = global_bone_rotations(i);
        }
        else
        {
            vec3 parent_position = global_bone_positions(bone_parents(i));
            quat parent_rotation = global_bone_rotations(bone_parents(i));

            local_bone_positions(i) = quat_inv_mul_vec3(parent_rotation,
                global_bone_positions(i) - parent_position);
            local_bone_rotations(i) = quat_inv_mul(parent_rotation, global_bone_rotations(i));
        }
    }
}
