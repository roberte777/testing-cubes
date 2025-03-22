use bevy::prelude::Component;

/// Possible types of voxels in the world
#[derive(Clone, Debug, Eq, PartialEq, Copy, Hash)]
pub enum VoxelType {
    Air,
    Dirt,
    Grass,
    Stone,
    // Add more voxel types as needed
} 