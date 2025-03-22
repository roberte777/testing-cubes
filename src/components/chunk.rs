use bevy::prelude::*;
use crate::constants::{CHUNK_SIZE, VOXEL_SIZE};
use crate::components::voxel::VoxelType;

/// 3D coordinate for a chunk in the world
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct ChunkCoord {
    pub x: i32,
    pub y: i32,
    pub z: i32,
}

impl ChunkCoord {
    /// Creates a new chunk coordinate
    pub fn new(x: i32, y: i32, z: i32) -> Self {
        Self { x, y, z }
    }

    /// Calculates the squared distance between two chunk coordinates
    /// Used for determining which chunks to load/unload
    pub fn distance_sq(&self, other: &ChunkCoord) -> i32 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        let dz = self.z - other.z;
        dx * dx + dy * dy + dz * dz
    }

    /// Converts chunk coordinates to world position
    pub fn to_world_pos(&self) -> Vec3 {
        Vec3::new(
            self.x as f32 * CHUNK_SIZE as f32 * VOXEL_SIZE,
            self.y as f32 * CHUNK_SIZE as f32 * VOXEL_SIZE,
            self.z as f32 * CHUNK_SIZE as f32 * VOXEL_SIZE,
        )
    }
}

/// Represents a single chunk of voxels in the world
pub struct VoxelChunk {
    /// Position of this chunk in the world
    pub coord: ChunkCoord,
    /// 3D array of voxel types in this chunk
    pub voxels: [[[VoxelType; CHUNK_SIZE as usize]; CHUNK_SIZE as usize]; CHUNK_SIZE as usize],
    /// Entity ID for the rendered mesh of this chunk
    pub entity: Option<Entity>,
    /// Flag indicating if the mesh needs to be regenerated
    pub needs_mesh_update: bool,
}

impl VoxelChunk {
    /// Creates a new chunk at the specified coordinates
    pub fn new(coord: ChunkCoord) -> Self {
        Self {
            coord,
            voxels: [[[VoxelType::Air; CHUNK_SIZE as usize]; CHUNK_SIZE as usize]; CHUNK_SIZE as usize],
            entity: None,
            needs_mesh_update: true,
        }
    }

    /// Sets a voxel at the specified position in this chunk
    pub fn set_voxel(&mut self, x: usize, y: usize, z: usize, voxel_type: VoxelType) {
        if x < CHUNK_SIZE as usize && y < CHUNK_SIZE as usize && z < CHUNK_SIZE as usize {
            self.voxels[x][y][z] = voxel_type;
            self.needs_mesh_update = true;
        }
    }

    /// Gets the voxel at the specified position in this chunk
    pub fn get_voxel(&self, x: usize, y: usize, z: usize) -> &VoxelType {
        &self.voxels[x][y][z]
    }
} 