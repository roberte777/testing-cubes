use bevy::{
    prelude::*,
    tasks::Task,
};
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};
use noise::{NoiseFn, Perlin};

use crate::components::chunk::{ChunkCoord, VoxelChunk};
use crate::components::voxel::VoxelType;

/// Main resource for managing the voxel world, including chunks and terrain generation
#[derive(Resource)]
pub struct VoxelWorld {
    /// Map of chunk coordinates to chunk data
    pub chunks: HashMap<ChunkCoord, Arc<Mutex<VoxelChunk>>>,
    /// Set of currently loaded chunk coordinates
    pub loaded_chunks: HashSet<ChunkCoord>,
    /// List of async tasks generating chunk meshes
    pub chunk_mesh_tasks: Vec<(ChunkCoord, Task<(Mesh, ChunkCoord, VoxelType)>)>,
    /// Noise generator for terrain generation
    pub terrain_generator: Perlin,
}

impl Default for VoxelWorld {
    fn default() -> Self {
        Self {
            chunks: HashMap::new(),
            loaded_chunks: HashSet::new(),
            chunk_mesh_tasks: Vec::new(),
            terrain_generator: Perlin::new(42), // Fixed seed for reproducibility
        }
    }
} 