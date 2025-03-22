use bevy::prelude::*;
use std::collections::HashMap;
use crate::components::voxel::VoxelType;

/// Stores material assets for different voxel types
#[derive(Resource)]
pub struct VoxelAssets {
    pub materials: HashMap<VoxelType, Handle<StandardMaterial>>,
}

impl Default for VoxelAssets {
    fn default() -> Self {
        Self {
            materials: HashMap::new(),
        }
    }
} 