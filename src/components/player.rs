use bevy::prelude::*;
use crate::components::chunk::ChunkCoord;

/// Player component containing position and view information
#[derive(Component)]
pub struct Player {
    pub chunk_coord: ChunkCoord,
    pub yaw: f32,
    pub pitch: f32,
}

/// Component to mark the skybox
#[derive(Component)]
pub struct Skybox; 