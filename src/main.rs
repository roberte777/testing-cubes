use bevy::prelude::*;

use testing_cubes::resources::{
    assets::VoxelAssets,
    world::VoxelWorld,
};
use testing_cubes::systems::{
    player::{player_movement, camera_control, cursor_grab},
    world::{chunk_loading, process_chunk_mesh_tasks, update_chunks},
    rendering::update_skybox,
    setup::setup,
};

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .init_resource::<VoxelAssets>()
        .init_resource::<VoxelWorld>()
        .add_systems(Startup, setup)
        .add_systems(
            Update,
            (
                player_movement,
                camera_control,
                cursor_grab,
                chunk_loading,
                process_chunk_mesh_tasks,
                update_chunks,
                update_skybox,
            ),
        )
        .run();
}