use bevy::{
    prelude::*,
    tasks::AsyncComputeTaskPool,
};
use std::sync::{Arc, Mutex};
use futures_lite;

use crate::components::player::Player;
use crate::components::chunk::{ChunkCoord, VoxelChunk};
use crate::resources::world::VoxelWorld;
use crate::resources::assets::VoxelAssets;
use crate::constants::{CHUNK_SIZE, RENDER_DISTANCE};
use crate::utils::mesh::generate_chunk_mesh;
use crate::utils::terrain::generate_terrain;

/// System to handle chunk loading/unloading based on player position
/// Loads chunks within render distance and unloads those outside it
pub fn chunk_loading(
    mut voxel_world: ResMut<VoxelWorld>,
    player_query: Query<&Player>,
    mut commands: Commands,
) {
    let thread_pool = AsyncComputeTaskPool::get();

    for player in player_query.iter() {
        let player_chunk = &player.chunk_coord;
        
        // Determine which chunks should be loaded
        let mut chunks_to_load = std::collections::HashSet::new();
        for x in -RENDER_DISTANCE..=RENDER_DISTANCE {
            for y in -RENDER_DISTANCE..=RENDER_DISTANCE {
                for z in -RENDER_DISTANCE..=RENDER_DISTANCE {
                    let chunk_coord = ChunkCoord::new(player_chunk.x + x, player_chunk.y + y, player_chunk.z + z);
                    
                    // Check if the chunk is within render distance (using squared distance for efficiency)
                    if ChunkCoord::new(x, y, z).distance_sq(&ChunkCoord::new(0, 0, 0)) <= RENDER_DISTANCE * RENDER_DISTANCE {
                        chunks_to_load.insert(chunk_coord);
                    }
                }
            }
        }

        // Load new chunks
        for chunk_coord in &chunks_to_load {
            if !voxel_world.loaded_chunks.contains(chunk_coord) {
                // Create a new chunk if it doesn't exist
                if !voxel_world.chunks.contains_key(chunk_coord) {
                    let mut chunk = VoxelChunk::new(chunk_coord.clone());
                    
                    // Generate terrain for this chunk
                    generate_terrain(&mut chunk, &voxel_world.terrain_generator);
                    
                    voxel_world.chunks.insert(chunk_coord.clone(), Arc::new(Mutex::new(chunk)));
                }
                
                // Mark the chunk as loaded
                voxel_world.loaded_chunks.insert(chunk_coord.clone());
                
                // Schedule mesh generation for this chunk
                if let Some(chunk_arc) = voxel_world.chunks.get(chunk_coord) {
                    let chunk_arc_clone = Arc::clone(chunk_arc);
                    let chunk_coord_clone = chunk_coord.clone();
                    
                    let task = thread_pool.spawn(async move {
                        generate_chunk_mesh(&chunk_arc_clone, chunk_coord_clone)
                    });
                    
                    voxel_world.chunk_mesh_tasks.push((chunk_coord.clone(), task));
                }
            }
        }

        // Unload chunks that are out of range
        let chunks_to_unload: Vec<ChunkCoord> = voxel_world
            .loaded_chunks
            .iter()
            .filter(|coord| !chunks_to_load.contains(*coord))
            .cloned()
            .collect();

        for chunk_coord in chunks_to_unload {
            // Remove the chunk from loaded chunks
            voxel_world.loaded_chunks.remove(&chunk_coord);
            
            // Despawn the chunk entity if it exists
            if let Some(chunk_arc) = voxel_world.chunks.get(&chunk_coord) {
                if let Ok(chunk) = chunk_arc.lock() {
                    if let Some(entity) = chunk.entity {
                        // Make sure the entity exists before trying to despawn it
                        if commands.get_entity(entity).is_some() {
                            commands.entity(entity).despawn_recursive();
                        }
                    }
                }
            }
        }
    }
}

/// System to process completed chunk mesh generation tasks
/// Takes completed meshes and creates/updates entities for them
pub fn process_chunk_mesh_tasks(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    voxel_assets: Res<VoxelAssets>,
    mut voxel_world: ResMut<VoxelWorld>,
) {
    // Process completed mesh tasks
    let mut completed_tasks = Vec::new();
    
    for (i, (chunk_coord, task)) in voxel_world.chunk_mesh_tasks.iter_mut().enumerate() {
        if let Some(mesh_result) = futures_lite::future::block_on(futures_lite::future::poll_once(task)) {
            completed_tasks.push((i, mesh_result));
        }
    }
    
    // Process completed tasks in reverse order to avoid index issues when removing
    for (index, (mesh, chunk_coord, voxel_type)) in completed_tasks.iter().rev() {
        // Remove the task from the list
        let _removed_task = voxel_world.chunk_mesh_tasks.remove(*index);
        
        // Only process if the chunk is still loaded
        if !voxel_world.loaded_chunks.contains(chunk_coord) {
            continue; // Skip this chunk as it's no longer loaded
        }
        
        // Get the material for this voxel type, or use stone as fallback
        let material = voxel_assets.materials.get(voxel_type)
            .unwrap_or_else(|| voxel_assets.materials.get(&crate::components::voxel::VoxelType::Stone).unwrap())
            .clone();
        
        // Add the mesh as an entity
        if let Some(chunk_arc) = voxel_world.chunks.get(chunk_coord) {
            if let Ok(mut chunk) = chunk_arc.lock() {
                // Calculate the world position for this chunk
                let chunk_pos = chunk.coord.to_world_pos();
                
                create_or_update_chunk_entity(
                    &mut commands,
                    &mut meshes,
                    &mut chunk,
                    mesh.clone(),
                    material,
                    chunk_pos,
                );
                
                chunk.needs_mesh_update = false;
            }
        }
    }
}

/// Creates a new chunk entity or updates an existing one with a new mesh
pub fn create_or_update_chunk_entity(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    chunk: &mut VoxelChunk,
    mesh: Mesh,
    material: Handle<StandardMaterial>,
    chunk_pos: Vec3,
) {
    // If the chunk already has an entity, update its mesh
    if let Some(entity) = chunk.entity {
        // Check if the entity still exists before using it
        if commands.get_entity(entity).is_some() {
            commands.entity(entity)
                .insert(meshes.add(mesh))
                .insert(material);
        } else {
            // Entity doesn't exist anymore, create a new one
            let entity = commands
                .spawn(PbrBundle {
                    mesh: meshes.add(mesh),
                    material,
                    transform: Transform::from_translation(chunk_pos),
                    ..default()
                })
                .id();
            
            chunk.entity = Some(entity);
        }
    } else {
        // Otherwise create a new entity
        let entity = commands
            .spawn(PbrBundle {
                mesh: meshes.add(mesh),
                material,
                transform: Transform::from_translation(chunk_pos),
                ..default()
            })
            .id();
        
        chunk.entity = Some(entity);
    }
}

/// System to find chunks that need mesh updates and schedule tasks for them
pub fn update_chunks(
    mut voxel_world: ResMut<VoxelWorld>,
) {
    let thread_pool = AsyncComputeTaskPool::get();
    
    // Find chunks that need a mesh update and collect their coordinates
    let chunks_to_update: Vec<(ChunkCoord, Arc<Mutex<VoxelChunk>>)> = voxel_world.chunks.iter()
        .filter_map(|(coord, chunk_arc)| {
            if let Ok(chunk) = chunk_arc.lock() {
                if chunk.needs_mesh_update && voxel_world.loaded_chunks.contains(coord) {
                    // Find if there's already a task for this chunk
                    let already_has_task = voxel_world
                        .chunk_mesh_tasks
                        .iter()
                        .any(|(task_coord, _)| task_coord == coord);
                    
                    if !already_has_task {
                        return Some((coord.clone(), Arc::clone(chunk_arc)));
                    }
                }
            }
            None
        })
        .collect();
    
    // Create tasks for chunks that need updates
    for (coord, chunk_arc) in chunks_to_update {
        // Create a new task for this chunk
        let chunk_arc_clone = Arc::clone(&chunk_arc);
        let coord_clone = coord.clone();
        
        let task = thread_pool.spawn(async move {
            generate_chunk_mesh(&chunk_arc_clone, coord_clone)
        });
        
        voxel_world.chunk_mesh_tasks.push((coord, task));
    }
} 