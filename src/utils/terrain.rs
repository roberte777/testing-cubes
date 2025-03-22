use noise::{NoiseFn, Perlin};

use crate::components::chunk::VoxelChunk;
use crate::components::voxel::VoxelType;
use crate::constants::CHUNK_SIZE;

/// Generates terrain for a chunk using Perlin noise
/// Creates a heightmap-based terrain with layers of stone, dirt, and grass
pub fn generate_terrain(chunk: &mut VoxelChunk, noise_fn: &Perlin) {
    let world_scale = 0.02;
    let height_scale = 32.0;

    // Create a clear area around spawn point (for debugging)
    let spawn_chunk_range = 1; // Clear 1 chunk in each direction from spawn
    let is_spawn_area = 
        chunk.coord.x >= -spawn_chunk_range && chunk.coord.x <= spawn_chunk_range &&
        chunk.coord.z >= -spawn_chunk_range && chunk.coord.z <= spawn_chunk_range &&
        chunk.coord.y >= -spawn_chunk_range && chunk.coord.y <= spawn_chunk_range;
    
    if is_spawn_area {
        // Fill the spawn area with air for testing
        for x in 0..CHUNK_SIZE as usize {
            for y in 0..CHUNK_SIZE as usize {
                for z in 0..CHUNK_SIZE as usize {
                    chunk.set_voxel(x, y, z, VoxelType::Air);
                }
            }
        }
        
        // Add a platform at y=0 if this is the bottom chunk
        if chunk.coord.y == -1 {
            for x in 0..CHUNK_SIZE as usize {
                for z in 0..CHUNK_SIZE as usize {
                    // Create a flat platform at the top of this chunk
                    chunk.set_voxel(x, (CHUNK_SIZE-1) as usize, z, VoxelType::Stone);
                }
            }
        }
        return;
    }

    for x in 0..CHUNK_SIZE {
        for z in 0..CHUNK_SIZE {
            // Calculate the world position for the noise function
            let world_x = (chunk.coord.x * CHUNK_SIZE + x) as f64 * world_scale;
            let world_z = (chunk.coord.z * CHUNK_SIZE + z) as f64 * world_scale;
            
            // Generate a height value using Perlin noise
            let noise_val = noise_fn.get([world_x, world_z]);
            let terrain_height = ((noise_val + 1.0) * 0.5 * height_scale) as i32;
            
            // Calculate absolute height in world space (independent of chunk y-coordinate)
            let absolute_terrain_height = terrain_height;
            
            for y in 0..CHUNK_SIZE {
                // Calculate the absolute y position in world coordinates
                let absolute_y = chunk.coord.y * CHUNK_SIZE + y;
                
                if absolute_y < absolute_terrain_height - 3 {
                    // Stone below the surface
                    chunk.set_voxel(x as usize, y as usize, z as usize, VoxelType::Stone);
                } else if absolute_y < absolute_terrain_height - 1 {
                    // Dirt layer
                    chunk.set_voxel(x as usize, y as usize, z as usize, VoxelType::Dirt);
                } else if absolute_y < absolute_terrain_height {
                    // Grass on top
                    chunk.set_voxel(x as usize, y as usize, z as usize, VoxelType::Grass);
                } else {
                    // Air above the surface
                    chunk.set_voxel(x as usize, y as usize, z as usize, VoxelType::Air);
                }
            }
        }
    }
} 