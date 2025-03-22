use bevy::{
    prelude::*,
    render::{mesh::Indices, render_resource::PrimitiveTopology, render_asset::RenderAssetUsages},
};
use std::{
    collections::HashSet,
    sync::{Arc, Mutex},
};

use crate::components::chunk::{ChunkCoord, VoxelChunk};
use crate::components::voxel::VoxelType;
use crate::constants::{CHUNK_SIZE, VOXEL_SIZE, DIRECTIONS, DIR_NORMALS, FACE_VERTICES, FACE_UVS};

/// Generates a mesh for a chunk based on its voxel data
/// Creates vertices, normals, UVs, and indices for visible voxel faces
pub fn generate_chunk_mesh(chunk_arc: &Arc<Mutex<VoxelChunk>>, chunk_coord: ChunkCoord) -> (Mesh, ChunkCoord, VoxelType) {
    let chunk = chunk_arc.lock().unwrap();
    
    // Vertices, normals, uvs, and indices for the mesh
    let mut vertices: Vec<[f32; 3]> = Vec::new();
    let mut normals: Vec<[f32; 3]> = Vec::new();
    let mut uvs: Vec<[f32; 2]> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();
    
    // Keep track of voxel types in this chunk for material selection
    let mut chunk_voxel_types = HashSet::new();
    let mut primary_voxel_type = VoxelType::Stone; // Default fallback
    
    // For each voxel in the chunk
    for x in 0..CHUNK_SIZE as usize {
        for y in 0..CHUNK_SIZE as usize {
            for z in 0..CHUNK_SIZE as usize {
                let voxel_type = chunk.get_voxel(x, y, z);
                
                // Skip air blocks since they're invisible
                if *voxel_type == VoxelType::Air {
                    continue;
                }
                
                // Track voxel types in this chunk
                chunk_voxel_types.insert(*voxel_type);
                primary_voxel_type = *voxel_type; // Use the last non-air voxel type as fallback
                
                // Check each of the six faces
                for (face_idx, (dx, dy, dz)) in DIRECTIONS.iter().enumerate() {
                    let nx = x as i32 + dx;
                    let ny = y as i32 + dy;
                    let nz = z as i32 + dz;
                    
                    // If the adjacent block is outside the chunk or is air, add a face
                    let should_draw_face = 
                        nx < 0 || nx >= CHUNK_SIZE || 
                        ny < 0 || ny >= CHUNK_SIZE || 
                        nz < 0 || nz >= CHUNK_SIZE ||
                        *chunk.get_voxel(nx as usize, ny as usize, nz as usize) == VoxelType::Air;
                    
                    if should_draw_face {
                        let vertex_offset = vertices.len() as u32;
                        
                        // Add the four vertices for this face
                        for vertex in &FACE_VERTICES[face_idx] {
                            vertices.push([
                                (x as f32 + vertex[0]) * VOXEL_SIZE,
                                (y as f32 + vertex[1]) * VOXEL_SIZE,
                                (z as f32 + vertex[2]) * VOXEL_SIZE,
                            ]);
                            
                            // Add the normal for this face
                            normals.push(DIR_NORMALS[face_idx]);
                        }
                        
                        // Add UVs for the face (could be extended to support texture atlases)
                        for uv in &FACE_UVS {
                            uvs.push(*uv);
                        }
                        
                        // Add indices for two triangles to form the face
                        indices.push(vertex_offset);
                        indices.push(vertex_offset + 1);
                        indices.push(vertex_offset + 2);
                        indices.push(vertex_offset);
                        indices.push(vertex_offset + 2);
                        indices.push(vertex_offset + 3);
                    }
                }
            }
        }
    }
    
    // Choose the most representative voxel type for the chunk
    // This is a simple approach - a better one would be to split the mesh by voxel type
    if chunk_voxel_types.contains(&VoxelType::Grass) {
        primary_voxel_type = VoxelType::Grass;
    } else if chunk_voxel_types.contains(&VoxelType::Dirt) {
        primary_voxel_type = VoxelType::Dirt;
    } else if chunk_voxel_types.contains(&VoxelType::Stone) {
        primary_voxel_type = VoxelType::Stone;
    }
    
    // Create the mesh
    let mut mesh = Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::default());
    
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, vertices);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uvs);
    mesh.insert_indices(Indices::U32(indices));
    
    (mesh, chunk_coord, primary_voxel_type)
} 