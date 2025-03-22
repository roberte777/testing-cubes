use bevy::{
    prelude::*,
    render::{mesh::Indices, render_resource::PrimitiveTopology},
    tasks::{AsyncComputeTaskPool, Task},
};
use noise::{NoiseFn, Perlin};
use std::{
    collections::{HashMap, HashSet},
    sync::{Arc, Mutex},
};
use futures_lite;

// Constants for configuration
const CHUNK_SIZE: i32 = 16;
const RENDER_DISTANCE: i32 = 6;
const VOXEL_SIZE: f32 = 1.0;

// Main function to set up and run the Bevy app
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
                chunk_loading,
                process_chunk_mesh_tasks,
                update_chunks,
            ),
        )
        .run();
}

// Resources
#[derive(Resource)]
struct VoxelAssets {
    material: Handle<StandardMaterial>,
}

impl Default for VoxelAssets {
    fn default() -> Self {
        Self {
            material: Handle::default(),
        }
    }
}

// Voxel and Chunk related data structures
#[derive(Clone, Debug, Eq, PartialEq, Copy)]
enum VoxelType {
    Air,
    Dirt,
    Grass,
    Stone,
    // Add more voxel types as needed
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
struct ChunkCoord {
    x: i32,
    y: i32,
    z: i32,
}

impl ChunkCoord {
    fn new(x: i32, y: i32, z: i32) -> Self {
        Self { x, y, z }
    }

    fn distance_sq(&self, other: &ChunkCoord) -> i32 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        let dz = self.z - other.z;
        dx * dx + dy * dy + dz * dz
    }
}

struct VoxelChunk {
    coord: ChunkCoord,
    voxels: [[[VoxelType; CHUNK_SIZE as usize]; CHUNK_SIZE as usize]; CHUNK_SIZE as usize],
    entity: Option<Entity>,
    needs_mesh_update: bool,
}

impl VoxelChunk {
    fn new(coord: ChunkCoord) -> Self {
        Self {
            coord,
            voxels: [[[VoxelType::Air; CHUNK_SIZE as usize]; CHUNK_SIZE as usize]; CHUNK_SIZE as usize],
            entity: None,
            needs_mesh_update: true,
        }
    }

    fn set_voxel(&mut self, x: usize, y: usize, z: usize, voxel_type: VoxelType) {
        if x < CHUNK_SIZE as usize && y < CHUNK_SIZE as usize && z < CHUNK_SIZE as usize {
            self.voxels[x][y][z] = voxel_type;
            self.needs_mesh_update = true;
        }
    }

    fn get_voxel(&self, x: usize, y: usize, z: usize) -> &VoxelType {
        &self.voxels[x][y][z]
    }
}

#[derive(Resource)]
struct VoxelWorld {
    chunks: HashMap<ChunkCoord, Arc<Mutex<VoxelChunk>>>,
    loaded_chunks: HashSet<ChunkCoord>,
    chunk_mesh_tasks: Vec<(ChunkCoord, Task<(Mesh, ChunkCoord)>)>,
    terrain_generator: Perlin,
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

// Component to mark the player
#[derive(Component)]
struct Player {
    chunk_coord: ChunkCoord,
}

// Setup function to initialize the world
fn setup(
    mut commands: Commands,
    _meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut voxel_assets: ResMut<VoxelAssets>,
) {
    // Create a simple material for voxels
    voxel_assets.material = materials.add(StandardMaterial {
        base_color: Color::rgb(0.8, 0.8, 0.8),
        ..default()
    });

    // Add a camera
    commands.spawn(Camera3dBundle {
        transform: Transform::from_xyz(0.0, 30.0, 30.0).looking_at(Vec3::ZERO, Vec3::Y),
        ..default()
    });

    // Add a light
    commands.spawn(DirectionalLightBundle {
        directional_light: DirectionalLight {
            illuminance: 10000.0,
            shadows_enabled: true,
            ..default()
        },
        transform: Transform::from_rotation(Quat::from_rotation_x(-std::f32::consts::FRAC_PI_4)),
        ..default()
    });

    // Add a player
    commands.spawn((
        Player {
            chunk_coord: ChunkCoord::new(0, 0, 0),
        },
        TransformBundle::from_transform(Transform::from_xyz(0.0, 20.0, 0.0)),
    ));
}

// System to handle player movement
fn player_movement(
    keyboard_input: Res<Input<KeyCode>>,
    time: Res<Time>,
    mut player_query: Query<(&mut Transform, &mut Player)>,
) {
    let speed = 10.0;
    for (mut transform, mut player) in player_query.iter_mut() {
        let mut direction = Vec3::ZERO;

        if keyboard_input.pressed(KeyCode::W) {
            direction.z -= 1.0;
        }
        if keyboard_input.pressed(KeyCode::S) {
            direction.z += 1.0;
        }
        if keyboard_input.pressed(KeyCode::A) {
            direction.x -= 1.0;
        }
        if keyboard_input.pressed(KeyCode::D) {
            direction.x += 1.0;
        }
        if keyboard_input.pressed(KeyCode::Space) {
            direction.y += 1.0;
        }
        if keyboard_input.pressed(KeyCode::ShiftLeft) {
            direction.y -= 1.0;
        }

        if direction != Vec3::ZERO {
            direction = direction.normalize();
            transform.translation += direction * speed * time.delta_seconds();
        }

        // Update player's chunk coordinate
        let chunk_x = (transform.translation.x / (CHUNK_SIZE as f32 * VOXEL_SIZE)).floor() as i32;
        let chunk_y = (transform.translation.y / (CHUNK_SIZE as f32 * VOXEL_SIZE)).floor() as i32;
        let chunk_z = (transform.translation.z / (CHUNK_SIZE as f32 * VOXEL_SIZE)).floor() as i32;
        
        let new_chunk_coord = ChunkCoord::new(chunk_x, chunk_y, chunk_z);
        if new_chunk_coord != player.chunk_coord {
            player.chunk_coord = new_chunk_coord;
        }
    }
}

// System to handle chunk loading/unloading based on player position
fn chunk_loading(
    mut voxel_world: ResMut<VoxelWorld>,
    player_query: Query<&Player>,
    mut commands: Commands,
) {
    let thread_pool = AsyncComputeTaskPool::get();

    for player in player_query.iter() {
        let player_chunk = &player.chunk_coord;
        
        // Determine which chunks should be loaded
        let mut chunks_to_load = HashSet::new();
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
                        commands.entity(entity).despawn_recursive();
                    }
                }
            }
        }
    }
}

// Terrain generation function
fn generate_terrain(chunk: &mut VoxelChunk, noise_fn: &Perlin) {
    let world_scale = 0.02;
    let height_scale = 32.0;

    for x in 0..CHUNK_SIZE {
        for z in 0..CHUNK_SIZE {
            // Calculate the world position for the noise function
            let world_x = (chunk.coord.x * CHUNK_SIZE + x) as f64 * world_scale;
            let world_z = (chunk.coord.z * CHUNK_SIZE + z) as f64 * world_scale;
            
            // Generate a height value using Perlin noise
            let noise_val = noise_fn.get([world_x, world_z]);
            let height = ((noise_val + 1.0) * 0.5 * height_scale) as i32;
            
            // World Y position of the top of this column
            let world_height = height + chunk.coord.y * CHUNK_SIZE;
            
            for y in 0..CHUNK_SIZE {
                let local_y = chunk.coord.y * CHUNK_SIZE + y;
                
                if local_y < world_height - 3 {
                    // Stone below the surface
                    chunk.set_voxel(x as usize, y as usize, z as usize, VoxelType::Stone);
                } else if local_y < world_height - 1 {
                    // Dirt layer
                    chunk.set_voxel(x as usize, y as usize, z as usize, VoxelType::Dirt);
                } else if local_y < world_height {
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

// Function to generate the mesh for a chunk
fn generate_chunk_mesh(chunk_arc: &Arc<Mutex<VoxelChunk>>, chunk_coord: ChunkCoord) -> (Mesh, ChunkCoord) {
    let chunk = chunk_arc.lock().unwrap();
    
    // Vertices, normals, uvs, and indices for the mesh
    let mut vertices: Vec<[f32; 3]> = Vec::new();
    let mut normals: Vec<[f32; 3]> = Vec::new();
    let mut uvs: Vec<[f32; 2]> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();
    
    // Directions for checking adjacent blocks (right, left, up, down, forward, back)
    let directions = [
        (1, 0, 0),  // right
        (-1, 0, 0), // left
        (0, 1, 0),  // up
        (0, -1, 0), // down
        (0, 0, 1),  // forward
        (0, 0, -1), // back
    ];
    
    // Corresponding normals for each direction
    let dir_normals = [
        [1.0, 0.0, 0.0],  // right
        [-1.0, 0.0, 0.0], // left
        [0.0, 1.0, 0.0],  // up
        [0.0, -1.0, 0.0], // down
        [0.0, 0.0, 1.0],  // forward
        [0.0, 0.0, -1.0], // back
    ];
    
    // Offsets for the vertices of a cube face
    let face_vertices = [
        // Right face (x+)
        [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1.0, 1.0, 1.0], [1.0, 0.0, 1.0]],
        // Left face (x-)
        [[0.0, 0.0, 1.0], [0.0, 1.0, 1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
        // Top face (y+)
        [[0.0, 1.0, 0.0], [0.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 0.0]],
        // Bottom face (y-)
        [[0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 1.0]],
        // Front face (z+)
        [[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0]],
        // Back face (z-)
        [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]],
    ];
    
    // UV coordinates for each vertex of a face
    let face_uvs = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
    
    // For each voxel in the chunk
    for x in 0..CHUNK_SIZE as usize {
        for y in 0..CHUNK_SIZE as usize {
            for z in 0..CHUNK_SIZE as usize {
                let voxel_type = chunk.get_voxel(x, y, z);
                
                // Skip air blocks since they're invisible
                if *voxel_type == VoxelType::Air {
                    continue;
                }
                
                // Check each of the six faces
                for (face_idx, (dx, dy, dz)) in directions.iter().enumerate() {
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
                        for vertex in &face_vertices[face_idx] {
                            vertices.push([
                                (x as f32 + vertex[0]) * VOXEL_SIZE,
                                (y as f32 + vertex[1]) * VOXEL_SIZE,
                                (z as f32 + vertex[2]) * VOXEL_SIZE,
                            ]);
                            
                            // Add the normal for this face
                            normals.push(dir_normals[face_idx]);
                        }
                        
                        // Add UVs for the face (could be extended to support texture atlases)
                        for uv in &face_uvs {
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
    
    // Create the mesh
    let mut mesh = Mesh::new(PrimitiveTopology::TriangleList);
    
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, vertices);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uvs);
    mesh.set_indices(Some(Indices::U32(indices)));
    
    (mesh, chunk_coord)
}

// System to process chunk mesh generation tasks
fn process_chunk_mesh_tasks(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    voxel_assets: Res<VoxelAssets>,
    mut voxel_world: ResMut<VoxelWorld>,
) {
    let mut completed_tasks: Vec<(usize, (Mesh, ChunkCoord))> = Vec::new();
    
    // Check which tasks are completed
    for (i, (_coord, task)) in voxel_world.chunk_mesh_tasks.iter_mut().enumerate() {
        if let Some(result) = futures_lite::future::block_on(futures_lite::future::poll_once(task)) {
            completed_tasks.push((i, result));
        }
    }
    
    // Process completed tasks in reverse order to avoid index invalidation
    for (index, (mesh, chunk_coord)) in completed_tasks.iter().rev() {
        // Remove the task from the list
        let _removed_task = voxel_world.chunk_mesh_tasks.remove(*index);
        
        // Add the mesh as an entity
        if let Some(chunk_arc) = voxel_world.chunks.get(&chunk_coord) {
            if let Ok(mut chunk) = chunk_arc.lock() {
                // Calculate the world position for this chunk
                let chunk_pos = Vec3::new(
                    chunk.coord.x as f32 * CHUNK_SIZE as f32 * VOXEL_SIZE,
                    chunk.coord.y as f32 * CHUNK_SIZE as f32 * VOXEL_SIZE,
                    chunk.coord.z as f32 * CHUNK_SIZE as f32 * VOXEL_SIZE,
                );
                
                // If the chunk already has an entity, update its mesh
                if let Some(entity) = chunk.entity {
                    commands.entity(entity).insert(meshes.add(mesh.clone()));
                } else {
                    // Otherwise create a new entity
                    let entity = commands
                        .spawn(PbrBundle {
                            mesh: meshes.add(mesh.clone()),
                            material: voxel_assets.material.clone(),
                            transform: Transform::from_translation(chunk_pos),
                            ..default()
                        })
                        .id();
                    
                    chunk.entity = Some(entity);
                }
                
                chunk.needs_mesh_update = false;
            }
        }
    }
}

// System to update chunks that need a mesh update
fn update_chunks(
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