/// World generation and rendering settings
pub const CHUNK_SIZE: i32 = 16;        // Size of a chunk in blocks
pub const RENDER_DISTANCE: i32 = 6;    // How many chunks to render in each direction
pub const VOXEL_SIZE: f32 = 1.0;       // Size of each voxel in world units

/// Player settings
pub const MOUSE_SENSITIVITY: f32 = 0.3; // Mouse sensitivity for camera rotation

/// Directions for checking adjacent blocks (right, left, up, down, forward, back)
pub const DIRECTIONS: [(i32, i32, i32); 6] = [
    (1, 0, 0),  // right
    (-1, 0, 0), // left
    (0, 1, 0),  // up
    (0, -1, 0), // down
    (0, 0, 1),  // forward
    (0, 0, -1), // back
];

/// Corresponding normals for each direction
pub const DIR_NORMALS: [[f32; 3]; 6] = [
    [1.0, 0.0, 0.0],  // right
    [-1.0, 0.0, 0.0], // left
    [0.0, 1.0, 0.0],  // up
    [0.0, -1.0, 0.0], // down
    [0.0, 0.0, 1.0],  // forward
    [0.0, 0.0, -1.0], // back
];

/// Vertices offsets for each face of a cube
pub const FACE_VERTICES: [[[f32; 3]; 4]; 6] = [
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

/// UV coordinates for each vertex of a face
pub const FACE_UVS: [[f32; 2]; 4] = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]; 