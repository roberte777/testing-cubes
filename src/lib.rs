pub mod components {
    pub mod voxel;
    pub mod chunk;
    pub mod player;
}

pub mod constants;

pub mod resources {
    pub mod assets;
    pub mod world;
}

pub mod systems {
    pub mod player;
    pub mod world;
    pub mod rendering;
    pub mod setup;
}

pub mod utils {
    pub mod terrain;
    pub mod mesh;
} 