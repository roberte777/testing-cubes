use bevy::prelude::*;
use bevy::math::primitives::Cuboid;

use crate::components::player::{Player, Skybox};
use crate::components::chunk::ChunkCoord;
use crate::resources::assets::VoxelAssets;
use crate::components::voxel::VoxelType;

/// Setup function to initialize the world, player, and resources
pub fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut voxel_assets: ResMut<VoxelAssets>,
) {
    // Create materials for each voxel type
    voxel_assets.materials.insert(VoxelType::Dirt, materials.add(StandardMaterial {
        base_color: Color::rgb(0.55, 0.27, 0.07), // Brown
        ..default()
    }));
    
    voxel_assets.materials.insert(VoxelType::Grass, materials.add(StandardMaterial {
        base_color: Color::rgb(0.0, 0.8, 0.0), // Green
        ..default()
    }));
    
    voxel_assets.materials.insert(VoxelType::Stone, materials.add(StandardMaterial {
        base_color: Color::rgb(0.5, 0.5, 0.5), // Gray
        ..default()
    }));

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

    // Add a skybox
    let skybox_size = 1000.0; // Large enough to contain the visible world
    let skybox_material = materials.add(StandardMaterial {
        base_color: Color::rgb(0.5, 0.7, 1.0), // Sky blue color
        emissive: Color::rgb(0.1, 0.1, 0.2),   // Make it slightly self-illuminating
        unlit: true,                           // Unaffected by lighting
        cull_mode: None,                       // Render the inside faces
        ..default()
    });
    
    // Spawn the skybox at the player's spawn position
    let player_spawn_position = Vec3::new(0.0, 50.0, 0.0);
    
    commands.spawn((
        PbrBundle {
            mesh: meshes.add(Cuboid::new(skybox_size, skybox_size, skybox_size)),
            material: skybox_material,
            transform: Transform::from_translation(player_spawn_position),
            // The skybox should be rendered behind everything else
            visibility: Visibility::Visible,
            ..default()
        },
        Skybox,
    ));

    // Add a debug cube at origin to help debug rendering
    commands.spawn(PbrBundle {
        mesh: meshes.add(Cuboid::new(1.0, 1.0, 1.0)),
        material: materials.add(StandardMaterial {
            base_color: Color::RED,
            ..default()
        }),
        transform: Transform::from_xyz(0.0, 0.0, 0.0),
        ..default()
    });

    // Add a debug cube at player spawn position
    commands.spawn(PbrBundle {
        mesh: meshes.add(Cuboid::new(1.0, 1.0, 1.0)),
        material: materials.add(StandardMaterial {
            base_color: Color::GREEN,
            ..default()
        }),
        transform: Transform::from_xyz(0.0, 50.0, 0.0),
        ..default()
    });

    // Add a player with camera as child
    commands.spawn((
        Player {
            chunk_coord: ChunkCoord::new(0, 0, 0),
            yaw: 0.0,
            pitch: 0.0,
        },
        TransformBundle::from_transform(Transform::from_xyz(0.0, 50.0, 0.0)),
    ))
    .with_children(|parent| {
        // Add camera as child of player
        parent.spawn((
            Camera3dBundle {
                transform: Transform::from_xyz(0.0, 2.0, 0.0),
                camera: Camera {
                    // Enable HDR for better lighting effects
                    hdr: true,
                    ..default()
                },
                ..default()
            },
            // Add fog as a separate component
            FogSettings {
                color: Color::rgba(0.5, 0.7, 1.0, 1.0), // Match skybox color
                falloff: FogFalloff::Linear {
                    start: 50.0, // Start fog at this distance
                    end: 300.0,  // Full fog at this distance
                },
                ..default()
            },
        ));
    });
} 