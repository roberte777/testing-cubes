use bevy::{
    prelude::*,
    input::{keyboard::KeyCode, mouse::{MouseButton, MouseMotion}},
    window::{CursorGrabMode, PrimaryWindow},
};

use crate::components::player::Player;
use crate::components::chunk::ChunkCoord;
use crate::constants::{CHUNK_SIZE, VOXEL_SIZE, MOUSE_SENSITIVITY};

/// Updates cursor grab mode based on mouse input
pub fn cursor_grab(
    mut window_query: Query<&mut Window, With<PrimaryWindow>>,
    key_input: Res<ButtonInput<KeyCode>>,
    mouse_input: Res<ButtonInput<MouseButton>>,
) {
    let mut window = window_query.single_mut();
    
    if key_input.just_pressed(KeyCode::Escape) {
        // Unlock cursor when Escape is pressed
        window.cursor.grab_mode = CursorGrabMode::None;
        window.cursor.visible = true;
    } else if mouse_input.just_pressed(MouseButton::Left) {
        // Lock cursor when left mouse button is clicked
        window.cursor.grab_mode = CursorGrabMode::Locked;
        window.cursor.visible = false;
    }
}

/// Handles camera rotation based on mouse movement
pub fn camera_control(
    window_query: Query<&Window, With<PrimaryWindow>>,
    mut player_query: Query<(&mut Player, &mut Transform)>,
    mut motion_evr: EventReader<MouseMotion>,
) {
    let window = window_query.single();
    
    // Only update camera if cursor is grabbed
    if window.cursor.grab_mode != CursorGrabMode::Locked {
        return;
    }
    
    let (mut player, mut transform) = player_query.single_mut();
    
    // Sum up all mouse motion events
    let mut delta = Vec2::ZERO;
    for ev in motion_evr.read() {
        delta += ev.delta;
    }
    
    if delta.length_squared() > 0.0 {
        // Update player rotation - flip directions for more intuitive control
        player.yaw -= delta.x * MOUSE_SENSITIVITY * 0.01;  // Increased sensitivity and negative movement
        player.pitch -= delta.y * MOUSE_SENSITIVITY * 0.01; // Negative movement for looking up/down
        
        // Clamp pitch to prevent camera flipping
        player.pitch = player.pitch.clamp(-std::f32::consts::FRAC_PI_2, std::f32::consts::FRAC_PI_2);
        
        // Apply rotation to transform
        transform.rotation = Quat::from_euler(EulerRot::YXZ, player.yaw, player.pitch, 0.0);
    }
}

/// Handles player movement based on keyboard input
pub fn player_movement(
    key_input: Res<ButtonInput<KeyCode>>,
    mut player_query: Query<(&mut Player, &mut Transform)>,
    time: Res<Time>,
) {
    let (mut player, mut transform) = player_query.single_mut();
    
    let mut direction = Vec3::ZERO;
    
    // Get player's forward/right directions from the current rotation
    let (forward, right) = {
        let rotation = transform.rotation;
        // In Bevy, -Z is forward by convention
        let forward = rotation * -Vec3::Z;
        let right = rotation * Vec3::X;
        (forward, right)
    };
    
    // Check WASD keys for movement
    if key_input.pressed(KeyCode::KeyW) {
        direction += forward;
    }
    if key_input.pressed(KeyCode::KeyS) {
        direction -= forward;
    }
    
    if key_input.pressed(KeyCode::KeyA) {
        direction -= right;
    }
    if key_input.pressed(KeyCode::KeyD) {
        direction += right;
    }
    
    // Up/down movement
    if key_input.pressed(KeyCode::Space) {
        direction += Vec3::Y;
    }
    if key_input.pressed(KeyCode::ShiftLeft) {
        direction -= Vec3::Y;
    }
    
    // Normalize direction if moving to avoid faster diagonal movement
    if direction.length_squared() > 0.0 {
        direction = direction.normalize();
    }
    
    // Adjust movement speed
    let mut speed = 8.0; // Normal speed - increased from 5.0
    
    if key_input.pressed(KeyCode::ControlLeft) {
        speed = 30.0; // Sprint speed - increased from 20.0
    }
    
    // Apply movement
    transform.translation += direction * speed * time.delta_seconds();
    
    // Update player's chunk coordinate
    let current_chunk_x = (transform.translation.x / (CHUNK_SIZE as f32 * VOXEL_SIZE)).floor() as i32;
    let current_chunk_y = (transform.translation.y / (CHUNK_SIZE as f32 * VOXEL_SIZE)).floor() as i32;
    let current_chunk_z = (transform.translation.z / (CHUNK_SIZE as f32 * VOXEL_SIZE)).floor() as i32;
    
    player.chunk_coord = ChunkCoord::new(current_chunk_x, current_chunk_y, current_chunk_z);
} 