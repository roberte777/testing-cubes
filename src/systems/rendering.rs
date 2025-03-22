use bevy::prelude::*;

use crate::components::player::{Player, Skybox};

/// System to update the skybox position to follow the player
pub fn update_skybox(
    player_query: Query<&Transform, With<Player>>,
    mut skybox_query: Query<&mut Transform, (With<Skybox>, Without<Player>)>,
) {
    if let Ok(player_transform) = player_query.get_single() {
        if let Ok(mut skybox_transform) = skybox_query.get_single_mut() {
            // Update skybox position to match player position
            skybox_transform.translation = player_transform.translation;
        }
    }
} 