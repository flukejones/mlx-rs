use std::collections::HashMap;

use mlx_lm::chat_template::ChatMessage;
use mlx_lm::{Image, Prompt, UserInput};

/// `UserInput` for a chat-mode turn with images and no kwargs.
pub fn build_chat_input(messages: Vec<ChatMessage>, images: Vec<Image>) -> UserInput {
    UserInput {
        prompt: Prompt::Chat(messages),
        images,
        audios: Vec::new(),
        videos: Vec::new(),
        template_kwargs: HashMap::new(),
    }
}
