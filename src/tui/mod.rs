mod app;
mod render;
mod state;
mod transcript;

pub use app::TuiApp;

#[cfg(test)]
pub(crate) use render::{
    composer_block, composer_cursor_position, composer_paragraph, layout_chunks,
    transcript_inner_size,
};
#[cfg(test)]
pub(crate) use state::{TuiAction, TuiState};
#[cfg(test)]
pub(crate) use transcript::TranscriptState;

#[cfg(test)]
mod tests;
