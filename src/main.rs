mod api;
mod tui;

use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    let url = std::env::var("OPENAPI_URL").unwrap_or_else(|_| {
        eprintln!("OPENAPI_URL env var not set");
        std::process::exit(1);
    });

    let client = api::ApiClient::new(url);
    tui::run(&client)?;
    Ok(())
}
