#[tokio::main]
async fn main() -> anyhow::Result<()> {
    glass::app::run().await
}
