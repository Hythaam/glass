use anyhow::{Context, Result, anyhow, bail};
use reqwest::Url;
use serde::Deserialize;
use std::env;
use std::ffi::OsString;
use std::fs;
use std::path::{Path, PathBuf};

const OLLAMA_URL_FLAG: &str = "--ollama-url";
const CONTEXT_LIMIT_TOKENS_FLAG: &str = "--context-limit-tokens";
const OLLAMA_URL_ENV: &str = "GLASS_OLLAMA_URL";
const CONTEXT_LIMIT_TOKENS_ENV: &str = "GLASS_CONTEXT_LIMIT_TOKENS";
const CONFIG_FILE_RELATIVE_PATH: &str = ".config/glass/config.toml";

#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct CliConfig {
    pub ollama_url: Option<String>,
    pub context_limit_tokens: Option<usize>,
}

impl CliConfig {
    pub fn from_args<I, T>(args: I) -> Result<Self>
    where
        I: IntoIterator<Item = T>,
        T: Into<OsString>,
    {
        let mut config = Self::default();
        let mut args = args.into_iter().map(Into::into);
        let _program = args.next();

        // Note: Unknown-argument behavior currently bails with an error. If we later add
        // positional arguments or broader CLI handling, revisit this behavior to allow
        // positional parsing or graceful ignoring of unknown flags.  -- TODO
        while let Some(raw_arg) = args.next() {
            let arg = os_string_to_string(raw_arg, "CLI arguments")?;
            match arg.as_str() {
                OLLAMA_URL_FLAG => {
                    config.ollama_url = Some(next_arg_value(&mut args, OLLAMA_URL_FLAG)?);
                }
                CONTEXT_LIMIT_TOKENS_FLAG => {
                    let value = next_arg_value(&mut args, CONTEXT_LIMIT_TOKENS_FLAG)?;
                    config.context_limit_tokens =
                        Some(parse_context_limit(&value, CONTEXT_LIMIT_TOKENS_FLAG)?);
                }
                _ if arg.starts_with("--ollama-url=") => {
                    config.ollama_url = Some(arg[OLLAMA_URL_FLAG.len() + 1..].to_owned());
                }
                _ if arg.starts_with("--context-limit-tokens=") => {
                    let value = &arg[CONTEXT_LIMIT_TOKENS_FLAG.len() + 1..];
                    config.context_limit_tokens =
                        Some(parse_context_limit(value, CONTEXT_LIMIT_TOKENS_FLAG)?);
                }
                _ => bail!("unrecognized argument: {arg}"),
            }
        }

        Ok(config)
    }
}

#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct EnvConfig {
    pub ollama_url: Option<String>,
    pub context_limit_tokens: Option<usize>,
}

impl EnvConfig {
    pub fn from_env() -> Result<Self> {
        Self::from_pairs(env::vars())
    }

    pub fn from_pairs<I, K, V>(pairs: I) -> Result<Self>
    where
        I: IntoIterator<Item = (K, V)>,
        K: AsRef<str>,
        V: AsRef<str>,
    {
        let mut config = Self::default();

        for (key, value) in pairs {
            match key.as_ref() {
                OLLAMA_URL_ENV => config.ollama_url = Some(value.as_ref().to_owned()),
                CONTEXT_LIMIT_TOKENS_ENV => {
                    config.context_limit_tokens = Some(parse_context_limit(
                        value.as_ref(),
                        CONTEXT_LIMIT_TOKENS_ENV,
                    )?)
                }
                _ => {}
            }
        }

        Ok(config)
    }
}

#[derive(Debug, Default, Clone, PartialEq, Eq, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct FileConfig {
    pub ollama_url: Option<String>,
    #[serde(deserialize_with = "deserialize_option_context_limit")]
    pub context_limit_tokens: Option<usize>,
}

impl FileConfig {
    pub fn from_default_location() -> Result<Self> {
        match default_config_path() {
            Some(path) => Self::from_path(path),
            None => Ok(Self::default()),
        }
    }

    pub fn from_path(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        match fs::read_to_string(path) {
            Ok(contents) => Self::from_toml_str(&contents)
                .with_context(|| format!("failed to parse config file {}", path.display())),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(Self::default()),
            Err(error) => {
                Err(error).with_context(|| format!("failed to read config file {}", path.display()))
            }
        }
    }

    pub fn from_toml_str(contents: &str) -> Result<Self> {
        toml::from_str(contents).context("config file must be valid TOML")
    }
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct Config {
    pub ollama_url: Url,
    pub context_limit_tokens: usize,
}

impl Config {
    pub fn load<I, T>(args: I) -> Result<Self>
    where
        I: IntoIterator<Item = T>,
        T: Into<OsString>,
    {
        let cli = CliConfig::from_args(args)?;
        let env = EnvConfig::from_env()?;
        let file = FileConfig::from_default_location()?;
        Self::from_sources(cli, env, file)
    }

    pub fn from_sources(cli: CliConfig, env: EnvConfig, file: FileConfig) -> Result<Self> {
        let ollama_url = cli
            .ollama_url
            .or(env.ollama_url)
            .or(file.ollama_url)
            .ok_or_else(|| anyhow!("ollama_url is required; provide via --ollama-url, GLASS_OLLAMA_URL, or config file"))?;
        let context_limit_tokens = cli
            .context_limit_tokens
            .or(env.context_limit_tokens)
            .or(file.context_limit_tokens)
            .ok_or_else(|| anyhow!("context_limit_tokens is required; provide via --context-limit-tokens, GLASS_CONTEXT_LIMIT_TOKENS, or config file"))?;

        Ok(Self {
            ollama_url: validate_ollama_url(&ollama_url)?,
            context_limit_tokens,
        })
    }
}

fn validate_ollama_url(raw: &str) -> Result<Url> {
    let url = Url::parse(raw).map_err(|e| anyhow!("ollama_url must be a full base URL: {e}"))?;

    // Require http or https scheme because Ollama speaks HTTP
    match url.scheme() {
        "http" | "https" => {}
        _ => bail!("ollama_url must use http or https scheme"),
    }

    // Require a non-empty host explicitly
    if url.host_str().map(|h| h.is_empty()).unwrap_or(true) {
        bail!("ollama_url must include a non-empty host");
    }

    // Disallow path, query, or fragment beyond a base '/'
    if url.path() != "/" {
        bail!("ollama_url must be a base URL and must not include a path");
    }
    if url.query().is_some() {
        bail!("ollama_url must not include a query string");
    }
    if url.fragment().is_some() {
        bail!("ollama_url must not include a fragment");
    }

    Ok(url)
}

fn next_arg_value<I>(args: &mut I, flag: &str) -> Result<String>
where
    I: Iterator<Item = OsString>,
{
    let value = args
        .next()
        .ok_or_else(|| anyhow!("{flag} requires a value"))?;
    os_string_to_string(value, flag)
}

fn os_string_to_string(value: OsString, label: &str) -> Result<String> {
    value
        .into_string()
        .map_err(|_| anyhow!("{label} must be valid UTF-8"))
}

fn parse_context_limit(raw: &str, source: &str) -> Result<usize> {
    let value = raw
        .parse::<usize>()
        .with_context(|| format!("{source} must be a positive integer"))?;
    if value == 0 {
        bail!("{source} must be a positive integer greater than zero");
    }
    Ok(value)
}

fn default_config_path() -> Option<PathBuf> {
    env::var_os("HOME").map(|home| PathBuf::from(home).join(CONFIG_FILE_RELATIVE_PATH))
}

fn deserialize_option_context_limit<'de, D>(
    deserializer: D,
) -> std::result::Result<Option<usize>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let opt = Option::<usize>::deserialize(deserializer)?;
    if let Some(0) = opt {
        return Err(serde::de::Error::custom(
            "context_limit_tokens must be a positive integer greater than zero",
        ));
    }
    Ok(opt)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loads_cli_over_env_over_file() {
        let file = FileConfig {
            ollama_url: Some("http://file:11434".into()),
            context_limit_tokens: Some(2048),
        };
        let env = EnvConfig {
            ollama_url: Some("http://env:11434".into()),
            context_limit_tokens: Some(4096),
        };
        let cli = CliConfig {
            ollama_url: Some("http://cli:11434".into()),
            context_limit_tokens: Some(8192),
        };

        let config = Config::from_sources(cli, env, file).unwrap();
        assert_eq!(config.ollama_url.as_str(), "http://cli:11434/");
        assert_eq!(config.context_limit_tokens, 8192);
    }

    #[test]
    fn env_over_file_when_cli_absent() {
        let file = FileConfig {
            ollama_url: Some("http://file:11434".into()),
            context_limit_tokens: Some(2048),
        };
        let env = EnvConfig {
            ollama_url: Some("http://env:11434".into()),
            context_limit_tokens: Some(4096),
        };
        let cli = CliConfig::default();

        let config = Config::from_sources(cli, env, file).unwrap();
        assert_eq!(config.ollama_url.as_str(), "http://env:11434/");
        assert_eq!(config.context_limit_tokens, 4096);
    }

    #[test]
    fn file_used_when_cli_and_env_absent() {
        let file = FileConfig {
            ollama_url: Some("http://file:11434".into()),
            context_limit_tokens: Some(2048),
        };
        let env = EnvConfig::default();
        let cli = CliConfig::default();

        let config = Config::from_sources(cli, env, file).unwrap();
        assert_eq!(config.ollama_url.as_str(), "http://file:11434/");
        assert_eq!(config.context_limit_tokens, 2048);
    }

    #[test]
    fn rejects_non_full_url() {
        let cli = CliConfig {
            ollama_url: Some("localhost:11434".into()),
            context_limit_tokens: Some(4096),
        };

        let error = Config::from_sources(cli, EnvConfig::default(), FileConfig::default())
            .unwrap_err()
            .to_string();
        assert!(error.contains("ollama_url"));
    }

    #[test]
    fn rejects_non_http_scheme() {
        let cli = CliConfig {
            ollama_url: Some("ftp://example.com".into()),
            context_limit_tokens: Some(4096),
        };

        let error = Config::from_sources(cli, EnvConfig::default(), FileConfig::default())
            .unwrap_err()
            .to_string();
        assert!(error.contains("http") || error.contains("https"));
    }

    #[test]
    fn rejects_zero_context_limit_in_env() {
        let res = EnvConfig::from_pairs([(CONTEXT_LIMIT_TOKENS_ENV, "0")]);
        assert!(res.is_err());
        let err = res.unwrap_err().to_string();
        assert!(err.contains(CONTEXT_LIMIT_TOKENS_ENV) || err.contains("positive"));
    }

    #[test]
    fn parses_cli_flags() {
        let cli = CliConfig::from_args([
            "glass",
            "--ollama-url",
            "http://cli:11434",
            "--context-limit-tokens=8192",
        ])
        .unwrap();

        assert_eq!(cli.ollama_url.as_deref(), Some("http://cli:11434"));
        assert_eq!(cli.context_limit_tokens, Some(8192));
    }

    #[test]
    fn parses_cli_equals_form_ollama_url() {
        let cli = CliConfig::from_args([
            "glass",
            "--ollama-url=http://cli:11434",
            "--context-limit-tokens=8192",
        ])
        .unwrap();

        assert_eq!(cli.ollama_url.as_deref(), Some("http://cli:11434"));
        assert_eq!(cli.context_limit_tokens, Some(8192));
    }

    #[test]
    fn parses_env_config() {
        let env = EnvConfig::from_pairs([
            (OLLAMA_URL_ENV, "http://env:11434"),
            (CONTEXT_LIMIT_TOKENS_ENV, "4096"),
        ])
        .unwrap();

        assert_eq!(env.ollama_url.as_deref(), Some("http://env:11434"));
        assert_eq!(env.context_limit_tokens, Some(4096));
    }

    #[test]
    fn parses_file_config_from_toml() {
        let file = FileConfig::from_toml_str(
            r#"
ollama_url = "http://file:11434"
context_limit_tokens = 2048
"#,
        )
        .unwrap();

        assert_eq!(file.ollama_url.as_deref(), Some("http://file:11434"));
        assert_eq!(file.context_limit_tokens, Some(2048));
    }

    #[test]
    fn rejects_zero_context_limit_in_toml() {
        let res = FileConfig::from_toml_str(
            r#"
ollama_url = "http://file:11434"
context_limit_tokens = 0
"#,
        );
        assert!(res.is_err());
    }

    #[test]
    fn rejects_zero_context_limit() {
        let res = CliConfig::from_args([
            "glass",
            "--ollama-url",
            "http://cli:11434",
            "--context-limit-tokens",
            "0",
        ]);
        assert!(res.is_err());
        let err = res.unwrap_err().to_string();
        assert!(err.contains("context_limit_tokens") || err.contains("positive"));
    }

    #[test]
    fn missing_ollama_url_is_error() {
        let error = Config::from_sources(
            CliConfig::default(),
            EnvConfig::default(),
            FileConfig::default(),
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("ollama_url"));
    }

    #[test]
    fn missing_context_limit_tokens_is_error() {
        let error = Config::from_sources(
            CliConfig {
                ollama_url: Some("http://cli:11434".into()),
                context_limit_tokens: None,
            },
            EnvConfig::default(),
            FileConfig::default(),
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("context_limit_tokens"));
    }

    #[test]
    fn missing_config_file_returns_default() {
        let file = FileConfig::from_path("does-not-exist-config.toml").unwrap();
        assert_eq!(file, FileConfig::default());
    }

    #[test]
    fn rejects_unknown_toml_fields() {
        let res = FileConfig::from_toml_str(r#"unknown = 1"#);
        assert!(res.is_err());
    }

    #[test]
    fn parses_cli_space_separated_context_limit() {
        let cli = CliConfig::from_args(["glass", "--context-limit-tokens", "8192"]).unwrap();
        assert_eq!(cli.context_limit_tokens, Some(8192));
    }
}
