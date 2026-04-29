use anyhow::{Context, Result, anyhow, bail};
use reqwest::Url;
use serde::Deserialize;
use std::env;
use std::ffi::OsString;
use std::fs;
use std::path::{Path, PathBuf};

const OLLAMA_URL_FLAG: &str = "--ollama-url";
const CONTEXT_LIMIT_TOKENS_FLAG: &str = "--context-limit-tokens";
const SYSTEM_PROMPT_FILE_FLAG: &str = "--system-prompt-file";
const OLLAMA_URL_ENV: &str = "GLASS_OLLAMA_URL";
const CONTEXT_LIMIT_TOKENS_ENV: &str = "GLASS_CONTEXT_LIMIT_TOKENS";
const SYSTEM_PROMPT_FILE_ENV: &str = "GLASS_SYSTEM_PROMPT_FILE";
const CONFIG_FILE_RELATIVE_PATH: &str = ".config/glass/config.toml";

#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct CliConfig {
    pub ollama_url: Option<String>,
    pub context_limit_tokens: Option<usize>,
    pub system_prompt_file: Option<String>,
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
            if arg == OLLAMA_URL_FLAG {
                config.set_flag_value(OLLAMA_URL_FLAG, next_arg_value(&mut args, OLLAMA_URL_FLAG)?)?;
            } else if arg == CONTEXT_LIMIT_TOKENS_FLAG {
                config.set_flag_value(
                    CONTEXT_LIMIT_TOKENS_FLAG,
                    next_arg_value(&mut args, CONTEXT_LIMIT_TOKENS_FLAG)?,
                )?;
            } else if arg == SYSTEM_PROMPT_FILE_FLAG {
                config.set_flag_value(
                    SYSTEM_PROMPT_FILE_FLAG,
                    next_arg_value(&mut args, SYSTEM_PROMPT_FILE_FLAG)?,
                )?;
            } else if let Some((flag, value)) = parse_inline_flag_value(&arg) {
                config.set_flag_value(flag, value.to_owned())?;
            } else {
                bail!("unrecognized argument: {arg}");
            }
        }

        Ok(config)
    }

    fn set_flag_value(&mut self, flag: &str, value: String) -> Result<()> {
        match flag {
            OLLAMA_URL_FLAG => self.ollama_url = Some(value),
            CONTEXT_LIMIT_TOKENS_FLAG => {
                self.context_limit_tokens = Some(parse_context_limit(&value, flag)?)
            }
            SYSTEM_PROMPT_FILE_FLAG => self.system_prompt_file = Some(value),
            _ => bail!("unsupported argument: {flag}"),
        }
        Ok(())
    }
}

#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct EnvConfig {
    pub ollama_url: Option<String>,
    pub context_limit_tokens: Option<usize>,
    pub system_prompt_file: Option<String>,
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
                SYSTEM_PROMPT_FILE_ENV => {
                    config.system_prompt_file = Some(value.as_ref().to_owned())
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
    pub system_prompt_file: Option<String>,
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
    pub system_prompt: Option<String>,
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
            .ok_or_else(|| missing_required_setting("ollama_url", OLLAMA_URL_FLAG, OLLAMA_URL_ENV))?;
        let context_limit_tokens = cli
            .context_limit_tokens
            .or(env.context_limit_tokens)
            .or(file.context_limit_tokens)
            .ok_or_else(|| {
                missing_required_setting(
                    "context_limit_tokens",
                    CONTEXT_LIMIT_TOKENS_FLAG,
                    CONTEXT_LIMIT_TOKENS_ENV,
                )
            })?;
        let system_prompt = cli
            .system_prompt_file
            .or(env.system_prompt_file)
            .or(file.system_prompt_file)
            .map(load_system_prompt_file)
            .transpose()?;

        Ok(Self {
            ollama_url: validate_ollama_url(&ollama_url)?,
            context_limit_tokens,
            system_prompt,
        })
    }
}

fn load_system_prompt_file(path: String) -> Result<String> {
    fs::read_to_string(&path)
        .with_context(|| format!("failed to read system prompt file {path}"))
        .and_then(|contents| {
            if contents.is_empty() {
                bail!("system prompt file {path} must not be empty");
            }
            Ok(contents)
        })
}

fn validate_ollama_url(raw: &str) -> Result<Url> {
    let url = Url::parse(raw).map_err(|e| anyhow!("ollama_url must be a full base URL: {e}"))?;

    // Require http or https scheme because the configured server speaks HTTP
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

fn missing_required_setting(name: &str, flag: &str, env_var: &str) -> anyhow::Error {
    anyhow!("{name} is required; provide via {flag}, {env_var}, or config file")
}

fn parse_inline_flag_value(arg: &str) -> Option<(&str, &str)> {
    for flag in [
        OLLAMA_URL_FLAG,
        CONTEXT_LIMIT_TOKENS_FLAG,
        SYSTEM_PROMPT_FILE_FLAG,
    ] {
        if let Some(value) = parse_flag_with_value(arg, flag) {
            return Some((flag, value));
        }
    }
    None
}

fn parse_flag_with_value<'a>(arg: &'a str, flag: &'static str) -> Option<&'a str> {
    let (name, value) = arg.split_once('=')?;
    (name == flag).then_some(value)
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
    use crate::test_support::fs::TestDir;

    const CLI_URL: &str = "http://cli:11434";
    const ENV_URL: &str = "http://env:11434";
    const FILE_URL: &str = "http://file:11434";

    fn cli_with(url: &str, limit: usize) -> CliConfig {
        CliConfig {
            ollama_url: Some(url.into()),
            context_limit_tokens: Some(limit),
            system_prompt_file: None,
        }
    }

    fn env_with(url: &str, limit: usize) -> EnvConfig {
        EnvConfig {
            ollama_url: Some(url.into()),
            context_limit_tokens: Some(limit),
            system_prompt_file: None,
        }
    }

    fn file_with(url: &str, limit: usize) -> FileConfig {
        FileConfig {
            ollama_url: Some(url.into()),
            context_limit_tokens: Some(limit),
            system_prompt_file: None,
        }
    }

    fn cli_url_only(url: &str) -> CliConfig {
        CliConfig {
            ollama_url: Some(url.into()),
            context_limit_tokens: None,
            system_prompt_file: None,
        }
    }

    fn assert_config_error(cli: CliConfig, env: EnvConfig, file: FileConfig) -> String {
        Config::from_sources(cli, env, file)
            .unwrap_err()
            .to_string()
    }

    fn assert_config_error_contains(
        cli: CliConfig,
        env: EnvConfig,
        file: FileConfig,
        expected_substrings: &[&str],
    ) {
        let error = assert_config_error(cli, env, file);
        assert!(
            expected_substrings
                .iter()
                .any(|expected| error.contains(expected)),
            "expected one of {expected_substrings:?} in error: {error}"
        );
    }

    fn assert_loaded(
        cli: CliConfig,
        env: EnvConfig,
        file: FileConfig,
        expected_url: &str,
        expected_limit: usize,
    ) {
        let config = Config::from_sources(cli, env, file).unwrap();
        assert_eq!(config.ollama_url.as_str(), expected_url);
        assert_eq!(config.context_limit_tokens, expected_limit);
    }

    #[test]
    fn loads_config_by_precedence() {
        let cases = [
            (
                cli_with(CLI_URL, 8192),
                env_with(ENV_URL, 4096),
                file_with(FILE_URL, 2048),
                "http://cli:11434/",
                8192,
            ),
            (
                CliConfig::default(),
                env_with(ENV_URL, 4096),
                file_with(FILE_URL, 2048),
                "http://env:11434/",
                4096,
            ),
            (
                CliConfig::default(),
                EnvConfig::default(),
                file_with(FILE_URL, 2048),
                "http://file:11434/",
                2048,
            ),
        ];

        for (cli, env, file, expected_url, expected_limit) in cases {
            assert_loaded(cli, env, file, expected_url, expected_limit);
        }
    }

    #[test]
    fn rejects_non_full_url() {
        assert_config_error_contains(
            cli_with("localhost:11434", 4096),
            EnvConfig::default(),
            FileConfig::default(),
            &["ollama_url"],
        );
    }

    #[test]
    fn rejects_non_http_scheme() {
        assert_config_error_contains(
            cli_with("ftp://example.com", 4096),
            EnvConfig::default(),
            FileConfig::default(),
            &["http", "https"],
        );
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
            CLI_URL,
            "--context-limit-tokens=8192",
            "--system-prompt-file",
            "prompt.txt",
        ])
        .unwrap();

        assert_eq!(cli.ollama_url.as_deref(), Some(CLI_URL));
        assert_eq!(cli.context_limit_tokens, Some(8192));
        assert_eq!(cli.system_prompt_file.as_deref(), Some("prompt.txt"));
    }

    #[test]
    fn parses_cli_equals_form_ollama_url() {
        let cli = CliConfig::from_args([
            "glass",
            "--ollama-url=http://cli:11434",
            "--context-limit-tokens=8192",
        ])
        .unwrap();

        assert_eq!(cli.ollama_url.as_deref(), Some(CLI_URL));
        assert_eq!(cli.context_limit_tokens, Some(8192));
    }

    #[test]
    fn parses_env_config() {
        let env = EnvConfig::from_pairs([
            (OLLAMA_URL_ENV, ENV_URL),
            (CONTEXT_LIMIT_TOKENS_ENV, "4096"),
            ("GLASS_SYSTEM_PROMPT_FILE", "prompt.txt"),
        ])
        .unwrap();

        assert_eq!(env.ollama_url.as_deref(), Some(ENV_URL));
        assert_eq!(env.context_limit_tokens, Some(4096));
        assert_eq!(env.system_prompt_file.as_deref(), Some("prompt.txt"));
    }

    #[test]
    fn parses_file_config_from_toml() {
        let file = FileConfig::from_toml_str(
            r#"
ollama_url = "http://file:11434"
context_limit_tokens = 2048
system_prompt_file = "prompt.txt"
"#,
        )
        .unwrap();

        assert_eq!(file.ollama_url.as_deref(), Some(FILE_URL));
        assert_eq!(file.context_limit_tokens, Some(2048));
        assert_eq!(file.system_prompt_file.as_deref(), Some("prompt.txt"));
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
        assert_config_error_contains(
            CliConfig::default(),
            EnvConfig::default(),
            FileConfig::default(),
            &["ollama_url"],
        );
    }

    #[test]
    fn missing_context_limit_tokens_is_error() {
        assert_config_error_contains(
            cli_url_only(CLI_URL),
            EnvConfig::default(),
            FileConfig::default(),
            &["context_limit_tokens"],
        );
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

    #[test]
    fn loads_system_prompt_file_with_cli_precedence() {
        let temp = TestDir::new("config-prompts");
        let cli_path = temp.write_text("cli-prompt.txt", "cli prompt");
        let env_path = temp.write_text("env-prompt.txt", "env prompt");
        let file_path = temp.write_text("file-prompt.txt", "file prompt");
        let file = FileConfig {
            ollama_url: file_with(FILE_URL, 2048).ollama_url,
            context_limit_tokens: file_with(FILE_URL, 2048).context_limit_tokens,
            system_prompt_file: Some(file_path.display().to_string()),
        };
        let env = EnvConfig {
            ollama_url: env_with(ENV_URL, 4096).ollama_url,
            context_limit_tokens: env_with(ENV_URL, 4096).context_limit_tokens,
            system_prompt_file: Some(env_path.display().to_string()),
        };
        let cli = CliConfig {
            ollama_url: cli_with(CLI_URL, 8192).ollama_url,
            context_limit_tokens: cli_with(CLI_URL, 8192).context_limit_tokens,
            system_prompt_file: Some(cli_path.display().to_string()),
        };

        let config = Config::from_sources(cli, env, file).unwrap();

        assert_eq!(config.system_prompt.as_deref(), Some("cli prompt"));
    }

    #[test]
    fn missing_system_prompt_file_is_error() {
        assert_config_error_contains(
            CliConfig {
                system_prompt_file: Some("/definitely/missing/prompt.txt".into()),
                ..cli_with(CLI_URL, 8192)
            },
            EnvConfig::default(),
            FileConfig::default(),
            &["system prompt", "prompt.txt"],
        );
    }
}
