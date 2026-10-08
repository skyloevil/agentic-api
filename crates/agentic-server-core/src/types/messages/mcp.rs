//! Typed Messages MCP connector declarations and public content projections.

use std::collections::{HashMap, HashSet};
use std::fmt;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::{GatewayToolResult, ToolParam};
use crate::tool::ToolError;
use crate::types::tools::{McpToolParam, ResponsesTool};

#[derive(Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct MessagesMcpServer {
    #[serde(rename = "type")]
    pub kind: MessagesMcpServerType,
    pub url: String,
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub authorization_token: Option<String>,
}

impl fmt::Debug for MessagesMcpServer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MessagesMcpServer")
            .field("name", &self.name)
            .field(
                "authorization_token",
                &self.authorization_token.as_ref().map(|_| "[REDACTED]"),
            )
            .finish_non_exhaustive()
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[serde(rename_all = "snake_case")]
pub enum MessagesMcpServerType {
    Url,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct McpToolConfig {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub enabled: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub defer_loading: Option<bool>,
}

#[derive(Debug, Clone, Default)]
pub(crate) struct McpToolsetConfig {
    pub default_config: McpToolConfig,
    pub configs: HashMap<String, McpToolConfig>,
}

impl McpToolsetConfig {
    pub(crate) fn effective(&self, name: &str) -> (bool, bool) {
        let config = self.configs.get(name);
        (
            config
                .and_then(|c| c.enabled)
                .or(self.default_config.enabled)
                .unwrap_or(true),
            config
                .and_then(|c| c.defer_loading)
                .or(self.default_config.defer_loading)
                .unwrap_or(false),
        )
    }
}

/// Resolve toolsets to the same typed declarations used by Responses discovery.
///
/// # Errors
/// Rejects ambiguous server identities and missing or duplicate toolsets before I/O.
pub fn connector_tools(servers: &[MessagesMcpServer], tools: &[ToolParam]) -> Result<Vec<ResponsesTool>, ToolError> {
    let mut names = HashSet::new();
    for server in servers {
        if server.name.is_empty() || !names.insert(server.name.as_str()) {
            return Err(ToolError::Config(
                "MCP server names must be non-empty and unique".to_owned(),
            ));
        }
        if !server.url.starts_with("https://") {
            return Err(ToolError::Config("Messages MCP server URLs must use HTTPS".to_owned()));
        }
    }
    let mut used = HashSet::new();
    let mut resolved = Vec::new();
    for tool in tools.iter().filter(|t| t.type_.as_deref() == Some("mcp_toolset")) {
        let server = tool
            .mcp_server_name
            .as_deref()
            .and_then(|name| servers.iter().find(|server| server.name == name))
            .ok_or_else(|| ToolError::Config("mcp_toolset must reference a declared MCP server".to_owned()))?;
        if !used.insert(server.name.as_str()) {
            return Err(ToolError::Config(
                "each MCP server must have exactly one mcp_toolset".to_owned(),
            ));
        }
        resolved.push(ResponsesTool::Mcp(McpToolParam {
            server_label: server.name.clone(),
            server_url: Some(server.url.clone()),
            authorization: server.authorization_token.clone(),
            connector_id: None,
            headers: None,
            allowed_tools: None,
            require_approval: Some("never".to_owned()),
            defer_loading: None,
            discovered_tools: Vec::new(),
            messages_config: Some(McpToolsetConfig {
                default_config: tool.default_config.clone().unwrap_or_default(),
                configs: tool.configs.clone().unwrap_or_default(),
            }),
        }));
    }
    if used.len() != servers.len() {
        return Err(ToolError::Config(
            "each MCP server must have exactly one mcp_toolset".to_owned(),
        ));
    }
    Ok(resolved)
}

/// Public Messages projections; execution and call IDs stay in the shared tool path.
#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum McpContentBlock {
    McpToolUse {
        id: String,
        name: String,
        server_name: String,
        input: Value,
    },
    McpToolResult {
        tool_use_id: String,
        is_error: bool,
        content: Vec<McpTextBlock>,
    },
}

#[derive(Debug, Serialize)]
pub struct McpTextBlock {
    #[serde(rename = "type")]
    kind: McpTextKind,
    text: String,
}
#[derive(Debug, Serialize)]
#[serde(rename_all = "snake_case")]
enum McpTextKind {
    Text,
}

impl McpContentBlock {
    #[must_use]
    pub fn result(result: &GatewayToolResult) -> Self {
        Self::McpToolResult {
            tool_use_id: result.tool_use_id.clone(),
            is_error: result.is_error,
            content: vec![McpTextBlock {
                kind: McpTextKind::Text,
                text: result.content.clone(),
            }],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn tool_settings_merge_each_field_independently() {
        let tool: ToolParam = serde_json::from_value(json!({"type":"mcp_toolset", "mcp_server_name":"s",
            "default_config":{"enabled":false, "defer_loading":true},
            "configs":{"enabled":{"enabled":true}, "eager":{"enabled":true,"defer_loading":false}, "disabled":{"defer_loading":false}}
        })).unwrap();
        let config = McpToolsetConfig {
            default_config: tool.default_config.unwrap(),
            configs: tool.configs.unwrap(),
        };
        assert_eq!(config.effective("unknown"), (false, true));
        assert_eq!(config.effective("enabled"), (true, true));
        assert_eq!(config.effective("eager"), (true, false));
        assert_eq!(config.effective("disabled"), (false, false));
        assert_eq!(McpToolsetConfig::default().effective("any"), (true, false));
    }

    #[test]
    fn connector_rejects_ambiguous_and_unused_servers() {
        let server: MessagesMcpServer =
            serde_json::from_value(json!({"type":"url", "name":"s", "url":"https://example.com/mcp"})).unwrap();
        let tool: ToolParam = serde_json::from_value(json!({"type":"mcp_toolset", "mcp_server_name":"s"})).unwrap();
        assert!(connector_tools(std::slice::from_ref(&server), std::slice::from_ref(&tool)).is_ok());
        assert!(connector_tools(&[server.clone(), server.clone()], std::slice::from_ref(&tool)).is_err());
        assert!(connector_tools(std::slice::from_ref(&server), &[tool.clone(), tool.clone()]).is_err());
        assert!(connector_tools(std::slice::from_ref(&server), &[]).is_err());
        assert!(connector_tools(&[], std::slice::from_ref(&tool)).is_err());
        let mut invalid = server;
        invalid.url = "http://example.com/mcp".to_owned();
        assert!(connector_tools(&[invalid], &[tool]).is_err());
    }
}
