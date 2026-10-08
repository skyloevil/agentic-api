//! Handler-to-HTTPS acceptance tests. Requires Python 3 and OpenSSL on Unix test hosts.
#![cfg(unix)]
#[allow(dead_code)]
mod common;
#[path = "messages_mcp_connector/fixture.rs"]
mod fixture;

use std::sync::{Arc, Mutex};

use agentic_core::tool::mcp::client::test_support::with_root_certificate;
use agentic_server::app::{ServerConfig, build_router};
use axum::Router;
use axum::body::Body;
use http::StatusCode;
use serde_json::{Value, json};
use tower::ServiceExt;
use tracing::instrument::WithSubscriber;

fn request(url: &str, stream: bool, mode: &str) -> Value {
    let deferred = mode == "deferred";
    json!({"model":mode,"max_tokens":128,"stream":stream,"extension":{"preserved":true},
        "messages":[{"role":"user","content":"use echo"}],
        "mcp_servers":[{"type":"url","name":"counter","url":url,"authorization_token":"connector-secret"}],
        "tools":[{"type":"tool_search_tool_regex_20251119","name":"tool_search_tool_regex"},
            {"type":"mcp_toolset","mcp_server_name":"counter","default_config":{"enabled":false,"defer_loading":true},
            "configs":{"echo":{"enabled":true,"defer_loading":deferred},"fail":{"enabled":true,"defer_loading":false}}}]})
}

async fn post(router: &Router, path: &str, body: &Value) -> (StatusCode, String) {
    let response = router
        .clone()
        .oneshot(
            http::Request::post(path)
                .header(http::header::CONTENT_TYPE, "application/json")
                .header("x-api-key", "test-key")
                .header("anthropic-beta", "mcp-client-2025-11-20")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    let status = response.status();
    let bytes = axum::body::to_bytes(response.into_body(), 1024 * 1024).await.unwrap();
    (status, String::from_utf8(bytes.to_vec()).unwrap())
}

#[derive(Clone)]
struct LogWriter(Arc<Mutex<Vec<u8>>>);
impl std::io::Write for LogWriter {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        self.0.lock().unwrap().extend_from_slice(bytes);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

async fn trusted_post(router: &Router, path: &str, body: &Value, cert: &[u8]) -> (StatusCode, String) {
    let logs = Arc::new(Mutex::new(Vec::new()));
    let writer = Arc::clone(&logs);
    let subscriber = tracing_subscriber::fmt()
        .with_max_level(tracing::Level::DEBUG)
        .without_time()
        .with_ansi(false)
        .with_writer(move || LogWriter(Arc::clone(&writer)))
        .finish();
    let response = with_root_certificate(cert.to_vec(), post(router, path, body).with_subscriber(subscriber))
        .await
        .unwrap();
    let log = String::from_utf8(logs.lock().unwrap().clone()).unwrap();
    assert!(!log.contains("connector-secret"));
    assert!(!log.contains("wrong-secret"));
    assert!(!response.1.contains("connector-secret"));
    assert!(!response.1.contains("wrong-secret"));
    response
}

fn blocks(response: &str, stream: bool) -> Vec<Value> {
    if !stream {
        let response: Value = serde_json::from_str(response).unwrap();
        assert_eq!(response["usage"]["output_tokens"], 6);
        return response["content"].as_array().unwrap().clone();
    }
    let events = response
        .lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .map(|line| serde_json::from_str::<Value>(line).unwrap())
        .collect::<Vec<_>>();
    assert_eq!(
        events.iter().filter(|event| event["type"] == "message_start").count(),
        1
    );
    assert_eq!(events.iter().filter(|event| event["type"] == "message_stop").count(), 1);
    let starts = events
        .iter()
        .filter(|event| event["type"] == "content_block_start")
        .collect::<Vec<_>>();
    let stops = events
        .iter()
        .filter(|event| event["type"] == "content_block_stop")
        .collect::<Vec<_>>();
    assert_eq!(starts.len(), stops.len());
    for (index, (start, stop)) in starts.iter().zip(stops).enumerate() {
        assert_eq!(start["index"], index);
        assert_eq!(stop["index"], index);
    }
    let terminal = events.iter().find(|event| event["type"] == "message_delta").unwrap();
    assert_eq!(terminal["usage"]["output_tokens"], 6);
    starts
        .into_iter()
        .map(|start| {
            let mut block = start["content_block"].clone();
            let input = events
                .iter()
                .filter(|event| event["type"] == "content_block_delta" && event["index"] == start["index"])
                .filter_map(|event| event["delta"]["partial_json"].as_str())
                .collect::<String>();
            if !input.is_empty() {
                block["input"] = serde_json::from_str(&input).unwrap();
            }
            block
        })
        .collect()
}

fn assert_normalized(body: &Value) {
    assert!(body.get("mcp_servers").is_none());
    assert!(!body.to_string().contains("connector-secret"));
    assert_eq!(body["extension"]["preserved"], true);
    let tools = body["tools"].as_array().unwrap();
    assert!(tools.iter().all(|tool| tool["type"] != "mcp_toolset"));
    assert!(tools.iter().any(|tool| tool["name"] == "mcp__counter__echo"));
    assert!(tools.iter().any(|tool| tool["name"] == "mcp__counter__fail"));
    assert!(!tools.iter().any(|tool| tool["name"] == "mcp__counter__disabled"));
}

#[tokio::test]
async fn https_bearer_json_sse_search_replay_and_count_tokens() {
    let mcp = fixture::HttpsMcp::start().await;
    let (url, requests, task) = fixture::inference().await;
    let router = build_router(
        common::test_state(&common::test_config(&url)),
        &ServerConfig::from_env(),
    );
    for stream in [false, true] {
        for mode in ["echo", "fail", "deferred"] {
            let body = request(&mcp.url, stream, mode);
            let before = requests.lock().await.len();
            let (status, response) = trusted_post(&router, "/v1/messages", &body, &mcp.certificate).await;
            assert_eq!(status, StatusCode::OK, "{response}");
            let content = blocks(&response, stream);
            let call = content.iter().find(|block| block["type"] == "mcp_tool_use").unwrap();
            assert_eq!(call["id"], "call");
            assert_eq!(call["server_name"], "counter");
            assert_eq!(call["input"]["text"], "hello");
            assert_eq!(call["name"], if mode == "fail" { "fail" } else { "echo" });
            let result = content.iter().find(|block| block["type"] == "mcp_tool_result").unwrap();
            assert_eq!(result["tool_use_id"], "call");
            assert_eq!(result["is_error"], mode == "fail");
            let output = result["content"][0]["text"].as_str().unwrap();
            if mode == "fail" {
                assert!(output.contains("fixture error"));
            } else {
                assert_eq!(output, "fixture output: hello");
            }
            let captured = requests.lock().await;
            assert_eq!(captured.len(), before + 2);
            assert_normalized(&captured[before]);
            assert_eq!(captured[before + 1]["messages"][2]["content"][0]["type"], "tool_result");
            if mode == "deferred" {
                assert_eq!(
                    captured[before + 1]["messages"][1]["content"][0]["input"]["pattern"],
                    "echo"
                );
                assert_eq!(
                    captured[before + 1]["messages"][1]["content"][1]["content"]["tool_references"][0]["tool_name"],
                    "mcp__counter__echo"
                );
            }
            drop(captured);
            verify_replay(&router, &mcp, &requests, &body, content, mode).await;
            verify_count(&router, &mcp, &requests, body).await;
        }
    }
    let observations = mcp.observations().await;
    assert!(observations.iter().all(|entry| entry["authorized"] == true));
    let calls = observations
        .iter()
        .filter(|entry| entry["method"] == "tools/call")
        .collect::<Vec<_>>();
    assert_eq!(calls.len(), 6);
    assert!(
        calls
            .iter()
            .all(|entry| entry["params"]["arguments"]["text"] == "hello")
    );
    task.abort();
    let _ = task.await;
    mcp.stop().await;
}

async fn verify_replay(
    router: &Router,
    mcp: &fixture::HttpsMcp,
    requests: &fixture::Requests,
    body: &Value,
    content: Vec<Value>,
    mode: &str,
) {
    let tool = if mode == "fail" { "fail" } else { "echo" };
    let internal_name = format!("mcp__counter__{tool}");
    let mut replay = body.clone();
    replay["tools"][1]["configs"].as_object_mut().unwrap().remove(tool);
    replay["messages"]
        .as_array_mut()
        .unwrap()
        .push(json!({"role":"assistant","content":content}));
    let before = requests.lock().await.len();
    let (status, response) = trusted_post(router, "/v1/messages", &replay, &mcp.certificate).await;
    assert_eq!(status, StatusCode::OK, "{response}");
    let captured = requests.lock().await;
    assert_eq!(captured.len(), before + 1);
    let last = captured.last().unwrap();
    let history = last["messages"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|message| message["content"].as_array())
        .flatten()
        .collect::<Vec<_>>();
    let call = history.iter().find(|block| block["type"] == "tool_use").unwrap();
    assert_eq!(call["name"], internal_name);
    assert_eq!(call["input"]["text"], "hello");
    assert!(
        history
            .iter()
            .any(|block| block["type"] == "tool_result" && block["tool_use_id"] == "call")
    );
    assert!(
        !last["tools"]
            .as_array()
            .unwrap()
            .iter()
            .any(|tool| tool["name"] == internal_name)
    );
    if mode == "deferred" {
        let search = history.iter().find(|block| block["type"] == "server_tool_use").unwrap();
        assert_eq!(search["input"]["pattern"], "echo");
        let result = history
            .iter()
            .find(|block| block["type"] == "tool_search_tool_result")
            .unwrap();
        assert_eq!(result["content"]["tool_references"], json!([]));
    }
}

async fn verify_count(router: &Router, mcp: &fixture::HttpsMcp, requests: &fixture::Requests, mut body: Value) {
    let calls_before = mcp
        .observations()
        .await
        .into_iter()
        .filter(|entry| entry["method"] == "tools/call")
        .count();
    body.as_object_mut().unwrap().remove("max_tokens");
    let (status, response) = trusted_post(router, "/v1/messages/count_tokens", &body, &mcp.certificate).await;
    assert_eq!(status, StatusCode::OK, "{response}");
    assert_eq!(serde_json::from_str::<Value>(&response).unwrap()["input_tokens"], 42);
    assert_normalized(requests.lock().await.last().unwrap());
    assert_eq!(
        mcp.observations()
            .await
            .into_iter()
            .filter(|entry| entry["method"] == "tools/call")
            .count(),
        calls_before
    );
}

#[tokio::test]
async fn tls_auth_and_deferred_configuration_fail_before_inference() {
    let mcp = fixture::HttpsMcp::start().await;
    let (url, requests, task) = fixture::inference().await;
    let router = build_router(
        common::test_state(&common::test_config(&url)),
        &ServerConfig::from_env(),
    );
    let body = request(&mcp.url, false, "deferred");
    let (status, response) = post(&router, "/v1/messages", &body).await;
    assert!(!status.is_success(), "an untrusted fixture certificate must fail");
    assert!(!response.contains("connector-secret"));
    let mut wrong = body.clone();
    wrong["mcp_servers"][0]["authorization_token"] = json!("wrong-secret");
    let (status, _) = trusted_post(&router, "/v1/messages", &wrong, &mcp.certificate).await;
    assert!(!status.is_success());
    assert!(
        mcp.observations()
            .await
            .iter()
            .any(|entry| entry["authorized"] == false)
    );
    for path in ["/v1/messages", "/v1/messages/count_tokens"] {
        let mut missing = body.clone();
        missing["tools"].as_array_mut().unwrap().remove(0);
        let (status, response) = trusted_post(&router, path, &missing, &mcp.certificate).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{response}");
        assert!(response.contains("upstream-hosted tool search"));
        let mut deferred_search = body.clone();
        deferred_search["tools"][0]["defer_loading"] = json!(true);
        let (status, response) = trusted_post(&router, path, &deferred_search, &mcp.certificate).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{response}");
        assert!(response.contains("cannot be deferred"));
    }
    assert!(requests.lock().await.is_empty());
    task.abort();
    let _ = task.await;
    mcp.stop().await;
}

#[tokio::test]
async fn truncated_calls_are_public_but_never_executed() {
    let mcp = fixture::HttpsMcp::start().await;
    let (url, requests, task) = fixture::inference().await;
    let router = build_router(
        common::test_state(&common::test_config(&url)),
        &ServerConfig::from_env(),
    );
    for stream in [false, true] {
        for mode in ["truncated", "truncated_mixed"] {
            let body = request(&mcp.url, stream, mode);
            let (status, response) = trusted_post(&router, "/v1/messages", &body, &mcp.certificate).await;
            assert_eq!(status, StatusCode::OK, "{response}");
            let content = if stream {
                let events = response
                    .lines()
                    .filter_map(|line| line.strip_prefix("data: "))
                    .map(|data| serde_json::from_str::<Value>(data).unwrap())
                    .collect::<Vec<_>>();
                let terminal = events.iter().find(|event| event["type"] == "message_delta").unwrap();
                assert_eq!(terminal["delta"]["stop_reason"], "max_tokens");
                assert_eq!(terminal["usage"]["output_tokens"], 3);
                assert_eq!(events.iter().filter(|event| event["type"] == "message_stop").count(), 1);
                events
                    .into_iter()
                    .filter(|event| event["type"] == "content_block_start")
                    .map(|event| event["content_block"].clone())
                    .collect::<Vec<_>>()
            } else {
                let message: Value = serde_json::from_str(&response).unwrap();
                assert_eq!(message["stop_reason"], "max_tokens");
                assert_eq!(message["usage"]["output_tokens"], 3);
                message["content"].as_array().unwrap().clone()
            };
            let call = content.iter().find(|block| block["type"] == "mcp_tool_use").unwrap();
            assert_eq!(call["server_name"], "counter");
            assert_eq!(call["name"], "echo");
            assert!(!content.iter().any(|block| block["type"] == "mcp_tool_result"));
            assert_eq!(
                content.iter().any(|block| block["name"] == "client_echo"),
                mode == "truncated_mixed"
            );
        }
    }
    assert_eq!(requests.lock().await.len(), 4);
    assert!(
        !mcp.observations()
            .await
            .iter()
            .any(|entry| entry["method"] == "tools/call")
    );
    task.abort();
    let _ = task.await;
    mcp.stop().await;
}
