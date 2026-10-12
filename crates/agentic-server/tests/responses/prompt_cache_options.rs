use super::*;
use serde_json::{Value, json};

fn enable_options(fixture: &mut StorageBackedState) {
    Arc::get_mut(&mut fixture.state.exec_ctx)
        .unwrap()
        .responses_config
        .prompt_cache_options_supported = true;
}

#[tokio::test]
async fn json_parity_and_durable_continuation_preserve_request_scoped_options() {
    let (url, captured, model) = spawn_mock_vllm_json_capture().await;
    let mut fixture = storage_backed_state(&url).await;
    enable_options(&mut fixture);
    let (gateway_url, gateway) = spawn_gateway(fixture.state.clone()).await;
    let client = reqwest::Client::new();
    let mut parent = Value::Null;
    for options in [
        json!({}),
        json!({"mode":"implicit"}),
        json!({"ttl":"30m"}),
        json!({"mode":"explicit","ttl":"30m"}),
    ] {
        for store in [false, true] {
            let response = client
                .post(format!("{gateway_url}/v1/responses"))
                .json(&json!({"model":"test","input":"parent context","store":store,
                    "prompt_cache_options":options}))
                .send()
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::OK);
            let body: Value = response.json().await.unwrap();
            if store {
                parent = body["id"].clone();
            }
        }
    }
    for options in [None, Some(Value::Null), Some(json!({"mode":"implicit"}))] {
        let mut request = json!({"model":"test","input":"continue","store":true,"previous_response_id":parent});
        if let Some(options) = options {
            request["prompt_cache_options"] = options;
        }
        let response = client
            .post(format!("{gateway_url}/v1/responses"))
            .json(&request)
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let _: Value = response.json().await.unwrap();
    }
    // Raw null survives; typed null is omitted. Neither path supplies defaults.
    for store in [false, true] {
        let response = client
            .post(format!("{gateway_url}/v1/responses"))
            .json(&json!({"model":"test","input":"hi","store":store,"prompt_cache_options":null}))
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let _: Value = response.json().await.unwrap();
    }
    let requests = captured.lock().await;
    assert_eq!(requests.len(), 13);
    for (pair, options) in requests[..8].as_chunks::<2>().0.iter().zip([
        json!({}),
        json!({"mode":"implicit"}),
        json!({"ttl":"30m"}),
        json!({"mode":"explicit","ttl":"30m"}),
    ]) {
        assert_eq!(pair[0]["prompt_cache_options"], options);
        assert_eq!(pair[1]["prompt_cache_options"], options);
    }
    assert!(requests[8].get("prompt_cache_options").is_none());
    assert!(requests[9].get("prompt_cache_options").is_none());
    assert_eq!(requests[10]["prompt_cache_options"], json!({"mode":"implicit"}));
    for request in &requests[8..11] {
        assert_eq!(request["input"][0]["content"], "parent context");
        assert!(request.get("previous_response_id").is_none());
    }
    assert_eq!(requests[11].get("prompt_cache_options"), Some(&Value::Null));
    assert!(requests[12].get("prompt_cache_options").is_none());
    gateway.abort();
    model.abort();
}

#[tokio::test]
async fn sse_parity_preserves_explicit_mode_and_existing_content_breakpoint() {
    let (url, captured, model) = spawn_tool_search_sse_sequence(vec![final_message_sse(); 2]).await;
    let mut fixture = storage_backed_state(&url).await;
    enable_options(&mut fixture);
    let (gateway_url, gateway) = spawn_gateway(fixture.state.clone()).await;
    let options = json!({"mode":"explicit","ttl":"30m"});
    let input = json!([{"type":"message","role":"developer","content":[{"type":"input_text","text":"stable prefix",
        "prompt_cache_breakpoint":{"mode":"explicit"}}]},{"type":"message","role":"user","content":"changing question"}]);
    for store in [false, true] {
        let response = reqwest::Client::new()
            .post(format!("{gateway_url}/v1/responses"))
            .json(&json!({"model":"test","input":input,"stream":true,"store":store,"prompt_cache_options":options}))
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = response.text().await.unwrap();
        assert!(body.contains("response.completed"));
    }
    let requests = captured.lock().await;
    assert_eq!(requests.len(), 2);
    for request in requests.iter() {
        assert_eq!(request["prompt_cache_options"], options);
        assert_eq!(request["input"], input);
    }
    gateway.abort();
    model.abort();
}

#[tokio::test]
async fn invalid_conflicting_and_unqualified_options_fail_before_upstream_or_storage() {
    let (url, captured, model) = spawn_mock_vllm_json_capture().await;
    for supported in [false, true] {
        let mut fixture = storage_backed_state(&url).await;
        if supported {
            enable_options(&mut fixture);
        }
        let (gateway_url, gateway) = spawn_gateway(fixture.state.clone()).await;
        for store in [false, true] {
            for options in [
                json!({"mode":"auto"}),
                json!({"mode":null}),
                json!({"ttl":"24h"}),
                json!({"ttl":null}),
                json!({"prewarm":true}),
                json!({"comparison_response_id":"resp_1"}),
                json!([]),
            ] {
                let response = reqwest::Client::new()
                    .post(format!("{gateway_url}/v1/responses"))
                    .json(&json!({"model":"test","input":"hi","store":store,"prompt_cache_options":options}))
                    .send()
                    .await
                    .unwrap();
                assert_eq!(response.status(), StatusCode::BAD_REQUEST);
            }
            let mut request = json!({"model":"test","input":"hi","store":store,
                "prompt_cache_options":{"mode":"explicit"}});
            if supported {
                request["prompt_cache_retention"] = json!("24h");
            }
            let response = reqwest::Client::new()
                .post(format!("{gateway_url}/v1/responses"))
                .json(&request)
                .send()
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::BAD_REQUEST);
            let body: Value = response.json().await.unwrap();
            assert_eq!(body["error"]["param"], "prompt_cache_options");
            assert_eq!(
                body["error"]["code"],
                if supported {
                    "invalid_value"
                } else {
                    "unsupported_parameter"
                }
            );
        }
        let count = sqlx::query_scalar::<_, i64>("SELECT COUNT(*) FROM responses")
            .fetch_one(fixture.pool.as_ref())
            .await
            .unwrap();
        assert_eq!(count, 0);
        gateway.abort();
    }
    assert!(captured.lock().await.is_empty());
    model.abort();
}
