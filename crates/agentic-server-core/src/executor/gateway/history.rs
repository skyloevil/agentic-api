use crate::executor::request::RequestContext;
use crate::tool::ToolRegistry;
use crate::types::io::{InputItem, OutputItem, ResponsesInput};

pub(in crate::executor) fn append_output_items_to_input(input: &mut ResponsesInput, output_items: &[OutputItem]) {
    for input_item in output_items.iter().filter_map(OutputItem::to_input_item) {
        input.push(input_item);
    }
}

pub(in crate::executor) fn append_tool_outputs(ctx: &mut RequestContext, tool_outputs: Vec<InputItem>) {
    for output in tool_outputs {
        ctx.new_input_items.push(output.clone().into());
        ctx.enriched_request.input.push(output);
    }
}

pub(in crate::executor) fn append_gateway_calls_to_new_input(
    ctx: &mut RequestContext,
    output_items: &[OutputItem],
    registry: &ToolRegistry,
) {
    ctx.new_input_items.extend(output_items.iter().filter_map(|item| {
        let OutputItem::FunctionCall(call) = item else {
            return None;
        };
        registry
            .is_gateway_owned_name(&call.name)
            .then(|| InputItem::FunctionCall(call.clone().into()).into())
    }));
}
