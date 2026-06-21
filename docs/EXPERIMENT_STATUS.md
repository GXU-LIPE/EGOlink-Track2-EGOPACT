# Experiment Status

## Protected Best

Current protected best from the audited state:

```text
version: V6_1_3_gpt55_guarded_endpoint
run_id: gpt55_endpoint_gate_20260617_105936
model: gpt-5.5
endpoint: https://ai-pixel.online/v1
```

Metrics on the fixed 4-task gate:

| Metric | Value |
|---|---:|
| joint_success | 0.500 |
| result_success | 0.750 |
| tool_success | 0.500 |
| micro_tool_accuracy | 0.7083333333333333 |
| avg_rounds | 4.5 |
| avg_tool_calls | 30.0 |

## Important Interpretation

V7/V8/V9/V10 variants are preserved in the repository because they contain useful diagnostic and candidate modules. They should not be described as final-ready without broader validation.

Observed pattern:

- V8_2 improved a 4-task smoke test, but did not generalize on `validation_A_small`.
- V9/V9.5/V10 can improve some micro-level metrics, but joint success remained insufficient on wider validation.
- No final submission is represented by this repository snapshot.

