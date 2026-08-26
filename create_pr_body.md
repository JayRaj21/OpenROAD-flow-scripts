## Summary

- **Triage agent** (`flow/util/triage_agent.py`): reads stage-by-stage metrics and diagnoses P&R quality failures (CTS→GRT parasitic cliff, congestion, residual violations) using Claude Opus 5 with adaptive thinking
- **Closed-loop agent** (`flow/util/loop_agent.py`): autonomously applies triage recommendations, re-runs affected flow stages via Docker, verifies improvement, and writes successful parameters back to `config.mk`
- **Post-CTS/GRT hooks** (`flow/scripts/post_cts_timing_repair.tcl`, `post_grt_timing_repair.tcl`): iterative cell upsizing using placement parasitics; activated by the loop agent via `POST_CTS_TCL`/`POST_GLOBAL_ROUTE_TCL`
- **Metrics collector** (`flow/util/pr_metrics.py`): parses WNS/TNS/Fmax/overflow from ORFS report and log files across all P&R stages into a single trajectory table
- **aes/nangate45 baseline fix**: triage agent correctly identified the CTS→GRT parasitic underestimation cliff; applying `SETUP_SLACK_MARGIN=0.03` + `POST_CTS_TCL` closed timing from WNS −0.330 ns to 0.000 ns, Fmax +49 MHz
- **Congestion ML pipeline** (`flow/ml/congestion/`): U-Net + GNN models for routing congestion prediction, thermal estimation, and variant generation
- **Unit tests** (`flow/util/test_loop_agent.py`): 24 tests covering allowlist enforcement, hook path translation, stale file sets, and config write-back — no API key or Docker required

## Test plan

- [ ] `python3 flow/util/test_loop_agent.py` — all 24 unit tests pass
- [ ] `ANTHROPIC_API_KEY=<key> python3 flow/util/loop_agent.py --platform nangate45 --design aes --tag base` — loop agent closes timing and writes params to config.mk
- [ ] `ANTHROPIC_API_KEY=<key> python3 flow/util/triage_agent.py --platform nangate45 --design aes --tag base` — triage agent diagnoses the CTS→GRT cliff correctly

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_019kUei3bDhQVmVchT8GsDMo
