---
name: cross-verify
description: Run external cross-verification with Gemini and/or Claude, then synthesize the results into Codex's final judgment. Use when the user asks to "검증해줘", "cross-verify", "크로스체크", verify an answer or technical judgment, or get a second-model review.
---

# Cross Verify

## Overview

Use Gemini and/or Claude as independent reviewers for technical answers and judgments. Treat their outputs as evidence; Codex must still inspect the project context and make the final judgment.

## Workflow

1. Ask the user which verification mode to run unless they already specified it:
   - `1`: Gemini (`gemini-2.5-pro`)
   - `2`: Claude (`opus-4.7`)
   - `3`: Gemini + Claude in parallel, recommended

2. Build a concise prompt that includes:
   - the exact question or answer to verify
   - relevant context (problem conditions, evaluation criteria)
   - the expected output format
   - an instruction to identify assumptions, risks, and concrete recommendations

3. Run the selected verifier:
   - Gemini: `gemini -m gemini-2.5-pro -p "..."`.
   - Claude: `claude{account} -m opus-4.7 -p "..."`, where `{account}` is `1` or `2` if the user specified `claude1` or `claude2`. Ask which account to use if Claude is selected and no account is known.
   - Parallel mode: run Gemini and Claude concurrently when possible, then compare the outputs.

4. Review the verifier output instead of forwarding it blindly:
   - Check claims against local files before accepting them.
   - Separate confirmed findings from model opinions.
   - Resolve disagreements explicitly.
   - State Codex's final recommendation.

5. Save the result if needed.

## Guardrails

- Do not treat external model output as authoritative.
- Do not expose secrets or API keys in prompts or saved files.
- Prefer concise prompts with exact context over broad dumps.
- If network or CLI access fails, report that verification could not be run and save no fabricated result.
