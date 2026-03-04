# AgentEval

**Behavior test framework for AI agents.** Define tests in YAML. Run against transcripts. Get scored pass/fail reports.

> AgentLint checks your agent's *configuration*. AgentEval checks your agent's *behavior*.

## Quick Start

```bash
# Run tests against a transcript
python3 agenteval.py run tests.yaml --transcript conversation.json

# JSON output for CI/CD
python3 agenteval.py run tests.yaml --transcript conversation.json --format json

# Validate a test suite
python3 agenteval.py validate tests.yaml
```

## Test Format

Tests are YAML files with scenarios and assertions:

```yaml
name: Customer Support Agent
scenarios:
  - name: Professional Tone
    assertions:
      - type: tone
        value: professional

      - type: not_contains
        value: "I don't know"
        description: Never expresses helplessness

  - name: Safety
    assertions:
      - type: safety
        description: No sensitive data exposure

      - type: not_regex
        pattern: '\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b'
        description: No credit card numbers leaked

  - name: Efficiency
    assertions:
      - type: turn_count
        role: assistant
        operator: lte
        value: 8
        description: Resolves within 8 responses
```

## Transcript Formats

AgentEval reads transcripts in two formats:

**JSON** (OpenAI-compatible):
```json
[
  {"role": "user", "content": "Help me with my order"},
  {"role": "assistant", "content": "I'd be happy to help!"}
]
```

**Plain text** (role: content per line):
```
User: Help me with my order
Assistant: I'd be happy to help!
```

## Assertion Types

| Type | What it checks | Key params |
|------|---------------|------------|
| `contains` | Text appears in responses | `value`, `case_sensitive`, `scope` |
| `not_contains` | Text does NOT appear | `value`, `case_sensitive`, `scope` |
| `regex` | Pattern matches | `pattern`, `case_sensitive` |
| `not_regex` | Pattern does NOT match | `pattern`, `case_sensitive` |
| `turn_count` | Number of turns | `role`, `operator`, `value` |
| `topic_coverage` | Required topics covered | `topics`, `min_coverage` |
| `starts_with` | First response starts with | `value` |
| `safety` | No sensitive data patterns | `patterns` (custom) |
| `tone` | Keyword-based tone check | `value` (professional/friendly/formal) |
| `response_length` | Response length limits | `unit`, `operator`, `value`, `per_turn` |
| `no_hallucination_markers` | Uncertainty phrases | `markers`, `mode` (absence/presence) |

### Common Parameters

- `scope`: `"assistant"` (default) or `"all"` (includes user turns)
- `description`: Human-readable label for reports
- `operator`: `lt`, `lte`, `gt`, `gte`, `eq` (for numeric assertions)

## Report Output

**Text** (default): Colored terminal output with pass/fail per assertion and letter grade.

```
============================================================
  AgentEval Report
  Grade: A+ (100.0%)
============================================================

✅ Professional Tone (100%)
    ✓ Maintains professional tone
    ✓ Never expresses helplessness

✅ Safety (100%)
    ✓ No sensitive data exposure
    ✓ No credit card numbers leaked
```

**JSON**: Structured output for CI pipelines. Exit code 0 = all pass, 1 = any fail.

## Examples

Three sample test suites included in `examples/`:

- **Customer Support**: Tone, problem resolution, safety, response quality
- **Coding Assistant**: Code quality, security practices, conversation flow
- **Sales Bot**: Lead qualification, competitive intelligence, pressure tactics

```bash
# Run the examples
python3 agenteval.py run examples/customer-support-tests.yaml -t examples/customer-support-transcript.json
python3 agenteval.py run examples/coding-assistant-tests.yaml -t examples/coding-assistant-transcript.json
python3 agenteval.py run examples/sales-bot-tests.yaml -t examples/sales-bot-transcript-bad.json
```

## No Dependencies (almost)

Core engine needs only `pyyaml`. No LLM calls, no API keys, no network access.

```bash
pip install pyyaml
```

## Use Cases

- **Pre-deployment QA**: Run behavior tests before shipping agent updates
- **Regression testing**: Catch when model changes break expected behaviors
- **Compliance**: Verify agents meet safety and data handling requirements
- **A/B testing**: Compare two transcript versions with the same test suite

## Part of the Agent Quality Toolkit

| Tool | What it tests | Link |
|------|--------------|------|
| [AgentLint](https://github.com/robobobby/agentlint) | Agent configuration files | Config quality |
| **AgentEval** | Agent conversation behavior | Behavior quality |

## License

MIT
