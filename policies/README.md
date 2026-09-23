# Policies

Each file is the agent's behaviour at one version. This is not configuration
sitting next to the agent, it **is** the agent: `Policy.render()` composes the
system prompt from `system_base`, `rules` and `few_shots`.

That is what makes the improvement loop real. A patch is a mutation of this JSON
and a version bump, not a human editing a prompt string between runs. Diff two
versions and you are reading exactly what the loop learned:

```bash
diff <(jq . v1.json) <(jq . v2.json)
```

`from_failure` on each rule records which scenario caused it to be added.
