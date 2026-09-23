# Runs

One scorecard per policy version: the pass rate of every scenario, the models
used, and the timestamp.

These are committed on purpose. The brief asks for before and after scores, and
a reviewer should be able to see the loop closed without needing an API key or
running anything. `vN.json` next to `policies/vN.json` is the evidence that the
change in behaviour produced the change in score.

`details` records why a scenario failed, so a regression can be traced to the
check that caught it.
