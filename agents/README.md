# Agent control

Per-owner task files so every agent can bootstrap from repo state without chat
history. Source of truth for tasks is [`../BACKLOG.md`](../BACKLOG.md); the files
here are filtered views plus working scope.

| Dir | Owner | Focus |
|-----|-------|-------|
| [`gerron-claude/`](gerron-claude/TASK.md) | Gerron/Claude | contracts, world state, scheduler, verification/replanning, NL + constraint mutation |
| [`gerron-gpt/`](gerron-gpt/TASK.md) | Gerron/GPT | MuJoCo env, arm primitives, table setting, Intel packaging, Qualcomm/Arduino, launchers |
| [`gerron-kimi/`](gerron-kimi/TASK.md) | Gerron/Kimi | arm capability map, evaluator, flourish envelope, Qualcomm profiling, receipts, trials |
| [`bryan-codex/`](bryan-codex/TASK.md) | Bryan/Codex | live graph UI, bimanual viz, device graph UI, judge demo mode, README/presentation |
| [`damion-claude/`](damion-claude/TASK.md) | Damion/Claude | requirements audit, deliberate breakage, rubric audits, acceptance suite, red-team |

## Working rules

- Claim a task by setting its row to `~` in `BACKLOG.md` and noting it in your
  `TASK.md`; mark `x` when the "Done when" is met.
- Don't start a task whose dependencies aren't `x` unless you're stubbing behind
  a frozen interface.
- The full repo scaffold (repo maps, coding pointers, active scopes) is
  **OQ-002**, owned by Gerron/GPT — expand this directory there.
