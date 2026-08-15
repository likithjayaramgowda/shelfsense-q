# Handoff

Single source of truth for "where am I and what's next".
Update this **before you stop working on a machine**, every time.
It is the thing that prevents losing a day to confusion on Monday morning.

---

## Current state

- **Last active machine:** laptop
- **Last updated:** 2026-08-15
- **Branch:** main

## Done

- [ ] nothing yet

## Next task

Bootstrap the repo and push to GitHub.

## Blockers

None.

## Notes for the other machine

Nothing yet.

---

### The relay rule

Only ONE machine is active at a time. Before leaving a machine:

```
git add -A
git commit -m "wip: <what you did>"
git push
```

Before starting on a machine:

```
git pull --rebase
```

If you forget the pull, you get a conflict and lose 20 minutes. If you forget
the push, you lose the work entirely until you're back at that machine.
