# AGENTS.md

This repository is the shareable extraction of conditional GGRF + Joint
Node–Edge Transformer EEG representation learning. The default downstream mode
is the deterministic `node_hidden_no_transport` readout.

## Startup Workflow

Before writing code:

1. **Confirm working directory** with `pwd`
2. **Read this file** completely
3. Read `README.md`, `docs/architecture.md`, and `docs/data_format.md`.
4. **Run `./init.sh`** to verify environment is healthy
5. **Read `feature_list.json`** to see current feature state
6. Review recent commits when this directory is initialized as a standalone Git repository.

If baseline verification is failing, repair that first before adding new scope.

## Working Rules

- **One feature at a time**: Pick exactly one unfinished feature from `feature_list.json`
- **Verification required**: Don't claim done without running verification commands
- **Update artifacts**: Before ending session, update `progress.md` and `feature_list.json`
- **Stay in scope**: Don't modify files unrelated to the current feature
- Preserve the fixed `(11,19,100)` history and 171-edge contracts.
- Never equate physical EEG time with Flow transport time `tau`.
- Never describe the no-transport classifier as ODE sampling (`NFE=0`).
- Do not add raw TUSZ data, absolute private paths, caches or checkpoints to Git.
- Fit normalization, source scales and dev thresholds without test access.

## Required Artifacts

- `feature_list.json` — Feature state tracker (source of truth)
- `progress.md` — Session continuity log
- `init.sh` — Standard startup and verification path
- `session-handoff.md` — Optional, for larger sessions

## Definition of Done

A feature is done only when ALL of the following are true:

- [ ] Target behavior is implemented
- [ ] Required verification actually ran (tests / lint / type-check)
- [ ] Evidence recorded in `feature_list.json` or `progress.md`
- [ ] Repository remains restartable from standard startup path

## End of Session

Before ending a session:

1. Update `progress.md` with current state
2. Update `feature_list.json` with new feature status
3. Record any unresolved risks or blockers
4. Leave the project restartable with `./init.sh`.

## Verification Commands

```bash
# Full verification (recommended)
./init.sh
```

Required checks:
- `python -m pytest -q`
- `python -m eeg_cgfm.cli.smoke`

## Escalation

If you encounter:
- **Architecture decisions**: Consult project architecture docs if present, otherwise ask user
- **Unclear requirements**: Check product/requirements docs if present, otherwise ask user
- **Repeated test failures**: Update progress, flag for human review
- **Scope ambiguity**: Re-read `feature_list.json` for definition of done
