# Deepfield PTv3 workspace

The v1.1 catalog is the canonical Pointcept/PTv3 annotation workspace for
Deepfield. SpineLab and the Board Rack training directories are provenance and
checkpoint sources; they are not forward-development workspaces.

## Project boundaries

| Project | Role | Training status |
|---|---|---|
| `Deepfield · Gold247 · 6-class [LOCKED]` | Frozen human-labelled source of truth | Approved baseline |
| `Deepfield · Polish Queue · 396 teacher seeds` | Full6 predictions awaiting manual correction | Never train until reviewed |
| `Deepfield · Quarantine · 18 rejected or incomplete` | Cleared or incomplete samples | Excluded |
| `Deepfield · Cervical C1-C2 · 24 unlabeled` | C1-C2-specific annotation queue | Separate anatomy policy |
| `Deepfield · Unlabeled VerSe · 6 bones` | Remaining canonical 32K bones | Manual review needed |
| `Deepfield · Specialist · ...` | Gold247 remaps for compatible specialist checkpoints | Evaluation/fine-tune source |
| `Deepfield · Legacy · ...` | Exact historical taxonomies and labels | Archive only |
| `Deepfield · Model Archive · ...` | Checkpoints whose output schema is not an active taxonomy | Archive only |

Gold247 contains 247 complete human-labelled bones from 12 subjects with a
frozen subject split of 170 train, 38 validation, and 39 sealed test. The 396
teacher seeds are explicitly not ground truth. The migration also retains 23
historical/current PTv3 checkpoints with their class maps frozen in the v1.1
model registry.

## Manual-polish workflow

1. Open `Deepfield · Polish Queue · 396 teacher seeds`.
2. Correct the existing prediction rather than repainting from zero.
3. Treat any point assigned to the wrong anatomical structure as higher
   priority than incomplete coverage of the correct surface.
4. Review the cloud from superior, inferior, lateral, posterior, and oblique
   views before considering it complete.
5. Do not add a corrected cloud to a training export until it has an explicit
   review record. Keep Gold247 unchanged.
6. Build the next model from Gold247 plus reviewed additions with subject-level
   train/validation/test separation. Never mix the untouched teacher queue into
   supervised ground truth.

The active six-class schema is:

1. `Unlabeled`
2. `Superior_Endplate`
3. `Inferior_Endplate`
4. `Pedicle`
5. `Body_Wall`
6. `Spinous_Process`

The specialist projects are deterministic remaps of Gold247, so they can be
used for controlled comparison without modifying the source labels.

## Reproduce or audit the migration

Audit only:

```bash
.venv/bin/python tools/migrate_deepfield_ptv3_workspace.py
```

Apply after reviewing the audit and closing 3Photon:

```bash
.venv/bin/python tools/migrate_deepfield_ptv3_workspace.py --apply
```

The apply path creates a timestamped catalog backup and migration receipt. It
is additive and idempotent: existing destination labels must match exactly,
and a conflicting array aborts the run instead of being overwritten.
