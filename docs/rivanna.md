# Running on Rivanna

How to run every experiment on UVA's Rivanna cluster with Slurm. Rivanna has no Docker, so
this replaces the README's Docker steps there; the commands inside the scripts are the same
ones the README describes.

## Getting a shell

- **Open OnDemand** (`ood.hpc.virginia.edu`, no VPN): Clusters → Rivanna shell, or Interactive
  Apps → Code Server for VS Code in the browser.
- **VS Code on your own machine:** Remote-SSH to `login.hpc.virginia.edu`. Off Grounds this
  needs the UVA Anywhere VPN.

Batch jobs keep running after you disconnect. The OnDemand shell closes after a few idle
minutes; just open a new one.

## One-time setup

```bash
mkdir -p /scratch/$USER/ds6050/logs        # Slurm writes logs here and won't create it
cd /scratch/$USER/ds6050
git clone https://github.com/jimartter-git/ds6050-final-project-public.git repo
cd repo
sbatch scripts/rivanna/00_setup.sbatch
```

Setup builds a Python 3.11 env at `/scratch/$USER/ds6050/env` with the
pinned packages and downloads the labels into `data/` (it streams a 3.2 GB archive and keeps
only the 2,326 label files). The log ends with `SETUP_OK`.

## Running everything

Submit from the repo root. Check that the smoke test passes before the main batch:

```bash
sbatch scripts/rivanna/01_smoke.sbatch          # tests and short runs; log ends SMOKE_OK
a=$(sbatch --parsable scripts/rivanna/02_arm.sbatch)
sbatch --dependency=afterok:$a scripts/rivanna/03_analyze.sbatch
```

Watch progress with `squeue -u $USER` and `sacct -j <job id> --format=JobID,State,Elapsed`.
Logs are in `/scratch/$USER/ds6050/logs/`.

`02_arm.sbatch` runs 21 jobs, six at a time: MB0, MT0, MT1, MT1 with visibility dropout, V1,
and MSTD and MVA (each pretrained, then fine-tuned), with seeds 42, 43 and 44. Each job trains,
evaluates validation and test, and runs two controlled-occlusion sweeps: single joint-frames
(`-occlusion`) and 8–16 frame spans (`-spans`). `03_analyze.sbatch` writes seed summaries,
training time, pretraining reconstruction error, McNemar tests for RQ1, RQ2a and RQ2b,
occlusion tertiles and both occlusion curves to `outputs/rivanna/analysis/`, then packs
everything except checkpoints into `/scratch/$USER/ds6050/results-<job id>.tgz`.

To run again, use a new output folder for both jobs, because training refuses to overwrite:
`sbatch --export=ALL,OUT=outputs/rivanna2 scripts/rivanna/02_arm.sbatch`, and the same
`--export` for the analysis job.

## Time and cost

The Nov 6 run (18 jobs, before the span sweeps and the dropout model) used about 100
allocation units on RTX A6000s and finished within an afternoon. Expect somewhat more now.
The scripts ask for A6000s on purpose: the same GPU type reproduces results exactly.

## Getting results off Rivanna

Download the tarball through the OnDemand file browser, or `scp` it from a login node.
Scratch files that go unused for 90 days are deleted, so don't leave results only there.

## Good to know

- Keep the default occlusion seed (0) so every model sees the same hidden joints.
- Test numbers are for reporting only; choose models and settings on validation.
