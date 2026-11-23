# Working on the code

- Branch from `main` and open a pull request; don't push to `main` directly.
- Run `python -m pytest -q` first, and say in the pull request what changed and how you
  checked it.
- New models go in `src/penn_action/models/` with a config in `configs/`. Reuse the shared
  data loading, masking, training and metrics code instead of writing a separate loop or
  notebook.
- If you change shared behaviour (preprocessing, masking, what a model is allowed to see),
  update the tests and the docs in the same pull request.
- Don't commit data, outputs, checkpoints or credentials.
