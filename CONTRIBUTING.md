# Contributing

Thank you for helping. The most useful contributions are videos where Meteor Cut gets something wrong: a missed
meteor, a plane called a satellite, a person kept as a meteor. Open an issue with the video (or a link), the command,
and `summary.json`.

## Development setup

```bash
git clone https://github.com/Deneb-Systems/meteor-cut
cd meteor-cut
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e . --group dev
pre-commit install
```

The tests do not need Sky Tracker or a licence key: `tests/conftest.py` has a stand-in tracker that finds bright blobs,
and a small synthetic night video with a meteor, a satellite and a scene cut. To run Meteor Cut on real videos you
need the SDK (no key: Meteor Cut uses its free mode).

## Checks

```bash
ruff check .          # lint
ruff format .         # format
mypy                  # strict types for the package
pytest --cov          # tests, with coverage (85 % required)
```

CI runs the same on Linux, macOS and Windows.

## Changing detection

Detection changes need evidence beyond unit tests: run before and after on real videos, and say in the pull request
what changed (meteors gained or lost, and whether each is real). A contact sheet of the meteors found, each one's frames
stacked into one image, is the quickest way to check them.

Keep thresholds in `Rules`, as fractions of the frame width, with a comment saying what they protect against.

## Style

- Type hints everywhere in `src/`; `mypy --strict` must pass.
- Docstrings say what a function is for and anything surprising; the code says how.
- User-facing messages (errors, CLI output, the page) are plain and specific, and say what to do.
- Add an entry under "Unreleased" in `CHANGELOG.md` for anything a user would notice.

## Releasing

1. Move the "Unreleased" entries in `CHANGELOG.md` under a new version heading and set `version` in `pyproject.toml`.
2. Commit, then tag and push: `git tag v0.2.0 && git push origin v0.2.0`.
3. The release workflow checks the tag matches the version, builds, publishes to PyPI and creates the GitHub release.

PyPI publishing uses trusted publishing: the project's PyPI settings trust `release.yml` in this repository (with the
`pypi` environment), so no token is stored anywhere.
