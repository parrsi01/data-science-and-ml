# Python Cheatsheet (Institutional Data/AI Lab)

## Short Simplified Definitions

- Virtual environment (`venv`): Isolated Python package environment for one project.
- Package: Reusable Python library installed with `pip`.
- Module: A Python file that can be imported.
- Dependency pinning: Recording exact package versions for reproducibility.

## Core Commands

```bash
python3 -m venv venv
source venv/bin/activate
python --version
pip install --upgrade pip
pip install -r requirements.in
pip freeze > requirements.txt
pytest -q
python -m ipykernel install --user --name institutional-ai-lab
deactivate
```

## Common Pitfalls

- Installing packages globally instead of inside `venv`
- Forgetting to update `requirements.txt` after dependency changes
- Mixing notebook-only experiments with production code logic
- Ignoring type/lint/test failures before commit

## Institutional Best Practices

- Keep reproducible environments per repository
- Separate exploratory notebooks from production modules
- Add tests for core transforms and model interfaces
- Prefer documented scripts and configs over manual notebook-only workflows
