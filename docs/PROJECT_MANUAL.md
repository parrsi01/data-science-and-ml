# Institutional Data & AI Engineering Lab

- Author: Simon Parris
- Date: 2026-02-22

## 1. Purpose of this Lab

This lab provides a reproducible workspace for institutional-grade data science, machine learning, and AI engineering. It supports aviation, humanitarian, and scientific use cases where traceability, quality control, and documentation are mandatory.

## 2. System Architecture Overview

- Local development environment with Python virtual environment (`venv`)
- Source code in `src/` and tests in `tests/`
- Data and model artifact placeholders in `datasets/` and `models/`
- Documentation and operational guidance in `docs/` and `cheatsheets/`
- MLOps and deployment assets in `mlops/` and `docker/`
- Experiment and algorithm-specific workstreams in `algorithm_marl_xgboost/`

## 3. How to Rebuild From Scratch

### System-Level Bootstrap (Ubuntu 22.04)

Use the following commands on a fresh machine (not executed automatically in this repository setup):

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y \
  python3.11 \
  python3.11-venv \
  python3-pip \
  git \
  make \
  build-essential \
  curl \
  wget \
  unzip \
  tree \
  htop \
  tmux \
  software-properties-common \
  apt-transport-https \
  ca-certificates \
  gnupg \
  lsb-release
```

### VS Code Install (Optional)

```bash
wget -qO- https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor > packages.microsoft.gpg
sudo install -o root -g root -m 644 packages.microsoft.gpg /usr/share/keyrings/
sudo sh -c 'echo "deb [arch=amd64 signed-by=/usr/share/keyrings/packages.microsoft.gpg] https://packages.microsoft.com/repos/code stable main" > /etc/apt/sources.list.d/vscode.list'
sudo apt update
sudo apt install code -y
```

### Repository Build

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.in
pip freeze > requirements.txt
pytest -q
```

## 4. Scientific Reproducibility Rules

- Record package versions in `requirements.txt` after dependency changes
- Keep raw data immutable; use derived datasets with clear lineage
- Version code, configs, and experiment metadata together
- Use deterministic seeds for stochastic training where practical
- Document assumptions, sampling criteria, and exclusion logic

## 5. Audit & Compliance Considerations

- Do not commit sensitive data, credentials, or protected identifiers
- Maintain access logs and provenance outside this repo when required
- Ensure model inputs/outputs can be explained and traced
- Preserve change history through Git commits and pull requests
- Apply documented review/approval steps before operational deployment

## 6. Model Risk Awareness

- Treat model outputs as decision support unless formally approved
- Monitor performance drift and data quality degradation
- Check fairness, safety, and unintended consequence risks
- Document confidence limits and known failure modes
- Define rollback procedures for production incidents

## 7. Version Control Rules

- One logical change per commit where possible
- Use descriptive commit messages
- Tag releases used for reporting or production decisions
- Protect main branches with review policies in remote hosting
- Keep notebooks cleaned before commit when feasible

## 8. Experiment Tracking Policy

- Assign each experiment a unique identifier
- Log dataset version, code commit, parameters, metrics, and artifacts
- Store validation methodology and acceptance criteria
- Retain failed experiments when they inform risk or design decisions
- Use MLflow or an equivalent registry/tracker for reproducible comparisons
