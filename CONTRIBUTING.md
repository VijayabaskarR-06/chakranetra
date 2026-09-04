# Contributing to Chakranetra

Thank you for your interest in contributing! Chakranetra is an open civic-tech platform — every improvement, no matter how small, helps make roads safer.

---

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How to Contribute](#how-to-contribute)
- [Setting Up Your Development Environment](#setting-up-your-development-environment)
- [Making Changes](#making-changes)
- [Pull Request Guidelines](#pull-request-guidelines)
- [Reporting Bugs](#reporting-bugs)
- [Requesting Features](#requesting-features)
- [Code Style](#code-style)

---

## Code of Conduct

This project is welcoming to all contributors. Please be respectful and constructive in all interactions. We follow the [Contributor Covenant](https://www.contributor-covenant.org/) code of conduct.

---

## How to Contribute

There are many ways to contribute beyond writing code:

- 🐛 **Report bugs** using the bug report template
- 💡 **Suggest features** using the feature request template
- 📖 **Improve documentation** (README, ARCHITECTURE.md, docstrings)
- 🧪 **Write or improve tests** in the `tests/` directory
- 🗺️ **Expand city/region coverage** by submitting sample GPS tracks or road photos
- 🌐 **Translate** the dashboard UI into regional languages

---

## Setting Up Your Development Environment

### Prerequisites

- Python 3.9+
- Git
- (Optional) Docker & Docker Compose

### Steps

1. **Fork** the repository on GitHub and clone your fork:

   ```bash
   git clone https://github.com/<your-username>/chakranetra.git
   cd chakranetra
   ```

2. **Create a virtual environment** and install dependencies:

   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # macOS / Linux
   source .venv/bin/activate

   pip install -r requirements.txt
   ```

3. **Run the demo** to verify your setup:

   ```bash
   python run_demo.py
   ```

4. **Run the test suite** to make sure everything passes:

   ```bash
   pytest tests/ -v
   ```

5. **(Optional) Start the API server:**

   ```bash
   uvicorn server.app:app --reload
   # → http://127.0.0.1:8000
   ```

---

## Making Changes

1. **Create a new branch** from `main` using a descriptive name:

   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/issue-description
   ```

2. **Make your changes.** Keep commits small and focused. Write clear commit messages:

   ```
   feat: add haversine dedup radius to config
   fix: correct mask thresholding order in detector
   docs: expand ARCHITECTURE.md with scaling notes
   test: add parity test for JS scoring engine
   ```

3. **Run the tests** after every non-trivial change:

   ```bash
   pytest tests/ -v --cov=roadlens --cov-report=term-missing
   ```

4. **Check parity** if you touch the detector or scoring engine:

   ```bash
   python tools/check_onnx_parity.py
   ```

---

## Pull Request Guidelines

- **Target the `main` branch** for all PRs.
- **Link the relevant issue** in the PR description (e.g., `Closes #42`).
- **Fill out the PR template** — it exists to make reviews faster.
- Keep PRs focused: one feature or fix per PR. Large PRs are harder to review and slower to merge.
- All CI checks (GitHub Actions) must pass before a PR can be merged.
- At least **one approving review** from a maintainer is required.

---

## Reporting Bugs

Use the **Bug Report** issue template (`.github/ISSUE_TEMPLATE/bug_report.md`). Please include:

- What you expected to happen
- What actually happened (error message, screenshot, or log output)
- Steps to reproduce
- Your environment (OS, Python version, GPU/CPU)

---

## Requesting Features

Use the **Feature Request** issue template (`.github/ISSUE_TEMPLATE/feature_request.md`). Describe:

- The problem this feature solves
- Your proposed solution
- Alternatives you've considered

---

## Code Style

- Follow **PEP 8** for Python code.
- Use **type hints** for all public function signatures.
- Write **docstrings** for all public functions and classes.
- All new logic must have at least one corresponding **test** in `tests/`.
- All tunable parameters must be added to **`config.yaml`**, not hardcoded.

---

## Questions?

Open a [GitHub Discussion](https://github.com/VijayabaskarR-06/chakranetra/discussions) or file an issue with the `question` label. We're happy to help!
