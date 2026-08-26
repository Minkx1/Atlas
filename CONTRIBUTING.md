# Contributing to Atlas

Thank you for your interest in contributing to Atlas!

## Getting Started

1. Fork the repository.
2. Clone your fork locally.
3. Install the development dependencies:

   ```bash
   pip install .
   ```
4. Create a branch for your changes:

   ```bash
   git checkout -b feat/my-change
   ```

## Development

Before submitting changes, make sure the project still works and that tests pass.

Run the linter:

```bash
ruff check .
```

If your change affects formatting or code quality, make sure the relevant checks pass before opening a pull request.

## Commit Messages

Atlas follows the [Conventional Commits](https://www.conventionalcommits.org/) specification.

Examples:

```text
feat: add plugin discovery
fix: handle missing TTS model
refactor: simplify event dispatching
docs: update architecture documentation
test: add EventManager tests
chore: update development dependencies
```

Use `fix` for actual bug fixes. Small improvements that don't add user-facing functionality should generally use `refactor`, `chore`, or another appropriate type rather than `feat`.

## Pull Requests

Before opening a pull request:

* Make sure your changes are focused and reasonably scoped.
* Run the test suite.
* Run the linter.
* Update documentation when necessary.
* Add or update tests for new behavior or bug fixes.
* Follow the existing project structure and coding style.

Please structure your PR based on PULL_REQUEST_TEMPLATE, e.g.

Please provide a clear description of:

* What changed.
* Why the change was needed.
* How it was tested.
* Any limitations or considerations reviewers should know about.

## Issues

Before opening an issue, check whether a similar issue already exists.

Please structure your issue based on relevant ISSUE_TEMPLATE, e.g.

For bug reports, include:

* Atlas version or commit.
* Operating system.
* Python version, if running from source.
* Relevant logs or error messages.
* Steps to reproduce the problem.

For feature requests, describe the problem or use case rather than only suggesting an implementation.

## Code Style

Keep the code:

* Simple and readable.
* Modular and testable.
* Consistent with the existing architecture.
* Free of unnecessary dependencies.
* Ruff-styled.

Avoid introducing unnecessary coupling between components. Atlas is designed around modular components and event-driven communication, so new functionality should preserve those boundaries where practical.

## License

By contributing to Atlas, you agree that your contributions will be licensed under the same license as the project.
