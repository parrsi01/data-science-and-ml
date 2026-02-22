# Git Cheatsheet (Institutional Data/AI Lab)

## Short Simplified Definitions

- Repository: A tracked project folder managed by Git.
- Commit: A saved snapshot of changes with a message.
- Branch: An isolated line of work.
- Remote: A hosted copy of the repository (e.g., GitHub).

## Core Commands

```bash
git init
git status
git add .
git commit -m "message"
git branch
git checkout -b feature/name
git pull --rebase
git push -u origin main
git log --oneline --decorate --graph
git diff
```

## Common Pitfalls

- Committing large datasets or secrets
- Vague commit messages that hide operational impact
- Working directly on `main` for risky changes
- Forgetting to sync before pushing

## Institutional Best Practices

- Use meaningful commit messages tied to change intent
- Review changes before commit (`git diff`)
- Protect default branches with review requirements
- Tag releases used in reports, audits, or production deployments
