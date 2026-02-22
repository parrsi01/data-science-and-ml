# Linux Cheatsheet (Institutional Data/AI Lab)

## Short Simplified Definitions

- Shell: Text interface used to run commands and automate work.
- Filesystem: How files and folders are organized on the machine.
- Process: A running program.
- Environment variable: A named setting available to commands and scripts.

## Core Commands

```bash
pwd                  # show current directory
ls -la               # list files (including hidden)
cd <path>            # change directory
mkdir -p <dir>       # create directory tree
cp -r <src> <dst>    # copy recursively
mv <src> <dst>       # move/rename
rm -r <path>         # remove recursively (use carefully)
find . -maxdepth 2   # inspect directory tree
grep -R "text" .     # search text in files
tail -f <file>       # follow log output
htop                 # interactive process monitor (if installed)
tmux                 # persistent terminal session manager (if installed)
```

## Common Pitfalls

- Running destructive commands (`rm -r`) in the wrong directory
- Editing system files without backups or change records
- Mixing project environments and global Python packages
- Assuming commands are portable across Linux/macOS/Windows

## Institutional Best Practices

- Use explicit paths in scripts for repeatability
- Log commands used for data preparation and system changes
- Prefer version-controlled scripts over manual terminal steps
- Minimize root/sudo usage and document when it is required
