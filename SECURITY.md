# Security Policy

This repository contains analysis code for public summary statistics and is intended to be safe to publish publicly. It does not require any application secrets to run.

## Supported Versions

Security fixes will be applied on the default branch and included in the next tagged release.

## Safe Use Guidelines

- Do not commit API keys, access tokens, `.env` files, notebooks with embedded secrets, or private study files.
- Do not replace the public OpenMed_AI / PGC inputs with restricted or participant-level data in this public repository.
- Review `git status` before every commit.
- Prefer creating a separate private repository if you need to test non-public data or credentials.

## Reporting A Vulnerability

If you believe the repository contains a secret, sensitive file, or unsafe instruction, report it privately to the maintainers instead of opening a public issue.
