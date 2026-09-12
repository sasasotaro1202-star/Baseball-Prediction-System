# Migration setup

The legacy `Baseball` repository is not publicly cloneable from GitHub Actions. The migration workflow therefore uses a repository secret named `LEGACY_REPO_TOKEN`.

## Required token
Create a fine-grained Personal Access Token with access to the legacy `Baseball` repository and **Contents: Read-only** permission.

Do not commit or paste the token into source code.

## Add the secret
In `Baseball-Prediction-System`:

Settings -> Secrets and variables -> Actions -> New repository secret

Name:
`LEGACY_REPO_TOKEN`

Value:
The fine-grained token.

Then run:
Actions -> Migrate Baseball system from legacy repository -> Run workflow

The workflow copies only Baseball code from `v4.4-baseball-integration`, excludes Soccer/generated datasets, performs syntax/import compilation checks, and does not run a backtest.
