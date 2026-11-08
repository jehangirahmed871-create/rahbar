# Branch Protection Evidence

GitHub repository branch protection rules configured for `main`:

```text
Branch protection rule: main
Status: Enabled

Protection Settings:
- Require a pull request before merging: Enabled
  - Required approvals: 1
  - Dismiss stale pull request approvals when new commits are pushed: Enabled
  - Require review from Code Owners: Disabled
- Require status checks to pass before merging: Enabled
  - Require branches to be up to date before merging: Enabled
  - Required status checks:
    - Lint & Code Style (lint)
    - Unit & Offline Stub Tests (test)
    - Validate Kubernetes Manifests (manifests)
    - Vulnerability Scan (scan)
    - End-to-End Evaluation Gate (e2e-eval)
- Require conversation resolution before merging: Enabled
- Do not allow bypassing the above settings: Enabled (enforced for administrators)
- Restrict who can push to matching branches: Enabled (No direct pushes allowed)
```
