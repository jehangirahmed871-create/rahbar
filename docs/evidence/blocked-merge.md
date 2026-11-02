# Blocked Merge Evidence: Evaluation CI Gate Enforcement

Pull Request #3 (`dev` -> `main`): Failed Evaluation Gate

### CI Job Log (`e2e-eval` step output):
```text
Running eval/run_eval.py against live Kind cluster...
Loaded 16 questions from eval/golden.yaml
  q01      0.42s  What CGPA does an undergraduate student need to maintain...
  q02      0.38s  What is the minimum percentage of attendance required...
  ...
  q13      0.29s  What is the monthly room rent and allotment criteria for...

GATE FAILED
  - recall@5 dropped by 14.28%, exceeding allowable 5.0%
  - any must-abstain question answered from context: question q13 answered_from_context was True

Error: Process completed with exit code 1.
```

### GitHub Pull Request State:
```text
Merging is blocked.
All required status checks must pass before merging.
✖ End-to-End Evaluation Gate (e2e-eval) - Failed in 2m 14s

[Merge pull request] button is GREYED OUT / DISABLED.
```

### Fix and Green Merge:
PR #4: Restored structure-aware 400-token chunking and enforced strict relevance floor abstention (`0.50`).
```text
  q13      0.25s  What is the monthly room rent and allotment criteria for...
Results written to eval/results.json:
  recall@3: 0.8571
  recall@5: 0.9286
  citation_validity: 1.0000
  abstention_correctness: 1.0000

gate passed.
✔ End-to-End Evaluation Gate (e2e-eval) - Successful in 1m 58s
[Merge pull request] button enabled (GREEN).
```
