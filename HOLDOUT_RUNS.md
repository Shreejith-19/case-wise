# dev_holdout evaluations (plan.md D14)

| when | system | predictions | corpus | note |
|---|---|---|---|---|
| 2026-11-17T15:25 | B-full | `C:\Users\91910\Documents\Capstone\Sem7\Claude\runs\B-full\pred_dev_holdout.json` | 33b624048b53 | Phase 3 frozen B-full (D15): single dev_holdout check |
| 2026-11-17T15:27 | B-full | `C:\Users\91910\Documents\Capstone\Sem7\Claude\runs\B-full\pred_dev_holdout.json` | 33b624048b53 | Re-score after ground_quotes v2 bug fix (found on dev_tune IN-STAT-0413/0464; quote marks only, LLM outputs cached and unchanged) |
| 2026-11-18T02:14 | C-full | `C:\Users\91910\Documents\Capstone\Sem7\Claude\runs\C-full\pred_dev_holdout.json` | 33b624048b53 | Phase 4 frozen C-full (D16): single dev_holdout check against the pre-registered criterion |
| 2026-11-18T16:49 | E-agent | `C:\Users\91910\Documents\Capstone\Sem7\Claude\runs\E-agent\pred_dev_holdout.json` | 33b624048b53 | Phase 6 E (agentic upper bound, untuned agent-v1): single dev_holdout check |
| 2026-11-18T17:21 | C-LS | `C:\Users\91910\Documents\Capstone\Sem7\Claude\runs\C-LS\pred_dev_holdout.json` | 33b624048b53 | Post-test option 1 C-LS: dev gate evaluation (pre-registered; no tuning) |
