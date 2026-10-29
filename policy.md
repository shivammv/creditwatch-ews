# CreditWatch: Operating Policy (corporate credit desk)

*All numbers below come from `artifacts/universe_sim.json` (1,000 simulated 800-borrower books per default rate, drawn from the untouched benchmark hold-out). Re-run `python run_all.py` to refresh.*

## 1. Capacity and alert threshold
| Item | Setting |
|---|---|
| Universe / capacity | 800 borrowers, **40 deep reviews per quarter (5%)** |
| Cost of errors | missed default = **20 ×** a false alarm → expected cost = **20·FN + 1·FP** |
| Cost-optimal cut-off | Bayes threshold on calibrated probs: `C_FP/(C_FP+C_FN) = 1/21 ≈ 4.76%`. After prior-shift to the bank's own default rate, flag if p ≥ 4.76%. Hard 40-name cap still applies. |
| Hard cap | **never more than 40 "priority review" names**; names above the cut-off but outside the 40 go to the overflow list (section 4) |

## 2. What the cap costs (expected cost per quarter, 800 borrowers)
| Book default rate | Defaulters | Recall inside 40 | Cost @ 40 names | Cost-optimal list size (cost) | Flag nobody | Names above 1/21 |
|---|---|---|---|---|---|---|
| 1.0% | 8 | 50% | 115.5 | 33 (113.9) | 160 | ~18 |
| 2.0% | 16 | 49% | 194.0 | 44 (193.3) | 320 | ~66 |
| 3.9% | 31 | 46% | 361.2 | 144 (334.3) | 620 | ~189 |

**Reading it:** for a normal book (1-2% default rate) the 40-name budget is essentially the cost-optimal list size, so the 5% constraint is not binding. In a stressed book (≈4%) the optimum is ~144 names; the 40-name cap then costs ~8% more than optimal, still ~42% cheaper than flagging nobody. When more than 40 names exceed the cut-off, that is itself a **portfolio-stress trigger** (section 4).

## 3. Quarterly ranking method
1. Build features from the **latest financial statements public on the run date** (fiscal year counts only after its filing/results date).
2. Model: calibrated probability of distress from the India-portable LightGBM (39 ratios).
3. Overlay (log-odds additions, set as policy priors, *not fitted*: 10 defaulters cannot support fitting; weights in `src/rank.py::OVERLAY`): Altman Z'' distress +0.8 / grey +0.3; Beneish M > -1.78 +0.5; Piotroski F ≤ 3 +0.4; interest cover < 1 +0.5; soft signals (rating downgrade +0.6, auditor red flag +0.5, delayed filing +0.5, promoter pledge > 50% +0.4, related-party stress +0.4; combined cap +1.5).
4. Rank by final risk; ties broken by lower interest cover, then higher leverage. Banks/NBFCs skip Z''/M/F and carry a "model indicative only" tag.

## 4. Review workflow and escalation
| Step | Owner | Timing | Output |
|---|---|---|---|
| Day 1: list + memos auto-generated (top 40, SHAP top-3 drivers, Z/M/F, soft flags) | System | quarter start | ranked list |
| Triage (15-20 min/name) | Credit analyst | days 2-10 | accept / downgrade priority / escalate |
| Deep review of confirmed names | Analyst + sector head | by day 30 | limit, collateral, covenant or exit proposal |
| Credit committee | CRO desk | next sitting | approved actions (humans approve every limit change) |
| **Overflow list** (above cut-off, rank 41+) | Sector head | quarterly | light-touch monitor: refresh financials, call RM; promote on any new soft signal |
| **Stress trigger**: > 100 names above cut-off (the 2%-default book averages ~66), or > 8 new soft flags in a quarter | CRO | immediate | capacity review, temporary extra analyst hours |

Escalate to Level 2 when ≥ 2 SHAP drivers are confirmed against the statements, or any soft signal is a rating downgrade to sub-investment grade, auditor resignation, or payment delay. Level 3 for exposure above the committee threshold or if Level 2 cannot rule out manipulation.

## 5. Feedback and monitoring
Tag each reviewed name's outcome. Re-calibrate (prior-shift) every quarter with the bank's own realised default rate; retrain yearly. Outcome tags are never used as features in the same quarter.

## 6. What we do not do
Use post-event prices, ratings or news; exceed 40 priority names; auto-change limits; claim Indian recall without a populated, leakage-checked panel.
