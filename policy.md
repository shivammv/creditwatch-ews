# CreditWatch: Operating Policy (corporate credit desk)

*All numbers below come from `artifacts/universe_sim.json` (1,000 simulated 800-borrower books per default rate, drawn from the untouched hold-out of the 4-quarter-lead model). Re-run `python run_all.py` to refresh. The 8-quarter figures are in `artifacts/universe_sim_8q.json`.*

## 1. Capacity and alert threshold
| Item | Setting |
|---|---|
| Universe / capacity | 800 borrowers, **40 deep reviews per quarter (5%)** |
| Cost of errors | missed default = **20 ×** a false alarm → expected cost = **20·FN + 1·FP** |
| Cost-optimal cut-off | Bayes threshold on calibrated probs: `C_FP/(C_FP+C_FN) = 1/21 ≈ 4.76%`. After prior-shift to the bank's own default rate, flag if p ≥ 4.76%. Hard 40-name cap still applies. |
| Hard cap | **never more than 40 "priority review" names**; names above the cut-off but outside the 40 go to the overflow list (section 4) |

## 2. What the cap costs (expected cost per quarter, 800 borrowers, 4-quarter-lead model)
| Book default rate | Defaulters | Recall inside 40 (95% range) | Cost @ 40 names | Cost-optimal list size (cost) | Flag nobody | Names above 1/21 |
|---|---|---|---|---|---|---|
| 1.0% | 8 | 56% (25-88%) | 105.7 | 37 (104.2) | 160 | ~39 |
| 2.0% | 16 | 55% (31-75%) | 175.6 | 43 (174.8) | 320 | ~80 |
| 4.1% | 33 | 48% (39-58%) | 367.7 | 149 (241.1) | 660 | ~161 |

**Reading it:** for a normal book (1-2% default rate) the 40-name budget is essentially the cost-optimal list size (the cap costs 0.8 to 1.5 cost units, under 1.5%), so the 5% constraint does not bind. In a stressed book (4.1%, CRISIL's 2017 all-issuer one-year default rate) the optimum is ~149 names; the 40-name cap then costs ~52% more than optimal, though it is still ~44% cheaper than flagging nobody. When more than 40 names exceed the cut-off, that is itself a **portfolio-stress trigger** (section 4).

**Indian evidence (pre-registered test, 10 defaulters):** the ratio model did not transfer to standalone Indian statements (2/10 caught at 4Q, 0/10 at 8Q). Until an India-trained model is validated, the model score is advisory, and triage leans on the hard triggers (rating downgrade to sub-investment grade, payment delay, auditor resignation) and on the debt-service lines in each memo (interest cover, cash earnings / liabilities).

## 3. Quarterly ranking method
1. Build features from the **latest financial statements public on the run date** (fiscal year counts only after its filing/results date).
2. Model: calibrated one-year probability of distress from the lead-matched, median-imputed LightGBM on 39 India-portable ratios (`model_production.joblib`); the 8-quarter model (`model_production_8q.joblib`) is shown alongside it for longer lead.
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
| **Stress trigger**: > 100 names above cut-off (the 2%-default book averages ~80), or > 8 new soft flags in a quarter | CRO | immediate | capacity review, temporary extra analyst hours |

Escalate to Level 2 when ≥ 2 SHAP drivers are confirmed against the statements, or any soft signal is a rating downgrade to sub-investment grade, auditor resignation, or payment delay. Level 3 for exposure above the committee threshold or if Level 2 cannot rule out manipulation.

## 5. Feedback and monitoring
Tag each reviewed name's outcome. Re-calibrate (prior-shift) every quarter with the bank's own realised default rate; retrain yearly. Outcome tags are never used as features in the same quarter.

## 6. What we do not do
Use post-event prices, ratings or news; exceed 40 priority names; auto-change limits; claim Indian recall without a populated, leakage-checked panel.
