# Trigger results (launcher entry, Modal H100! x1 each, pre-declared criteria)

| run | validity | EFFECT (per logged step) | verdict |
| --- | --- | --- | --- |
| tis [0.99,1.01] | all pass | tis_clipfrac 0.192 / 0.211 / 0.265 | PASS: truncation takes effect |
| icepop [0.99,1.01] | all pass | tis_clipfrac (masked) 0.192 / 0.225 / 0.267 | PASS: masking takes effect |
| mis-mask token [0.99,1.01] | all pass | mask_fraction_low+high 0.192 / 0.225 / 0.267 | PASS: masking takes effect |
| opsm-trainer delta 1e-6, 2 opt steps | all pass | opsm_clipfrac 0 / 0 / 0 / 0.1875 / 0 / 0.25 | PASS: OPSM masks sequences on the 2nd step of rounds 2 and 3 |

No zero-gradient or other invariant failure occurred. Windows (UTC): 18:50:42 to 19:23:29, 4 runs,
~33 H100 min, ~ $2.2 (estimate). Each app_after_stop.txt is empty and `modal container list`
shows no algo1a container afterwards. The thresholds are test settings, not recommendations.
