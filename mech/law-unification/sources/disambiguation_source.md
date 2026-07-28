# EXPLORATORY

Fixed-T=5 outer-horizon disambiguation lane. These results are exploratory.

Updated UTC: 2026-07-26T14:38:13.111999+00:00
Progress: 72/72 valid cells; 0 invalid; 0 pending.

Block math: H1024/S5120 = 5 windows/learner, 131072 tokens/window, 655360 tokens/learner, 2621440 tokens total; H2048/S10240 = 5 windows/learner, 262144 tokens/window, 1310720 tokens/learner, 5242880 tokens total. Both use 20 syncer updates (= 4 learners x T5).
Exact packed-capacity minimum over seeds 401/409/419 is 26527 complete blocks (seed 419 learner 3), leaving 16287 blocks of slack at S10240.

## Curves

CURVE H1024_S5120_mu0: eta*=0.0443859 D=1 status=INTERIOR eta95=[0.0431009,0.0458699] D95=[1,1] bootstrap_valid=1000/1000
CURVE H1024_S5120_corr: eta*=0.00359466 D=0.809865 status=INTERIOR eta95=[0.00353272,0.00366846] D95=[0.799752,0.832495] bootstrap_valid=1000/1000
CURVE H1024_S5120_raw: eta*=0.0113326 D=2.5532 status=INTERIOR eta95=[0.0109724,0.0116429] D95=[2.46784,2.7013] bootstrap_valid=1000/1000
CURVE H2048_S10240_mu0: eta*=0.051691 D=1 status=INTERIOR eta95=[0.0514319,0.0519336] D95=[1,1] bootstrap_valid=1000/1000
CURVE H2048_S10240_corr: eta*=0.00386349 D=0.747421 status=INTERIOR eta95=[0.00382885,0.0038926] D95=[0.740221,0.752297] bootstrap_valid=1000/1000
CURVE H2048_S10240_raw: eta*=0.0122755 D=2.37478 status=INTERIOR eta95=[0.0121686,0.0124528] D95=[2.35935,2.39782] bootstrap_valid=1000/1000

## Hypothesis Comparison

Existing H512_S2560 corrected anchor: D=0.861517, paired-bootstrap 95% CI [0.843686, 0.883431] (read-only v3 analysis).
Existing H512_S2560 raw anchor: D=2.566822, paired-bootstrap 95% CI [2.507529, 2.654581] (read-only v3 analysis).
Corrected observations: S5120 D=0.809865 vs age=0.86, duration=0.75; S10240 D=0.747421 vs age=0.86, duration=0.53.
Log-scale RMSE: age=0.107919; duration=0.249061.
Raw observations: S5120 D=2.5532; S10240 D=2.37478; inside 2.13-2.57 band=[True, True].
EDGE CHECK: no completed curve currently requests an outward rung.

DISAMBIG VERDICT: MIXED details=Dcorr falls 0.8615->0.8099->0.7474, but much more slowly than duration predictions 0.75/0.53; age log-RMSE 0.108 vs duration 0.249
