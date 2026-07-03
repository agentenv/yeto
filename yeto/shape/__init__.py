"""yeto shape: automatic compute planning for a fine-tuning run.

Turns (model, budget, constraints) into a concrete fleet plan — how many
learner islands of which instance type in which region — by maximizing
effective training FLOPs under cost, quota, and spot-placement-score
constraints. See `yeto shape --help`.
"""
