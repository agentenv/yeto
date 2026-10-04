"""Step 2 (miles-next-venv, PYTHONPATH=<image /root/miles>): Miles' own argument
provider + validate_policy_loss_variant_args on yeto's argv. Not the full
upstream parse_args/validate_args (Megatron half needs a CUDA container)."""
import argparse, json, subprocess, sys
import miles.utils.arguments as A
root = A.__file__.split("/miles/utils/")[0]
print("miles from", A.__file__)
print("git HEAD", subprocess.check_output(["git", f"--git-dir={root}/.git", "rev-parse", "HEAD"], text=True).strip())
cases = json.load(open(sys.argv[1]))
for name, case in cases.items():
    p = argparse.ArgumentParser(allow_abbrev=False)
    A.get_miles_extra_args_provider()(p)
    ns, unknown = p.parse_known_args(case["argv"])
    assert unknown == [], unknown
    try:
        A.validate_policy_loss_variant_args(ns)
        verdict = "accepted"
    except AssertionError as exc:
        verdict = f"rejected: {exc}"
    print(name, case["argv"], "->", ns.policy_loss_variant, ns.sapo_tau_pos, ns.sapo_tau_neg,
          ns.gmpo_log_clip_low, ns.gmpo_log_clip_high, "calculate_per_token_loss",
          ns.calculate_per_token_loss, "eps", ns.eps_clip, ns.eps_clip_high, "|", verdict,
          "| yeto:", "rejected" if case["yeto_rejections"] else "accepted")
