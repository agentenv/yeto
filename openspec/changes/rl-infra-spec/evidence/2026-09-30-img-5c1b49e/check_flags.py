import argparse, subprocess, miles.utils.arguments as A
print("miles from", A.__file__)
print("git HEAD", subprocess.check_output(["git","--git-dir=/tmp/img-verify-5c1b/root/miles/.git","rev-parse","HEAD"],text=True).strip())
def parse(argv):
    p = argparse.ArgumentParser(allow_abbrev=False); A.get_miles_extra_args_provider()(p)
    ns, unk = p.parse_known_args(argv); assert unk == [], unk; return ns
ns = parse([])
print("defaults", ns.policy_loss_variant, ns.sapo_tau_pos, ns.sapo_tau_neg, ns.gmpo_log_clip_low, ns.gmpo_log_clip_high)
for argv in (["--policy-loss-variant","cispo"],
             ["--policy-loss-variant","sapo","--sapo-tau-pos","0.9","--sapo-tau-neg","1.2"],
             ["--policy-loss-variant","gmpo","--gmpo-log-clip-low","0.3","--gmpo-log-clip-high","0.5"]):
    ns = parse(argv)
    A.validate_policy_loss_variant_args(ns)
    print("OK", argv, ns.policy_loss_variant, ns.sapo_tau_pos, ns.sapo_tau_neg, ns.gmpo_log_clip_low, ns.gmpo_log_clip_high)
for argv in (["--policy-loss-variant","gmpo","--calculate-per-token-loss"], ["--policy-loss-variant","sapo","--sapo-tau-pos","0"]):
    ns = parse(argv)
    try: A.validate_policy_loss_variant_args(ns); print("UNEXPECTED ACCEPT", argv)
    except AssertionError as e: print("rejected as expected", argv, "->", e)
try: parse(["--policy-loss-variant","bogus"]); print("UNEXPECTED")
except SystemExit: print("argparse rejects bogus variant")
