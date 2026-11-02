import re
def _num(s):
    s = str(s).replace(",", "")
    m = re.findall(r"-?\d+(?:\.\d+)?", s)
    return float(m[-1]) if m else None
async def score(args, sample, **kwargs):
    label = sample.label
    gold = _num(str(label).split("####")[-1]) if label is not None else None
    text = sample.response or ""
    boxed = re.findall(r"\\boxed\{([^}]*)\}", text)
    pred = _num(boxed[-1]) if boxed else _num(text[-200:])
    return 1.0 if (gold is not None and pred is not None and abs(pred - gold) < 1e-6) else 0.0
