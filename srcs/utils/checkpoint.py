from pathlib import Path

def resolve_checkpoint(ckpt=None, output_root="./Output"):
    if ckpt:
        return ckpt

    candidates = sorted(
        Path(output_root).glob("output_*/weight/*.pt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError("No checkpoint found under Output/output_*/weight/")
    return str(candidates[0])
