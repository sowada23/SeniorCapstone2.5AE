from pathlib import Path


def make_run_dir(base_dir="./Output", prefix="output_"):
    base = Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)

    existing = []
    for path in base.iterdir():
        if path.is_dir() and path.name.startswith(prefix):
            suffix = path.name[len(prefix):]
            if suffix.isdigit():
                existing.append(int(suffix))

    next_id = 1 if not existing else max(existing) + 1
    run_name = f"{prefix}{next_id:03d}"

    run_dir = base / run_name
    train_dir = run_dir / "train"
    test_dir = run_dir / "test"
    weight_dir = run_dir / "weight"

    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    return {
        "run_id": next_id,
        "run_name": run_name,
        "run_dir": str(run_dir),
        "train_dir": str(train_dir),
        "test_dir": str(test_dir),
        "weight_dir": str(weight_dir)
    }
