import argparse
import shlex
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


MODE_DEFAULTS = {
    "custom": {
        "cfg": "cfg/training/train.yaml",
        "hyp": "data/hyp.scratch.custom.yaml",
        "weights": "",
        "batch_size": 8,
        "train_name": "custom_m4sfwd",
        "eval_name": "custom_m4sfwd_eval",
        "infer_name": "custom_m4sfwd_infer",
    },
    "yolov7-tiny": {
        "cfg": "cfg/training/yolov7-tiny.yaml",
        "hyp": "data/hyp.scratch.tiny.yaml",
        "weights": "yolov7-tiny.pt",
        "batch_size": 16,
        "train_name": "yolov7_tiny_m4sfwd",
        "eval_name": "yolov7_tiny_m4sfwd_eval",
        "infer_name": "yolov7_tiny_m4sfwd_infer",
    },
}


def shell_join(parts):
    return " ".join(shlex.quote(str(part)) for part in parts)


def resolve_path(path_str):
    path = Path(path_str)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def default_weight_path(mode):
    train_name = MODE_DEFAULTS[mode]["train_name"]
    train_root = PROJECT_ROOT / "runs" / "train"
    candidates = []

    for run_dir in train_root.glob(f"{train_name}*"):
        best_pt = run_dir / "weights" / "best.pt"
        if run_dir.is_dir() and best_pt.exists():
            candidates.append(best_pt)

    if not candidates:
        return train_root / train_name / "weights" / "best.pt"

    return max(candidates, key=lambda p: p.stat().st_mtime)


def build_common_train_args(args, mode_cfg):
    weights = args.weights if args.weights is not None else mode_cfg["weights"]
    batch_size = args.batch_size if args.batch_size is not None else mode_cfg["batch_size"]
    run_name = args.name or mode_cfg["train_name"]
    return [
        "--weights", weights,
        "--cfg", args.cfg or mode_cfg["cfg"],
        "--data", args.data,
        "--hyp", args.hyp or mode_cfg["hyp"],
        "--epochs", str(args.epochs),
        "--batch-size", str(batch_size),
        "--img-size", str(args.img_size), str(args.img_size),
        "--device", args.device,
        "--project", args.project,
        "--name", run_name,
        "--workers", str(args.workers),
    ] + (["--exist-ok"] if args.exist_ok else [])


def build_train_command(args):
    mode_cfg = MODE_DEFAULTS[args.mode]
    cmd = [sys.executable, "train.py"]
    cmd.extend(build_common_train_args(args, mode_cfg))
    if args.limit_train_images:
        cmd.extend(["--limit-train-images", str(args.limit_train_images)])
    return cmd + args.extra


def build_eval_command(args):
    mode_cfg = MODE_DEFAULTS[args.mode]
    weights = args.weights or str(default_weight_path(args.mode))
    run_name = args.name or mode_cfg["eval_name"]
    cmd = [
        sys.executable, "test.py",
        "--weights", weights,
        "--data", args.data,
        "--batch-size", str(args.batch_size),
        "--img-size", str(args.img_size),
        "--device", args.device,
        "--task", args.task,
        "--project", args.project,
        "--name", run_name,
        "--conf-thres", str(args.conf_thres),
        "--iou-thres", str(args.iou_thres),
    ]
    if args.exist_ok:
        cmd.append("--exist-ok")
    if args.verbose:
        cmd.append("--verbose")
    return cmd + args.extra


def build_infer_command(args):
    mode_cfg = MODE_DEFAULTS[args.mode]
    weights = args.weights or str(default_weight_path(args.mode))
    run_name = args.name or mode_cfg["infer_name"]
    cmd = [
        sys.executable, "detect.py",
        "--weights", weights,
        "--source", args.source,
        "--img-size", str(args.img_size),
        "--device", args.device,
        "--project", args.project,
        "--name", run_name,
        "--conf-thres", str(args.conf_thres),
        "--iou-thres", str(args.iou_thres),
    ]
    if args.exist_ok:
        cmd.append("--exist-ok")
    if args.save_txt:
        cmd.append("--save-txt")
    return cmd + args.extra


def run_command(cmd):
    print(f"[run_experiment] cwd={PROJECT_ROOT}")
    print(f"[run_experiment] cmd={shell_join(cmd)}")
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


def create_parser():
    parser = argparse.ArgumentParser(
        description="Unified train/eval/infer launcher for ForestUAV custom model and YOLOv7-tiny."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train a model")
    add_mode_args(train_parser)
    train_parser.add_argument("--cfg", default=None, help="Override model cfg path")
    train_parser.add_argument("--hyp", default=None, help="Override hyperparameter yaml path")
    train_parser.add_argument("--weights", default=None, help="Override initial weights path")
    train_parser.add_argument("--data", default="data/m4sfwd.yaml", help="Dataset yaml path")
    train_parser.add_argument("--epochs", type=int, default=300)
    train_parser.add_argument("--batch-size", type=int, default=None, help="Override default batch size")
    train_parser.add_argument("--img-size", type=int, default=640)
    train_parser.add_argument("--device", default="0")
    train_parser.add_argument("--workers", type=int, default=8)
    train_parser.add_argument("--project", default="runs/train")
    train_parser.add_argument("--name", default=None)
    train_parser.add_argument("--limit-train-images", type=int, default=0)
    train_parser.add_argument("--exist-ok", action="store_true")
    train_parser.add_argument("--dry-run", action="store_true", help="Print the resolved command without running it")
    train_parser.add_argument("extra", nargs=argparse.REMAINDER, help="Extra args passed to train.py")

    eval_parser = subparsers.add_parser("eval", help="Evaluate a trained model on train/val/test split")
    add_mode_args(eval_parser)
    eval_parser.add_argument("--weights", default=None, help="Weights path, default is the mode's best.pt")
    eval_parser.add_argument("--data", default="data/m4sfwd.yaml", help="Dataset yaml path")
    eval_parser.add_argument("--batch-size", type=int, default=16)
    eval_parser.add_argument("--img-size", type=int, default=640)
    eval_parser.add_argument("--device", default="0")
    eval_parser.add_argument("--task", choices=["train", "val", "test"], default="test")
    eval_parser.add_argument("--project", default="runs/test")
    eval_parser.add_argument("--name", default=None)
    eval_parser.add_argument("--conf-thres", type=float, default=0.001)
    eval_parser.add_argument("--iou-thres", type=float, default=0.65)
    eval_parser.add_argument("--verbose", action="store_true")
    eval_parser.add_argument("--exist-ok", action="store_true")
    eval_parser.add_argument("--dry-run", action="store_true", help="Print the resolved command without running it")
    eval_parser.add_argument("extra", nargs=argparse.REMAINDER, help="Extra args passed to test.py")

    infer_parser = subparsers.add_parser("infer", help="Run inference on images/videos/folders")
    add_mode_args(infer_parser)
    infer_parser.add_argument("--weights", default=None, help="Weights path, default is the mode's best.pt")
    infer_parser.add_argument("--source", default="datasets/M4SFWD/images/test", help="Inference source")
    infer_parser.add_argument("--img-size", type=int, default=640)
    infer_parser.add_argument("--device", default="0")
    infer_parser.add_argument("--project", default="runs/detect")
    infer_parser.add_argument("--name", default=None)
    infer_parser.add_argument("--conf-thres", type=float, default=0.25)
    infer_parser.add_argument("--iou-thres", type=float, default=0.45)
    infer_parser.add_argument("--save-txt", action="store_true")
    infer_parser.add_argument("--exist-ok", action="store_true")
    infer_parser.add_argument("--dry-run", action="store_true", help="Print the resolved command without running it")
    infer_parser.add_argument("extra", nargs=argparse.REMAINDER, help="Extra args passed to detect.py")

    return parser


def add_mode_args(parser):
    parser.add_argument(
        "--mode",
        choices=sorted(MODE_DEFAULTS.keys()),
        required=True,
        help="custom uses cfg/training/train.yaml; yolov7-tiny uses cfg/training/yolov7-tiny.yaml",
    )


def validate_inputs(args):
    data_path = resolve_path(args.data) if hasattr(args, "data") else None
    if data_path and not data_path.exists():
        raise FileNotFoundError(f"Dataset yaml not found: {data_path}")

    if getattr(args, "cfg", None):
        cfg_path = resolve_path(args.cfg)
        if not cfg_path.exists():
            raise FileNotFoundError(f"Model cfg not found: {cfg_path}")

    if getattr(args, "hyp", None):
        hyp_path = resolve_path(args.hyp)
        if not hyp_path.exists():
            raise FileNotFoundError(f"Hyperparameter yaml not found: {hyp_path}")

    if getattr(args, "weights", None):
        weight_value = args.weights
        if weight_value and Path(weight_value).suffix == ".pt":
            weight_path = resolve_path(weight_value)
            if not weight_path.exists():
                print(f"[run_experiment] warning: weights not found yet: {weight_path}")
    elif args.command in {"eval", "infer"}:
        resolved_weight = default_weight_path(args.mode)
        print(f"[run_experiment] default weights: {resolved_weight}")
        if not resolved_weight.exists():
            raise FileNotFoundError(
                f"No trained weights were found for mode '{args.mode}'. "
                f"Expected something like: {resolved_weight}"
            )


def main():
    parser = create_parser()
    args = parser.parse_args()
    validate_inputs(args)

    if args.command == "train":
        cmd = build_train_command(args)
    elif args.command == "eval":
        cmd = build_eval_command(args)
    else:
        cmd = build_infer_command(args)

    if args.dry_run:
        print(f"[run_experiment] cwd={PROJECT_ROOT}")
        print(f"[run_experiment] cmd={shell_join(cmd)}")
        return

    run_command(cmd)


if __name__ == "__main__":
    main()
