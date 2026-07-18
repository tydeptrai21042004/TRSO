#!/usr/bin/env python
"""
continual_multihd.py

One-command continual PEFT evaluation for TWO tasks:
- Train Task A (save ckpt_A)
- Eval Task A (Acc_A_before)
- Train Task B starting from ckpt_A backbone/adapters (save ckpt_B)
- Eval Task B (Acc_B)
- Eval Task A again using ckpt_B backbone/adapters and (optionally) Task-A head from ckpt_A (Acc_A_after)
- Print forgetting: F_A = Acc_A_before - Acc_A_after

Supports:
- Same label space / domain shift (nb_classes A == nb_classes B): single-head is enough
- Different label spaces (nb_classes differ): multi-head eval by swapping head from ckpt_A via --head_from

This script calls main.py as a subprocess, so it works with your existing training code.
"""

import argparse
import os
import re
import shlex
import subprocess
from pathlib import Path
from typing import Optional, Tuple


ACC_PATTERNS = [
    re.compile(r"Accuracy of the network on .*?:\s*([0-9]+\.?[0-9]*)%"),   # common in DeiT-style repos
    re.compile(r"\bacc1\b[^0-9]*([0-9]+\.?[0-9]*)"),                     # fallback if logs include acc1=...
    re.compile(r"\bAcc\b[^0-9]*([0-9]+\.?[0-9]*)%"),                     # generic
]


def _run(cmd, cwd=None, env=None) -> str:
    print("\n[CMD]", " ".join(cmd))
    p = subprocess.run(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    out = p.stdout
    print(out)
    if p.returncode != 0:
        raise SystemExit(f"Command failed with exit code {p.returncode}")
    return out


def _parse_nb_classes(cmd_tokens) -> Optional[int]:
    # parse --nb_classes 10
    for i, t in enumerate(cmd_tokens):
        if t == '--nb_classes' and i + 1 < len(cmd_tokens):
            try:
                return int(cmd_tokens[i + 1])
            except Exception:
                return None
    return None


def _parse_acc(stdout: str) -> Optional[float]:
    # Try multiple patterns, take the LAST match in the output.
    acc = None
    for pat in ACC_PATTERNS:
        for m in pat.finditer(stdout):
            try:
                acc = float(m.group(1))
            except Exception:
                pass
    return acc


def _find_checkpoint(out_dir: Path) -> Path:
    # Prefer common names; otherwise choose most recently modified .pth
    candidates = [
        out_dir / 'checkpoint-best.pth',
        out_dir / 'checkpoint_best.pth',
        out_dir / 'best.pth',
        out_dir / 'checkpoint.pth',
    ]
    for c in candidates:
        if c.exists():
            return c

    pths = list(out_dir.glob('*.pth'))
    if not pths:
        raise FileNotFoundError(f"No .pth checkpoint found in {out_dir}")
    pths.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return pths[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', type=str, default='.', help='Path to repo root (where main.py lives).')
    ap.add_argument('--python', type=str, default='python', help='Python executable.')
    ap.add_argument('--taskA', type=str, required=True,
                    help='Args string for Task A run, e.g. "--dataset cifar10 --data_path ./data/cifar10 --nb_classes 10 --backbone ... --tuning_method dt ..."')
    ap.add_argument('--taskB', type=str, required=True,
                    help='Args string for Task B run (same format as taskA).')
    ap.add_argument('--out', type=str, default='./continual_runs', help='Root output dir.')
    ap.add_argument('--tag', type=str, default='exp', help='Subfolder name under --out.')
    ap.add_argument('--extra_train', type=str, default='', help='Extra args appended to BOTH training runs.')
    ap.add_argument('--extra_eval', type=str, default='', help='Extra args appended to ALL eval runs.')
    ap.add_argument('--skip_trainA', action='store_true', help='Skip Task A training (assumes ckpt_A exists).')
    ap.add_argument('--skip_trainB', action='store_true', help='Skip Task B training (assumes ckpt_B exists).')
    args = ap.parse_args()

    repo = Path(args.repo).resolve()
    main_py = repo / 'main.py'
    if not main_py.exists():
        raise FileNotFoundError(f"main.py not found at {main_py}")

    root = Path(args.out).resolve() / args.tag
    root.mkdir(parents=True, exist_ok=True)

    # Tokenize argument strings
    taskA_tokens = shlex.split(args.taskA)
    taskB_tokens = shlex.split(args.taskB)
    extra_train = shlex.split(args.extra_train) if args.extra_train else []
    extra_eval = shlex.split(args.extra_eval) if args.extra_eval else []

    nbA = _parse_nb_classes(taskA_tokens)
    nbB = _parse_nb_classes(taskB_tokens)
    if nbA is None or nbB is None:
        print("[Warn] Could not parse --nb_classes from taskA/taskB. Multi-head auto-detection may be wrong.")
    multi_head = (nbA is not None and nbB is not None and nbA != nbB)

    outA = root / 'taskA_train'
    outB = root / 'taskB_train'
    outA.mkdir(exist_ok=True)
    outB.mkdir(exist_ok=True)

    ckptA = outA / 'checkpoint-best.pth'
    ckptB = outB / 'checkpoint-best.pth'

    # ---- Train Task A ----
    if not args.skip_trainA:
        cmdA_train = [args.python, str(main_py), *taskA_tokens, '--output_dir', str(outA), *extra_train]
        _run(cmdA_train, cwd=str(repo))
        ckptA = _find_checkpoint(outA)
    else:
        ckptA = _find_checkpoint(outA)
    print(f"[Info] ckpt_A = {ckptA}")

    # ---- Eval Task A (before) ----
    cmdA_eval_before = [args.python, str(main_py), *taskA_tokens,
                        '--eval', 'True',
                        '--output_dir', str(root / 'eval_A_before'),
                        '--finetune', str(ckptA),
                        *extra_eval]
    out_eval_A_before = _run(cmdA_eval_before, cwd=str(repo))
    acc_A_before = _parse_acc(out_eval_A_before)
    if acc_A_before is None:
        print("[Warn] Could not parse Acc_A_before from logs.")

    # ---- Train Task B from ckpt_A backbone/adapters ----
    if not args.skip_trainB:
        cmdB_train = [args.python, str(main_py), *taskB_tokens,
                      '--output_dir', str(outB),
                      '--finetune', str(ckptA),
                      *extra_train]
        _run(cmdB_train, cwd=str(repo))
        ckptB = _find_checkpoint(outB)
    else:
        ckptB = _find_checkpoint(outB)
    print(f"[Info] ckpt_B = {ckptB}")

    # ---- Eval Task B ----
    cmdB_eval = [args.python, str(main_py), *taskB_tokens,
                 '--eval', 'True',
                 '--output_dir', str(root / 'eval_B'),
                 '--finetune', str(ckptB),
                 *extra_eval]
    out_eval_B = _run(cmdB_eval, cwd=str(repo))
    acc_B = _parse_acc(out_eval_B)
    if acc_B is None:
        print("[Warn] Could not parse Acc_B from logs.")

    # ---- Eval Task A again after training B ----
    cmdA_eval_after = [args.python, str(main_py), *taskA_tokens,
                       '--eval', 'True',
                       '--output_dir', str(root / 'eval_A_after'),
                       '--finetune', str(ckptB)]
    if multi_head:
        # Swap-in Task-A head from ckpt_A (multi-head evaluation)
        cmdA_eval_after += ['--head_from', str(ckptA)]
    cmdA_eval_after += extra_eval
    out_eval_A_after = _run(cmdA_eval_after, cwd=str(repo))
    acc_A_after = _parse_acc(out_eval_A_after)
    if acc_A_after is None:
        print("[Warn] Could not parse Acc_A_after from logs.")

    # ---- Summary ----
    print("\n===== Continual Summary =====")
    print(f"Setting: {'multi-head (different label space)' if multi_head else 'single-head (same label space/domain shift)'}")
    print(f"Acc_A_before: {acc_A_before}")
    print(f"Acc_B:        {acc_B}")
    print(f"Acc_A_after:  {acc_A_after}")
    if (acc_A_before is not None) and (acc_A_after is not None):
        print(f"Forgetting F_A = Acc_A_before - Acc_A_after = {acc_A_before - acc_A_after:.6f}")
    print("============================\n")


if __name__ == '__main__':
    main()
