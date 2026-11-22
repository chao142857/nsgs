#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PYTHON="${SGS_PYTHON:-python3}"
STAGE="${1:?use build, cpu, physics, pilot, evaluate or render}"; shift
case "$STAGE" in
    build) exec "$PYTHON" src/course.py . ;;
    cpu) exec "$PYTHON" cpu_proof.py "$@" ;;
    physics) exec "$PYTHON" -u src/physics_accept.py "$@" ;;
    pilot) exec "$PYTHON" -u src/train.py "$@" ;;
    evaluate) exec "$PYTHON" -u src/strict_eval.py "$@" ;;
    render)
        eval_dir="${1:?evaluation directory}"; out_dir="${2:?movie output directory}"
        mkdir -p "$out_dir"
        "$PYTHON" render/prepare.py "$eval_dir" "$out_dir/prepared"
        swiftc -O render/replay.swift -o "$out_dir/replay"
        exec "$out_dir/replay" "$eval_dir/visuals" "$out_dir/prepared" "$out_dir"
        ;;
    *) exit 2 ;;
esac
