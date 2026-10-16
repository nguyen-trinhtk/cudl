#!/usr/bin/env bash
# Profiling wrappers. Reports go to $CUDL_PROFILE_DIR (default: profiles/).
#
#   scripts/profile.sh nsys <cmd...>             # timeline with CUDA + NVTX ranges
#   scripts/profile.sh ncu <kernel-regex> <cmd...>  # full metrics for one kernel
#
# Examples:
#   scripts/profile.sh nsys build/bench/bench_copy --iters 20
#   scripts/profile.sh ncu copy_vec4 build/bench/bench_copy --warmup 0 --iters 1
set -euo pipefail

usage() {
    sed -n '2,10p' "$0" | sed 's/^# \{0,1\}//'
    exit "${1:-2}"
}

[[ $# -ge 2 ]] || usage
mode=$1
shift

out_dir=${CUDL_PROFILE_DIR:-profiles}
mkdir -p "$out_dir"
stamp=$(date +%Y%m%d-%H%M%S)

case "$mode" in
    nsys)
        nsys profile --trace=cuda,nvtx --stats=true --force-overwrite=true \
            -o "$out_dir/nsys-$stamp" "$@"
        ;;
    ncu)
        [[ $# -ge 2 ]] || usage
        kernel=$1
        shift
        # Skip warmup launches; profile NCU_LAUNCH_COUNT launches of the matching kernel.
        ncu --set full --import-source yes \
            --kernel-name "regex:$kernel" \
            --launch-skip "${NCU_LAUNCH_SKIP:-0}" \
            --launch-count "${NCU_LAUNCH_COUNT:-1}" \
            -o "$out_dir/ncu-$kernel-$stamp" "$@"
        ;;
    -h|--help) usage 0 ;;
    *) usage ;;
esac
