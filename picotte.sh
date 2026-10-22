# 1. Build (login node)
module load gcc/9.2.0 cmake/3.24.3 cuda12.3/toolkit/12.3.2
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=70
cmake --build build -j

# 2. Test, then bench only if the tests pass
jid=$(sbatch --parsable scripts/slurm/test.sbatch)
sbatch --dependency=afterok:$jid scripts/slurm/bench.sbatch
