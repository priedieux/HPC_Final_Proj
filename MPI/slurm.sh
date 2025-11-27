#!/bin/bash
#SBATCH --job-name=image_mpi
#SBATCH --nodes=4
#SBATCH --ntasks=16
#SBATCH --time=00:30:00
#SBATCH --output=mpi_results_%j.txt

module load openmpi

echo "=========================================="
echo "MPI Image Processing Benchmark"
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_JOB_NUM_NODES"
echo "=========================================="

# Generate test images if needed
if [ ! -f "test_gradient_large.ppm" ]; then
    python3 gen_img.py
fi

# Test with different process counts
for nprocs in 1 2 4 8 16; do
    echo ""
    echo "=== Testing with $nprocs processes ==="
    
    # Grayscale filter
    echo "Filter: grayscale"
    mpirun -np $nprocs ./image_proc_mpi.exe test_gradient_large.ppm \
        output_gray_${nprocs}p.ppm grayscale

    # Blur filter (more complex with halo exchange)
    echo "Filter: blur"
    mpirun -np $nprocs ./image_proc_mpi.exe test_gradient_large.ppm \
        output_blur_${nprocs}p.ppm blur

    # Edge detection filter (Sobel with halo exchange)
    echo "Filter: edge"
    mpirun -np $nprocs ./image_proc_mpi.exe test_gradient_large.ppm \
        output_edge_${nprocs}p.ppm edge

    # Brightness filter
    echo "Filter: brighten"
    mpirun -np $nprocs ./image_proc_mpi.exe test_gradient_large.ppm \
        output_bright_${nprocs}p.ppm brighten
done

echo ""
echo "=========================================="
echo "Benchmark Complete!"
echo "=========================================="