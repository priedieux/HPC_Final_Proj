#!/bin/bash
# benchmark_mpi.sh - Run comprehensive benchmarks for MPI image processing

echo "======================================"
echo "Image Processing MPI Benchmark"
echo "======================================"

# Create results directory
mkdir -p results

# Filters to test
filters=("grayscale" "blur" "edge" "brighten")

# Process counts to test
processes=(1 2 4 8 16)

# Image sizes
declare -A images
images["small"]="test_gradient_small.ppm"
images["medium"]="test_gradient_medium.ppm"
images["large"]="test_gradient_large.ppm"

# Generate test images if they don't exist
if [ ! -f "test_gradient_small.ppm" ]; then
    echo "Generating test images..."
    python3 gen_img.py
fi

# Results file
results_file="results/benchmark_mpi_results.txt"
echo "MPI Benchmark Results - $(date)" > $results_file
echo "======================================" >> $results_file

# Run benchmarks
for size_name in "${!images[@]}"; do
    input_file="${images[$size_name]}"
    
    if [ ! -f "$input_file" ]; then
        echo "Warning: $input_file not found, skipping..."
        continue
    fi
    
    echo ""
    echo "Testing with $size_name image: $input_file"
    echo "" >> $results_file
    echo "Image Size: $size_name ($input_file)" >> $results_file
    echo "--------------------------------------" >> $results_file
    
    for filter in "${filters[@]}"; do
        echo ""
        echo "  Filter: $filter"
        echo "" >> $results_file
        echo "Filter: $filter" >> $results_file
        
        for np in "${processes[@]}"; do
            output_file="results/output_${size_name}_${filter}_${np}p.ppm"
            
            echo -n "    $np processes... "

            # Run with mpirun and capture output
            # Add --oversubscribe to allow more processes than physical cores
            output=$(mpirun --oversubscribe -np $np ./image_proc_mpi.exe "$input_file" "$output_file" "$filter" 2>&1)
            exit_code=$?

            if [ $exit_code -ne 0 ]; then
                echo "FAILED (exit code: $exit_code)"
                echo "  $np processes: FAILED" >> $results_file
            else
                # Extract timing (only from rank 0 output)
                time=$(echo "$output" | grep "Processing time" | head -1 | awk '{print $3}')

                if [ -z "$time" ]; then
                    echo "FAILED"
                    echo "  $np processes: FAILED" >> $results_file
                else
                    echo "$time seconds"
                    echo "  $np processes: $time seconds" >> $results_file
                fi
            fi
        done
        echo "" >> $results_file
    done
done

echo ""
echo "======================================"
echo "Benchmark complete!"
echo "Results saved to: $results_file"
echo "Output images saved to: results/"
echo "======================================"

# Calculate and display speedup
echo ""
echo "Calculating speedup metrics..."
python3 - << 'EOF'
import re

with open('results/benchmark_mpi_results.txt', 'r') as f:
    content = f.read()

# Prepare output
output_lines = []
output_lines.append("\n" + "="*70)
output_lines.append("SPEEDUP ANALYSIS")
output_lines.append("="*70)

# Parse results by image size
sections = content.split('\n')

image_data = {}
current_size = None
current_filter = None

for line in sections:
    # Detect image size
    if line.startswith('Image Size:'):
        current_size = line.split('Image Size:')[1].strip().split('(')[0].strip()
        if current_size not in image_data:
            image_data[current_size] = {}

    # Detect filter
    if line.startswith('Filter:'):
        current_filter = line.split('Filter:')[1].strip()
        if current_size and current_filter:
            if current_filter not in image_data[current_size]:
                image_data[current_size][current_filter] = {}

    # Extract timing data
    match = re.search(r'(\d+) processes: ([\d.]+) seconds', line)
    if match and current_size and current_filter:
        procs = int(match.group(1))
        time = float(match.group(2))
        image_data[current_size][current_filter][procs] = time

# Generate analysis for each image size
for size_name in sorted(image_data.keys()):
    output_lines.append(f"\n{'='*70}")
    output_lines.append(f"IMAGE SIZE: {size_name.upper()}")
    output_lines.append('='*70)

    for filter_name in sorted(image_data[size_name].keys()):
        times = image_data[size_name][filter_name]

        if 1 in times and len(times) > 1:
            baseline = times[1]
            output_lines.append(f"\nFilter: {filter_name}")
            output_lines.append(f"  Procs  | Time (s) | Speedup | Efficiency")
            output_lines.append(f"  -------|----------|---------|------------")
            for p in sorted(times.keys()):
                speedup = baseline / times[p]
                efficiency = (speedup / p) * 100
                output_lines.append(f"  {p:5d}  | {times[p]:8.4f} | {speedup:7.2f} | {efficiency:9.1f}%")

output_lines.append("\n" + "="*70)

# Print to console
for line in output_lines:
    print(line)

# Append to results file
with open('results/benchmark_mpi_results.txt', 'a') as f:
    f.write('\n\n')
    for line in output_lines:
        f.write(line + '\n')

print("\nSpeedup metrics added to results/benchmark_mpi_results.txt")
EOF

echo ""
echo "Done!"