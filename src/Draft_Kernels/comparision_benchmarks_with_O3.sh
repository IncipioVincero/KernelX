#!/bin/bash

# Image dimensions and channels (control variables)
HEIGHT=1024
WIDTH=1024
CHANNELS=3

# List of programs to benchmark
PROGRAMS=("conv_with_time" "conv_inline_ptx_time")

# CSV output
OUTPUT_CSV="benchmark_results_with_O3.csv"
echo "Program,Time(ms),GFLOPs,OperationalIntensity" > $OUTPUT_CSV

# Calculate FLOPs and Bytes once (same for all programs if args fixed)
OPS_PER_ELEM=$((5 * 5 * 2)) # 5x5 mask, each with 1 mul + 1 add
TOTAL_ELEMENTS=$((HEIGHT * WIDTH * CHANNELS))
FLOPS=$(echo "$OPS_PER_ELEM * $TOTAL_ELEMENTS" | bc)

BYTES=$(echo "$TOTAL_ELEMENTS * 4 * 2 + 25 * 4" | bc) # input+output + mask once
OI=$(echo "scale=4; $FLOPS / $BYTES" | bc)

for prog in "${PROGRAMS[@]}"; do
    # Compile
    nvcc -O3 -o $prog $prog.cu

    # Run and extract time
    TIME_MS=$(./$prog $HEIGHT $WIDTH $CHANNELS | grep "Kernel execution time" | awk '{print $4}')
    
    # Compute GFLOPs
    GFLOPs=$(echo "scale=2; $FLOPS / ($TIME_MS * 1e6)" | bc)

    # Save to CSV
    echo "$prog,$TIME_MS,$GFLOPs,$OI" >> $OUTPUT_CSV
done

