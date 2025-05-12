#!/bin/bash

# =======================
# CUDA Compilation Script
# =======================

#Description: This script is a refined version of the output from the following command: nvcc convolution.cu -arch=sm_90 -o convolution_dry --dryrun
#NOTE: This template enables users to view steps used in the NVCC compiler utility.

#DISCLAIMER: cudafe++, ptxas, nvvm, cicc, fatbinary, nvlink and all variables in this script (except from SOURCE_FILE and SOURCE_PATH) are proprietary tools from NVIDIA. Consult NVIDIA's documentation at: https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/


# Environment Setup
_NVVM_BRANCH_=nvvm
_SPACE_=
_CUDART_=cudart
_HERE_=/usr/local/cuda-12.3/bin
_THERE_=/usr/local/cuda-12.3/bin
_TARGET_SIZE_=
_TARGET_DIR_=targets/x86_64-linux
TOP=/usr/local/cuda-12.3
CUDA_VER=
NVVMIR_LIBRARY_DIR="$TOP/nvvm/libdevice"

# Exported Paths
export LD_LIBRARY_PATH="$TOP/targets/x86_64-linux/lib:$LD_LIBRARY_PATH"
export PATH="$TOP/nvvm/bin:$TOP/bin:$PATH"

INCLUDES="-I$TOP/targets/x86_64-linux/include"
# LIBRARIES="-L$TOP/targets/x86_64-linux/lib/stubs -L$TOP/targets/x86_64-linux/lib"

CUDAFE_FLAGS=
PTXAS_FLAGS=

# Replace with actual filenames
SOURCE_FILE=<insert_source_file_name>
SOURCE_PATH=<insert_source_path>

# ========
# Step 1: Preprocess with GCC
# ========
gcc -D__CUDA_ARCH_LIST__=900 -E -x c++ -D__CUDACC__ -D__NVCC__ \
    "$INCLUDES" \
    -D__CUDACC_VER_MAJOR__=12 -D__CUDACC_VER_MINOR__=3 -D__CUDACC_VER_BUILD__=107 \
    -D__CUDA_API_VER_MAJOR__=12 -D__CUDA_API_VER_MINOR__=3 \
    -D__NVCC_DIAG_PRAGMA_SUPPORT__=1 \
    -include "$TOP/targets/x86_64-linux/include/cuda_runtime.h" \
    -m64 "$SOURCE_PATH" \
    -o "/tmp/tmpxft_$$-5_${SOURCE_FILE}.cpp4.ii"

# ========
# Step 2: cudafe++
# ========
cudafe++ --c++17 --gnu_version=110401 --display_error_number \
    --orig_src_file_name "${SOURCE_FILE}.cu" \
    --orig_src_path_name "$SOURCE_PATH" \
    --allow_managed --m64 --parse_templates \
    --gen_c_file_name "/tmp/tmpxft_$$-6_${SOURCE_FILE}.cudafe1.cpp" \
    --stub_file_name "/tmp/tmpxft_$$-6_${SOURCE_FILE}.cudafe1.stub.c" \
    --gen_module_id_file --module_id_file_name "/tmp/tmpxft_$$-4_${SOURCE_FILE}.module_id" \
    "/tmp/tmpxft_$$-5_${SOURCE_FILE}.cpp4.ii"

# ========
# Step 3: Preprocess again with GCC
# ========
gcc -D__CUDA_ARCH__=900 -D__CUDA_ARCH_LIST__=900 -E -x c++ \
    -DCUDA_DOUBLE_MATH_FUNCTIONS -D__CUDACC__ -D__NVCC__ \
    "$INCLUDES" \
    -D__CUDACC_VER_MAJOR__=12 -D__CUDACC_VER_MINOR__=3 -D__CUDACC_VER_BUILD__=107 \
    -D__CUDA_API_VER_MAJOR__=12 -D__CUDA_API_VER_MINOR__=3 \
    -D__NVCC_DIAG_PRAGMA_SUPPORT__=1 \
    -include "cuda_runtime.h" \
    -m64 "${SOURCE_FILE}.cu" \
    -o "/tmp/tmpxft_$$-9_${SOURCE_FILE}.cpp1.ii"

# ========
# Step 4: cicc - device code compilation
# ========
"$TOP/nvvm/bin/cicc" --c++17 --gnu_version=110401 --display_error_number \
    --orig_src_file_name "${SOURCE_FILE}.cu" \
    --orig_src_path_name "$SOURCE_PATH" \
    --allow_managed -arch compute_90 -m64 --no-version-ident \
    -ftz=0 -prec_div=1 -prec_sqrt=1 -fmad=1 \
    --include_file_name "/tmp/tmpxft_$$-3_${SOURCE_FILE}.fatbin.c" \
    -tused --module_id_file_name "/tmp/tmpxft_$$-4_${SOURCE_FILE}.module_id" \
    --gen_c_file_name "/tmp/tmpxft_$$-6_${SOURCE_FILE}.cudafe1.c" \
    --stub_file_name "/tmp/tmpxft_$$-6_${SOURCE_FILE}.cudafe1.stub.c" \
    --gen_device_file_name "/tmp/tmpxft_$$-6_${SOURCE_FILE}.cudafe1.gpu" \
    "/tmp/tmpxft_$$-9_${SOURCE_FILE}.cpp1.ii" \
    -o "/tmp/tmpxft_$$-6_${SOURCE_FILE}.ptx"

# ========
# Step 5: ptxas - assemble PTX to SASS
# ========
ptxas -arch=sm_90 -m64 \
    "/tmp/tmpxft_$$-6_${SOURCE_FILE}.ptx" \
    -o "/tmp/tmpxft_$$-10_${SOURCE_FILE}.sm_90.cubin"

# ========
# Step 6: fatbinary - embed compiled binaries
# ========
fatbinary -64 \
    --cicc-cmdline="-ftz=0 -prec_div=1 -prec_sqrt=1 -fmad=1 " \
    --image3=kind=elf,sm=90,file="/tmp/tmpxft_$$-10_${SOURCE_FILE}.sm_90.cubin" \
    --image3=kind=ptx,sm=90,file="/tmp/tmpxft_$$-6_${SOURCE_FILE}.ptx" \
    --embedded-fatbin="/tmp/tmpxft_$$-3_${SOURCE_FILE}.fatbin.c"

# ========
# Step 7: Compile host-side stub
# ========
gcc -D__CUDA_ARCH__=900 -D__CUDA_ARCH_LIST__=900 -c -x c++ \
    -DCUDA_DOUBLE_MATH_FUNCTIONS "$INCLUDES" -m64 \
    "/tmp/tmpxft_$$-6_${SOURCE_FILE}.cudafe1.cpp" \
    -o "/tmp/tmpxft_$$-11_${SOURCE_FILE}.o"

# ========
# Step 8: Link device object
# ========
nvlink -m64 --arch=sm_90 \
    --register-link-binaries="/tmp/tmpxft_$$-7_${SOURCE_FILE}_dlink.reg.c" \
    -L"$TOP/targets/x86_64-linux/lib/stubs" \
    -L"$TOP/targets/x86_64-linux/lib" \
    -cpu-arch=X86_64 "/tmp/tmpxft_$$-11_${SOURCE_FILE}.o" \
    -lcudadevrt \
    -o "/tmp/tmpxft_$$-12_${SOURCE_FILE}_dlink.sm_90.cubin" \
    --host-ccbin "gcc"

# ========
# Step 9: Final fatbinary for device linking
# ========
fatbinary -64 \
    --cicc-cmdline="-ftz=0 -prec_div=1 -prec_sqrt=1 -fmad=1 " \
    -link \
    --image3=kind=elf,sm=90,file="/tmp/tmpxft_$$-12_${SOURCE_FILE}_dlink.sm_90.cubin" \
    --embedded-fatbin="/tmp/tmpxft_$$-8_${SOURCE_FILE}_dlink.fatbin.c"

# ========
# Step 10: Final GCC compile with embedded fatbin
# ========
gcc -D__CUDA_ARCH_LIST__=900 -c -x c++ \
    -DFATBINFILE="\"/tmp/tmpxft_$$-8_${SOURCE_FILE}_dlink.fatbin.c\"" \
    -DREGISTERLINKBINARYFILE="\"/tmp/tmpxft_$$-7_${SOURCE_FILE}_dlink.reg.c\"" \
    -I. -D__NV_EXTRA_INITIALIZATION= -D__NV_EXTRA_FINALIZATION= \
    -D__CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__ \
    "$INCLUDES" \
    -D__CUDACC_VER_MAJOR__=12 -D__CUDACC_VER_MINOR__=3 -D__CUDACC_VER_BUILD__=107 \
    -D__CUDA_API_VER_MAJOR__=12 -D__CUDA_API_VER_MINOR__=3 \
    -D__NVCC_DIAG_PRAGMA_SUPPORT__=1 \
    -m64 "$TOP/bin/crt/link.stub" \
    -o "/tmp/tmpxft_$$-13_${SOURCE_FILE}_dlink.o"

# Optionally clean up temp files (uncomment if needed)
# rm /tmp/tmpxft_$$-*.fatbin

