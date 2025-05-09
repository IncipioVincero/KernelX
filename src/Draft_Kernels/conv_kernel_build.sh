#!/bin/bash


_NVVM_BRANCH_=nvvm
_SPACE_=
_CUDART_=cudart
_HERE_=/usr/local/cuda-12.3/bin
_THERE_=/usr/local/cuda-12.3/bin
_TARGET_SIZE_=
_TARGET_DIR_=
_TARGET_DIR_=targets/x86_64-linux
TOP=/usr/local/cuda-12.3/bin/..
NVVMIR_LIBRARY_DIR=/usr/local/cuda-12.3/bin/../nvvm/libdevice
LD_LIBRARY_PATH=/usr/local/cuda-12.3/bin/../lib:/usr/local/cuda-12.3/targets/x86_64-linux/lib:/usr/local/lib/python3.9/site-packages/nvidia/cudnn/lib
PATH=/usr/local/cuda-12.3/bin/../nvvm/bin:/usr/local/cuda-12.3/bin:/usr/local/cuda-12.3/bin:/usr/share/Modules/bin:/usr/local/bin:/usr/bin:/usr/local/sbin:/usr/sbin:/opt/dell/srvadmin/sbin:/insomnia001/home/kw2175/bin:/opt/dell/srvadmin/sbin
INCLUDES="-I/usr/local/cuda-12.3/bin/../targets/x86_64-linux/include"
LIBRARIES=  "-L/usr/local/cuda-12.3/bin/../targets/x86_64-linux/lib/stubs" "-L/usr/local/cuda-12.3/bin/targets/x86_64-linux/lib"
CUDAFE_FLAGS=
PTXAS_FLAGS=
SOURCE_FILE=<insert source file name>
SOURCE_PATH=<insert your source path>

gcc -D__CUDA_ARCH_LIST__=900 -E -x c++ -D__CUDACC__ -D__NVCC__  "-I/usr/local/cuda-12.3/bin/targets/x86_64-linux/include"    -D__CUDACC_VER_MAJOR__=12 -D__CUDACC_VER_MINOR__=3 -D__CUDACC_VER_BUILD__=107 -D__CUDA_API_VER_MAJOR__=12 -D__CUDA_API_VER_MINOR__=3 -D__NVCC_DIAG_PRAGMA_SUPPORT__=1 -include "cuda_runtime.h" -m64 $SOURCE_FILE -o "/tmp/tmpxft_002342a0_00000000-5_$SOURCE_FILE.cpp4.ii"

cudafe++ --c++17 --gnu_version=110401 --display_error_number --orig_src_file_name $SOURCE_FILE --orig_src_path_name $SOURCE_PATH --allow_managed  --m64 --parse_templates --gen_c_file_name "/tmp/tmpxft_002342a0_00000000-6_$SOURCE_FILE.cudafe1.cpp" --stub_file_name "tmpxft_002342a0_00000000-6_$SOURCE_FILE.cudafe1.stub.c" --gen_module_id_file --module_id_file_name "/tmp/tmpxft_002342a0_00000000-4_$SOURCE_FILE.module_id" "/tmp/tmpxft_002342a0_00000000-5_conv_single_kernel.cpp4.ii"

gcc -D__CUDA_ARCH__=900 -D__CUDA_ARCH_LIST__=900 -E -x c++  -DCUDA_DOUBLE_MATH_FUNCTIONS -D__CUDACC__ -D__NVCC__  "-I/usr/local/cuda-12.3/bin/targets/x86_64-linux/include"    -D__CUDACC_VER_MAJOR__=12 -D__CUDACC_VER_MINOR__=3 -D__CUDACC_VER_BUILD__=107 -D__CUDA_API_VER_MAJOR__=12 -D__CUDA_API_VER_MINOR__=3 -D__NVCC_DIAG_PRAGMA_SUPPORT__=1 -include "cuda_runtime.h" -m64 $SOURCE_FILE -o "/tmp/tmpxft_002342a0_00000000-9_conv_single_kernel.cpp1.ii"
cicc --c++17 --gnu_version=110401 --display_error_number --orig_src_file_name $SOURCE_FILE --orig_src_path_name $SOURCE_PATH --allow_managed   -arch compute_90 -m64 --no-version-ident -ftz=0 -prec_div=1 -prec_sqrt=1 -fmad=1 --include_file_name "tmpxft_002342a0_00000000-3_conv_single_kernel.fatbin.c" -tused --module_id_file_name "/tmp/tmpxft_002342a0_00000000-4_conv_single_kernel.module_id" --gen_c_file_name "/tmp/tmpxft_002342a0_00000000-6_conv_single_kernel.cudafe1.c" --stub_file_name "/tmp/tmpxft_002342a0_00000000-6_conv_single_kernel.cudafe1.stub.c" --gen_device_file_name "/tmp/tmpxft_002342a0_00000000-6_conv_single_kernel.cudafe1.gpu"  "/tmp/tmpxft_002342a0_00000000-9_conv_single_kernel.cpp1.ii" -o "/tmp/tmpxft_002342a0_00000000-6_conv_single_kernel.ptx"

ptxas -arch=sm_90 -m64  "/tmp/tmpxft_002342a0_00000000-6_conv_single_kernel.ptx"  -o "/tmp/tmpxft_002342a0_00000000-10_conv_single_kernel.sm_90.cubin"

fatbinary -64 --cicc-cmdline="-ftz=0 -prec_div=1 -prec_sqrt=1 -fmad=1 " "--image3=kind=elf,sm=90,file=/tmp/tmpxft_002342a0_00000000-10_conv_single_kernel.sm_90.cubin" "--image3=kind=ptx,sm=90,file=/tmp/tmpxft_002342a0_00000000-6_conv_single_kernel.ptx" --embedded-fatbin="/tmp/tmpxft_002342a0_00000000-3_conv_single_kernel.fatbin.c"

rm /tmp/tmpxft_002342a0_00000000-3_conv_single_kernel.fatbin

gcc -D__CUDA_ARCH__=900 -D__CUDA_ARCH_LIST__=900 -c -x c++  -DCUDA_DOUBLE_MATH_FUNCTIONS "-I/usr/local/cuda-12.3/bin/../targets/x86_64-linux/include"   -m64 "/tmp/tmpxft_002342a0_00000000-6_conv_single_kernel.cudafe1.cpp" -o "/tmp/tmpxft_002342a0_00000000-11_conv_single_kernel.o"

nvlink -m64 --arch=sm_90 --register-link-binaries="/tmp/tmpxft_002342a0_00000000-7_conv_single_kernel_dryrun_dlink.reg.c"    "-L/usr/local/cuda-12.3/bin/../targets/x86_64-linux/lib/stubs" "-L/usr/local/cuda-12.3/bin/../targets/x86_64-linux/lib" -cpu-arch=X86_64 "/tmp/tmpxft_002342a0_00000000-11_conv_single_kernel.o"  -lcudadevrt  -o "/tmp/tmpxft_002342a0_00000000-12_conv_single_kernel_dryrun_dlink.sm_90.cubin" --host-ccbin "gcc"

fatbinary -64 --cicc-cmdline="-ftz=0 -prec_div=1 -prec_sqrt=1 -fmad=1 " -link "--image3=kind=elf,sm=90,file=/tmp/tmpxft_002342a0_00000000-12_conv_single_kernel_dryrun_dlink.sm_90.cubin" --embedded-fatbin="/tmp/tmpxft_002342a0_00000000-8_conv_single_kernel_dryrun_dlink.fatbin.c"

rm /tmp/tmpxft_002342a0_00000000-8_conv_single_kernel_dryrun_dlink.fatbin

gcc -D__CUDA_ARCH_LIST__=900 -c -x c++ -DFATBINFILE="\"/tmp/tmpxft_002342a0_00000000-8_conv_single_kernel_dryrun_dlink.fatbin.c\"" -DREGISTERLINKBINARYFILE="\"/tmp/tmpxft_002342a0_00000000-7_conv_single_kernel_dryrun_dlink.reg.c\"" -I. -D__NV_EXTRA_INITIALIZATION= -D__NV_EXTRA_FINALIZATION= -D__CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__  "-I/usr/local/cuda-12.3/bin/../targets/x86_64-linux/include"    -D__CUDACC_VER_MAJOR__=12 -D__CUDACC_VER_MINOR__=3 -D__CUDACC_VER_BUILD__=107 -D__CUDA_API_VER_MAJOR__=12 -D__CUDA_API_VER_MINOR__=3 -D__NVCC_DIAG_PRAGMA_SUPPORT__=1 -m64 "/usr/local/cuda-12.3/bin/crt/link.stub" -o "/tmp/tmpxft_002342a0_00000000-13_conv_single_kernel_dryrun_dlink.o"

g++ -D__CUDA_ARCH_LIST__=900 -m64 -Wl,--start-group "/tmp/tmpxft_002342a0_00000000-13_conv_single_kernel_dryrun_dlink.o" "/tmp/tmpxft_002342a0_00000000-11_conv_single_kernel.o"   "-L/usr/local/cuda-12.3/bin/../targets/x86_64-linux/lib/stubs" "-L/usr/local/cuda-12.3/bin/../targets/x86_64-linux/lib"  -lcudadevrt  -lcudart_static  -lrt -lpthread  -ldl  -Wl,--end-group -o "conv_single_kernel_dryrun"

