#!/bin/bash

SOURCE_FILES=('file1', 'file2', 'file3')
SOURCE_PATH=$HOME


function ptx_compile(){
   printf "Converting kernels to PTX files\n"
}



function sass_compile(){
  printf "Converting binaries to SASS files\n"
	
}

if [[ -n $SOURCE_PATH ]]; then
    cd "$SOURCE_PATH"
else #move to current working directory
    cd "."
fi

ptx_compile

sass_compile


#TO-DO: 
# Find out if files are cubin files- in general check the file types
#check commands to see what's going on

NVCC:

nvcc -ptx "$cufile" -arch "$ARCH" -o "$cufile".ptx
nvcc -ptx "$cufile" -arch "$ARCH" -o "$cufile".ptx
nvcc -arch="$ARCH" -cubin "$ptxfile" -o "$ptxfile".cubin
nvcc -arch="$ARCH" -cubin $ptxfile -o "$ptxfile".cubin

CUOBJDUMP:

cuobjdump --dump-sass convolution.cubin > convolution.sass
cuobjdump --dump-sass -arch sm_89 convolution_arch89.cubin > convolution_arch89.sass
cuobjdump --dump-sass -arch sm_90 convolution_arch90.cubin > convolution_arch90.sass
cuobjdump --gpu-architecture=sm_90 convolution_inline.binary > convolution_inline_sm90.SASS
cuobjdump --gpu-architecture=sm_90a convolution_inline.binary > convolution_inline_sm90a.SASS


DRYRUN: (View compilation steps)

nvcc convolution.cu -arch=sm_90 -o convolution_dry --dryrun
