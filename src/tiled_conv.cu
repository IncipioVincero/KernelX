
//Tiled 2D Convolution Kernel Using Caching for Halos and Constant Memory for F
//Source: Programming Massively Parallel Processors: A Hands-on Approach 4th Edition (page 169)

#define TILE_DIM 32
__constant__ float F_c[2*FILTER_RADIUS+1][2*FILTER_RADIUS+1];
__global__ void convolution_cached_tiled_2D_const_mem_kernel (float *N, float *P, int width, int height)
{
	int col = blockIdx.x*TILE_DIM + threadIdx.x;
	int row = blockIdx.y*TILE_DIM + threadIdx.y;
	//loading input tile
	__shared__ N_s[TILE_DIM][TILE_DIM];
	if (row<height && col<width)
	{
		N_s[threadIdx.y][threadIdx.x] = N[row*width + col];
	}
	else
	{
		N_s[threadIdx.y][threadIdx.x] = 0.0;
	}
	__syncthreads();
	//Calculating output elements
	//turning off the threads at the edges of the block
	if (col < width && row < height)
	{
		float Pvalue = 0.0f;
		for (int fRow = 0; fRow < 2*FILTER_RADIUS+1; fRow++)
		{
			for (int fCol = 0; fCol < 2*FILTER_RADIUS+1; fCol++)
			{
				if (threadIdx.x-FILTER_RADIUS + fCol >= 0 &&
				    threadIdx.x-FILTER_RADIUS + fCol < TILE_DIM &&
				    threadIdx.y-FILTER_RADIUS + fRow >= 0 &&
				    threadIdx.y-FILTER_RADIUS + fRow < TILE_DIM)
				{
					Pvalue += F[fRow][fCol]*N_s[threadIdx.y + fRow][threadIdxl.x + fCol];
				}
				else
				{
					if (row-FILTER_RADIUS + fRow >= 0 &&
				            row-FILTER_RADIUS + fRow < height &&
					    col-FILTER_RADIUS + fCol >= 0 &&
					    col-FILTER_RADIUS + fCol < width)
					{
						Pvalue += F[fRow][fCol] * N[(row-FILTER_RADIUS + fRow) * width + col-FILTER_RADIUS + fCol];
					}
				}
			}
			P[row * width + col] = Pvalue;
		}
	}


}
