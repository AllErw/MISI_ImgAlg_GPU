// NOTES: 
// 1. DO NOT PRECOMPUTE DELAYS, as the memory operations required to apply these during DAS/DMAS actually
//    result in 2-fold SLOWER computation.
// 2. For chunk-wise computations, consider providing the chunk size as an input somehow.

/* Image reconstruction on the GPU is performed by distributing the pixels across the blocks and threads
   and computing each pixel value independently and in parallel.*/

#include "MISI_ImgAlg_GPU.h"
#include <math.h>

// ==========================================================================================================
// ==                                                  DAS                                                 ==
// ==========================================================================================================
// DAS reconstruction for a single, stationary receiver. This is representative of the freespace or freehand
// imaging setups, where a single stationary receiver is  paired with a number of sources distributed across 
// a 1D or 2D aperture, for instance using scanning optics.
__global__
void DnS_1rec_fixed_pos_GPU(
	float *rf_data,	float *source_locations,	float *receiver_location,	float *image_coordinates,
	float c, float fsamp, int Nsrc, int Nt, int Nimg,
	float *image)
{
    int srccnt,tcnt;
    int Nsrc2 = 2*Nsrc,     Nimg2 = 2*Nimg;
    float feff = fsamp/c;
    float ximg,yimg,zimg;
	float dist_src_refl,dist_refl_hydr;
    
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

    for (int imgcnt = index; imgcnt < Nimg;  imgcnt += stride) {
		ximg = image_coordinates[imgcnt];
        yimg = image_coordinates[imgcnt+Nimg];
        zimg = image_coordinates[imgcnt+Nimg2];
        dist_refl_hydr = sqrtf(powf(ximg - receiver_location[0] , 2) + 
                               powf(yimg - receiver_location[1] , 2) + 
                               powf(zimg - receiver_location[2] , 2) );
        for (srccnt=0; srccnt<Nsrc; srccnt++){
            dist_src_refl = sqrtf(powf(ximg - source_locations[srccnt] , 2) + 
                                  powf(yimg - source_locations[srccnt+Nsrc] , 2) + 
                                  powf(zimg - source_locations[srccnt+Nsrc2] , 2) );
            tcnt = roundf( (dist_refl_hydr + dist_src_refl)*feff );
			if (tcnt < Nt) { image[imgcnt] += rf_data[tcnt + srccnt * Nt]; }	// To ensure time sample actually exists!
        }
    }
	
	return;
}

// DAS reconstruction assuming a single receiver that coincides with each source position.
// This is approximately representative of the case where a 2-fibre probe is mechanically translated.
__global__
void DnS_1rec_at_src_GPU(
	float* rf_data, float* source_locations, float* image_coordinates,
	float c, float fsamp, int Nsrc, int Nt, int Nimg,
	float* image)
{
	int srccnt, tcnt;
	int Nsrc2 = 2 * Nsrc, Nimg2 = 2 * Nimg;
	float feff = fsamp / c;
	float ximg, yimg, zimg;
	float dist_src_refl;

	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int imgcnt = index; imgcnt < Nimg; imgcnt += stride) {
		ximg = image_coordinates[imgcnt];
		yimg = image_coordinates[imgcnt + Nimg];
		zimg = image_coordinates[imgcnt + Nimg2];
		for (srccnt = 0; srccnt < Nsrc; srccnt++) {
			dist_src_refl = sqrtf(powf(ximg - source_locations[srccnt], 2) +
								  powf(yimg - source_locations[srccnt + Nsrc], 2) +
								  powf(zimg - source_locations[srccnt + Nsrc2], 2));
			tcnt = roundf(2.0f * dist_src_refl * feff);
			if (tcnt < Nt) { image[imgcnt] += rf_data[tcnt + srccnt * Nt]; }	// To ensure time sample actually exists!
		}
	}

	return;
}

// DAS reconstruction in the most general case, with arbitrary number of sources and receivers. Each column 
// in rf_data corresponds to a unique source-receiver pair, the locations of which are given in the 
// corresponding rows of source_locations and receiver_locations.
__global__
void DnS_Nrec_arb_pos_GPU(
	float* rf_data, float* source_locations, float* receiver_locations, float* image_coordinates,
	float c, float fsamp, int Nsrc, int Nt, int Nimg,
	float* image)
{
	int srccnt, tcnt;
	int Nsrc2 = 2 * Nsrc, Nimg2 = 2 * Nimg;
	float feff = fsamp / c;
	float ximg, yimg, zimg;
	float dist_src_refl, dist_refl_hydr;

	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int imgcnt = index; imgcnt < Nimg; imgcnt += stride) {
		ximg = image_coordinates[imgcnt];
		yimg = image_coordinates[imgcnt + Nimg];
		zimg = image_coordinates[imgcnt + Nimg2];
		for (srccnt = 0; srccnt < Nsrc; srccnt++) {
			dist_refl_hydr = sqrtf(powf(ximg - receiver_locations[srccnt], 2) +
								   powf(yimg - receiver_locations[srccnt+Nsrc], 2) +
								   powf(zimg - receiver_locations[srccnt+Nsrc2], 2));
			dist_src_refl = sqrtf(powf(ximg - source_locations[srccnt], 2) +
								  powf(yimg - source_locations[srccnt + Nsrc], 2) +
								  powf(zimg - source_locations[srccnt + Nsrc2], 2));
			tcnt = roundf((dist_refl_hydr + dist_src_refl) * feff);
			if (tcnt < Nt) { image[imgcnt] += rf_data[tcnt + srccnt * Nt]; }	// To ensure time sample actually exists!
		}
	}

	return;
}





// ==========================================================================================================
// ==                                                  DMAS                                                ==
// ==========================================================================================================
// DMAS reconstruction for a single, stationary receiver. This is representative of the freespace or freehand
// imaging setups, where a single stationary receiver is  paired with a number of sources distributed across 
// a 1D or 2D aperture, for instance using scanning optics.
__global__
void DMnS_1rec_fixed_pos_GPU(
	float *rf_data,	float *source_locations, float *receiver_location, float *image_coordinates,
	float c, float fsamp, int Nsrc, int Nt, int Nimg,
	float *image)
{
	int srccnt, tcnt, icnt, jcnt;
	int Nsrc2 = 2 * Nsrc, Nimg2 = 2 * Nimg;
	float feff = fsamp / c;
	float ximg, yimg, zimg;
	float dist_src_refl, dist_refl_hydr;
	float si, sj, product = 0.0f;

	float* delayed = (float*)malloc(Nsrc * sizeof(float)); //Allocates memory in RAM, not on GPU!!!
	memset(delayed, 0.0f, Nsrc * sizeof(float));

	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int imgcnt = index; imgcnt < Nimg; imgcnt += stride) {
		ximg = image_coordinates[imgcnt];
		yimg = image_coordinates[imgcnt + Nimg];
		zimg = image_coordinates[imgcnt + Nimg2];
		dist_refl_hydr = sqrtf(powf(ximg - receiver_location[0] , 2) +
							   powf(yimg - receiver_location[1] , 2) +
							   powf(zimg - receiver_location[2] , 2) );
		// This loop delays the RF data and stores the appropriate values in variable "delayed":
		for (srccnt = 0; srccnt < Nsrc; srccnt++) { 
			dist_src_refl = sqrtf(powf(ximg - source_locations[srccnt] , 2) +
								  powf(yimg - source_locations[srccnt + Nsrc] , 2) +
								  powf(zimg - source_locations[srccnt + Nsrc2] , 2) );
			tcnt = roundf((dist_refl_hydr + dist_src_refl)*feff);
			if (tcnt < Nt) { delayed[srccnt] = rf_data[tcnt + srccnt * Nt]; }
			else		   { delayed[srccnt] = 0.0f; }
			//image[imgcnt] += delayed[srccnt] + 1.1f;	// Delay and sum. Uncommenting this (and commenting out the loop below) proves that the code works fine up to here!
		}
		// This loop sums over i and j and performs the actual DM&S computations:
		for (icnt = 0; icnt < Nsrc - 1; icnt++) {
			si = delayed[icnt];
			for (jcnt = icnt + 1; jcnt < Nsrc; jcnt++) {
				sj = delayed[jcnt];
				product = sqrtf( fabsf(si*sj) );
				product = copysignf(product, (si*sj));
				image[imgcnt] += product;
			}
		}
	}

	free(delayed);

	return;
}

// DMAS reconstruction assuming a single receiver that coincides with each source position.
// This is approximately representative of the case where a 2-fibre probe is mechanically translated.
__global__
void DMnS_1rec_at_src_GPU(
	float* rf_data, float* source_locations, float* image_coordinates,
	float c, float fsamp, int Nsrc, int Nt, int Nimg,
	float* image)
{
	int srccnt, tcnt, icnt, jcnt;
	int Nsrc2 = 2 * Nsrc, Nimg2 = 2 * Nimg;
	float feff = fsamp / c;
	float ximg, yimg, zimg;
	float dist_src_refl;
	float si, sj, product = 0.0f;

	float* delayed = (float*)malloc(Nsrc * sizeof(float));
	memset(delayed, 0.0f, Nsrc * sizeof(float));

	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int imgcnt = index; imgcnt < Nimg; imgcnt += stride) {
		ximg = image_coordinates[imgcnt];
		yimg = image_coordinates[imgcnt + Nimg];
		zimg = image_coordinates[imgcnt + Nimg2];
		// This loop delays the RF data and stores the appropriate values in variable "delayed":
		for (srccnt = 0; srccnt < Nsrc; srccnt++) {
			dist_src_refl = sqrtf(powf(ximg - source_locations[srccnt], 2) +
				powf(yimg - source_locations[srccnt + Nsrc], 2) +
				powf(zimg - source_locations[srccnt + Nsrc2], 2));
			tcnt = roundf(2.0f * dist_src_refl * feff);
			if (tcnt < Nt) { delayed[srccnt] = rf_data[tcnt + srccnt * Nt]; }
			else { delayed[srccnt] = 0.0f; }
			//image[imgcnt] += delayed[srccnt] + 1.1f;	// Delay and sum. Uncommenting this (and commenting out the loop below) proves that the code works fine up to here!
		}
		// This loop sums over i and j and performs the actual DM&S computations:
		for (icnt = 0; icnt < Nsrc - 1; icnt++) {
			si = delayed[icnt];
			for (jcnt = icnt + 1; jcnt < Nsrc; jcnt++) {
				sj = delayed[jcnt];
				product = sqrtf(fabsf(si * sj));
				product = copysignf(product, (si * sj));
				image[imgcnt] += product;
			}
		}
	}

	free(delayed);

	return;
}

// DMAS reconstruction in the most general case, with arbitrary number of sources and receivers. Each column 
// in rf_data corresponds to a unique source-receiver pair, the locations of which are given in the 
// corresponding rows of source_locations and receiver_locations.
__global__
void DMnS_Nrec_arb_pos_GPU(
	float* rf_data, float* source_locations, float* receiver_locations, float* image_coordinates,
	float c, float fsamp, int Nsrc, int Nt, int Nimg,
	float* image)
{
	int srccnt, tcnt, icnt, jcnt;
	int Nsrc2 = 2 * Nsrc, Nimg2 = 2 * Nimg;
	float feff = fsamp / c;
	float ximg, yimg, zimg;
	float dist_src_refl, dist_refl_hydr;
	float si, sj, product = 0.0f;

	float* delayed = (float*)malloc(Nsrc * sizeof(float));
	memset(delayed, 0.0f, Nsrc * sizeof(float));

	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int imgcnt = index; imgcnt < Nimg; imgcnt += stride) {
		ximg = image_coordinates[imgcnt];
		yimg = image_coordinates[imgcnt + Nimg];
		zimg = image_coordinates[imgcnt + Nimg2];
		// This loop delays the RF data and stores the appropriate values in variable "delayed":
		for (srccnt = 0; srccnt < Nsrc; srccnt++) {
			dist_refl_hydr = sqrtf(powf(ximg - receiver_locations[srccnt], 2) +
								   powf(yimg - receiver_locations[srccnt + Nsrc], 2) +
								   powf(zimg - receiver_locations[srccnt + Nsrc2], 2));
			dist_src_refl = sqrtf(powf(ximg - source_locations[srccnt], 2) +
				powf(yimg - source_locations[srccnt + Nsrc], 2) +
				powf(zimg - source_locations[srccnt + Nsrc2], 2));
			tcnt = roundf((dist_refl_hydr + dist_src_refl) * feff);
			if (tcnt < Nt) { delayed[srccnt] = rf_data[tcnt + srccnt * Nt]; }
			else { delayed[srccnt] = 0.0f; }
			//image[imgcnt] += delayed[srccnt] + 1.1f;	// Delay and sum. Uncommenting this (and commenting out the loop below) proves that the code works fine up to here!
		}
		// This loop sums over i and j and performs the actual DM&S computations:
		for (icnt = 0; icnt < Nsrc - 1; icnt++) {
			si = delayed[icnt];
			for (jcnt = icnt + 1; jcnt < Nsrc; jcnt++) {
				sj = delayed[jcnt];
				product = sqrtf(fabsf(si * sj));
				product = copysignf(product, (si * sj));
				image[imgcnt] += product;
			}
		}
	}

	free(delayed);

	return;
}





// ==========================================================================================================
// ==                                                  SLSC                                                ==
// ==========================================================================================================
// SLSC reconstruction for a single, stationary receiver. This is representative of the freespace or freehand
// imaging setups, where a single stationary receiver is  paired with a number of sources distributed across 
// a 1D or 2D aperture, for instance using scanning optics.
__global__
void SLSC_1rec_fixed_pos_GPU(
	float *rf_data, float *source_locations, float *receiver_location, float *image_coordinates, float *rf_data_squared,
	float c, float fsamp, int Nsrc, int Nt, int Nimg, int m, int w,
	float *image)
{
	int imgcnt, srccnt, mcnt, wcnt, index1, index2;
	int Nsrc2 = 2 * Nsrc, Nimg2 = 2 * Nimg;
	float feff = fsamp / c;
	float ximg, yimg, zimg;
	float dist_src_refl, dist_refl_hydr, Rhat, numerator, denominator, denominator1, denominator2;
	
	// Initialise arrival times vector:
	int* arr_times = (int*)malloc(Nsrc * sizeof(int));
	memset(arr_times, 0, Nsrc * sizeof(int));
	
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	// The actual computation:
	for (imgcnt = index; imgcnt < Nimg; imgcnt += stride) {// Loop over all image pixels
		
		// Extract coordinates of current pixel:
		ximg = image_coordinates[imgcnt];
		yimg = image_coordinates[imgcnt + Nimg];
		zimg = image_coordinates[imgcnt + Nimg2];

		dist_refl_hydr = sqrtf(powf(ximg - receiver_location[0], 2) +
							   powf(yimg - receiver_location[1], 2) +
							   powf(zimg - receiver_location[2], 2));

		// Compute the arrival times across the aperture:
		for (srccnt = 0; srccnt < Nsrc; srccnt++) {
			dist_src_refl = sqrtf(powf(ximg - source_locations[srccnt], 2) +
								  powf(yimg - source_locations[srccnt + Nsrc], 2) +
								  powf(zimg - source_locations[srccnt + Nsrc2], 2));
			arr_times[srccnt] = roundf((dist_refl_hydr + dist_src_refl) * feff);
		}
		
		for (mcnt = 0; mcnt < m; mcnt++) {
			// Re-initialise Rhat to zero:
			Rhat = 0.0;

			for (srccnt = 0; srccnt < Nsrc - mcnt; srccnt++) {
				numerator = 0.0; denominator1 = 0.0; denominator2 = 0.0;

				for (wcnt = 0; wcnt < w; wcnt++) {
					index1 = wcnt + arr_times[srccnt] + srccnt * Nt;
					index2 = wcnt + arr_times[srccnt + mcnt] + (srccnt + mcnt) * Nt;
					if ((wcnt + arr_times[srccnt + mcnt]) < Nt) {
						numerator += rf_data[index1] * rf_data[index2];
						denominator1 += rf_data_squared[index1];
						denominator2 += rf_data_squared[index2];
					}
				}
				denominator = sqrtf(denominator1 * denominator2);

				if (denominator == 0.0) { denominator = 0.000000001; } // avoid divide by 0
				Rhat += numerator / denominator;
			}
			Rhat /= (1.0f * (Nsrc - mcnt));
			image[imgcnt] += Rhat;
		}
	}
	free(arr_times);
	return;
}

__global__
void SLSC_1rec_at_src_GPU(
	float* rf_data, float* source_locations, float* image_coordinates, float* rf_data_squared,
	float c, float fsamp, int Nsrc, int Nt, int Nimg, int m, int w,
	float* image)
{
	int imgcnt, srccnt, mcnt, wcnt, index1, index2;
	int Nsrc2 = 2 * Nsrc, Nimg2 = 2 * Nimg;
	float feff = fsamp / c;
	float ximg, yimg, zimg;
	float dist_src_refl, Rhat, numerator, denominator, denominator1, denominator2;

	// Initialise arrival times vector:
	int* arr_times = (int*)malloc(Nsrc * sizeof(int));
	memset(arr_times, 0, Nsrc * sizeof(int));

	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	// The actual computation:
	for (imgcnt = index; imgcnt < Nimg; imgcnt += stride) {// Loop over all image pixels

		// Extract coordinates of current pixel:
		ximg = image_coordinates[imgcnt];
		yimg = image_coordinates[imgcnt + Nimg];
		zimg = image_coordinates[imgcnt + Nimg2];

		// Compute the arrival times across the aperture:
		for (srccnt = 0; srccnt < Nsrc; srccnt++) {
			dist_src_refl = sqrtf(powf(ximg - source_locations[srccnt], 2) +
				powf(yimg - source_locations[srccnt + Nsrc], 2) +
				powf(zimg - source_locations[srccnt + Nsrc2], 2));
			arr_times[srccnt] = roundf(2.0f * dist_src_refl * feff);
		}

		for (mcnt = 0; mcnt < m; mcnt++) {
			// Re-initialise Rhat to zero:
			Rhat = 0.0;

			for (srccnt = 0; srccnt < Nsrc - mcnt; srccnt++) {
				numerator = 0.0; denominator1 = 0.0; denominator2 = 0.0;

				for (wcnt = 0; wcnt < w; wcnt++) {
					index1 = wcnt + arr_times[srccnt] + srccnt * Nt;
					index2 = wcnt + arr_times[srccnt + mcnt] + (srccnt + mcnt) * Nt;
					if ((wcnt + arr_times[srccnt + mcnt]) < Nt) {
						numerator += rf_data[index1] * rf_data[index2];
						denominator1 += rf_data_squared[index1];
						denominator2 += rf_data_squared[index2];
					}
				}
				denominator = sqrtf(denominator1 * denominator2);

				if (denominator == 0.0) { denominator = 0.000000001; } // avoid divide by 0
				Rhat += numerator / denominator;
			}
			Rhat /= (1.0f * (Nsrc - mcnt));
			image[imgcnt] += Rhat;
		}
	}
	free(arr_times);
	return;
}

__global__
void SLSC_Nrec_arb_pos_GPU(
	float* rf_data, float* source_locations, float* receiver_locations, float* image_coordinates, float* rf_data_squared,
	float c, float fsamp, int Nsrc, int Nt, int Nimg, int m, int w,
	float* image)
{
	int imgcnt, srccnt, mcnt, wcnt, index1, index2;
	int Nsrc2 = 2 * Nsrc, Nimg2 = 2 * Nimg;
	float feff = fsamp / c;
	float ximg, yimg, zimg;
	float dist_src_refl, dist_refl_hydr, Rhat, numerator, denominator, denominator1, denominator2;

	// Initialise arrival times vector:
	int* arr_times = (int*)malloc(Nsrc * sizeof(int));
	memset(arr_times, 0, Nsrc * sizeof(int));

	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	// The actual computation:
	for (imgcnt = index; imgcnt < Nimg; imgcnt += stride) {// Loop over all image pixels

		// Extract coordinates of current pixel:
		ximg = image_coordinates[imgcnt];
		yimg = image_coordinates[imgcnt + Nimg];
		zimg = image_coordinates[imgcnt + Nimg2];

		// Compute the arrival times across the aperture:
		for (srccnt = 0; srccnt < Nsrc; srccnt++) {
			dist_refl_hydr = sqrtf(powf(ximg - receiver_locations[srccnt], 2) +
				powf(yimg - receiver_locations[srccnt + Nsrc], 2) +
				powf(zimg - receiver_locations[srccnt + Nsrc2], 2));
			dist_src_refl = sqrtf(powf(ximg - source_locations[srccnt], 2) +
				powf(yimg - source_locations[srccnt + Nsrc], 2) +
				powf(zimg - source_locations[srccnt + Nsrc2], 2));
			arr_times[srccnt] = roundf((dist_refl_hydr + dist_src_refl) * feff);
		}

		for (mcnt = 0; mcnt < m; mcnt++) {
			// Re-initialise Rhat to zero:
			Rhat = 0.0;

			for (srccnt = 0; srccnt < Nsrc - mcnt; srccnt++) {
				numerator = 0.0; denominator1 = 0.0; denominator2 = 0.0;

				for (wcnt = 0; wcnt < w; wcnt++) {
					index1 = wcnt + arr_times[srccnt] + srccnt * Nt;
					index2 = wcnt + arr_times[srccnt + mcnt] + (srccnt + mcnt) * Nt;
					if ((wcnt + arr_times[srccnt + mcnt]) < Nt) {
						numerator += rf_data[index1] * rf_data[index2];
						denominator1 += rf_data_squared[index1];
						denominator2 += rf_data_squared[index2];
					}
				}
				denominator = sqrtf(denominator1 * denominator2);

				if (denominator == 0.0) { denominator = 0.000000001; } // avoid divide by 0
				Rhat += numerator / denominator;
			}
			Rhat /= (1.0f * (Nsrc - mcnt));
			image[imgcnt] += Rhat;
		}
	}
	free(arr_times);
	return;
}









// ==========================================================================================================
// =                                      EXTERNAL INTERFACE ROUTINES:                                      =
// ==========================================================================================================

// ==========================================================================================================
/* Each function call is repeated twice; once for all pixels at once, and once for chunks of pixels. These 
   chunks are either 393216 (DAS), 5120 (DMAS), or 10240 (SLSC) pixels large. The chunk variant was introduced
   to avoid operating system freezes or time-outs. Read up on WDDM to figure out why. In a nut shell: the WDDM 
   driver model checks whether the GPU responds within a certain time frame; if not, the GPU driver is reset 
   to ensure system stability. Thing is, some image reconstructions can take longer than this wait time -> 
   run time errors result! So dividing the image reconstruction into chunks circumvents this problem at the 
   expense of a little overhead.
*/

/* NOTE - ALL CHUNK-STYLE KERNEL CALLS ARE PERFORMED MUCH TOO OFTEN. WHILE EACH OF THOSE CALLS RESULTS IN 
   EXACTLY ZERO COMPUTATIONS, I IMAGINE THE KERNEL CALL AND NON-COMPUTE PARTS CAUSE SOME OVERHEAD... TRY!   */

// Delay-and-sum (DAS):
void __declspec(dllexport) DnS_1rec_fixed_pos_GPU_interface(
	float *rf_data,	float *source_locations,	float *receiver_location,	float *image_coordinates,
	float c, float fsamp, int Nsrc, int Nt, int Nimg,
	int *CUDAparams,
	float *image)
{
	// preallocate space on the GPU for the image, and initialise to 0:
	float *imageGPU, *rf_dataGPU, *source_locationsGPU, *receiver_locationGPU, *image_coordinatesGPU;
	cudaMalloc ( &imageGPU, Nimg*sizeof(float) );				cudaMemcpy( imageGPU    , image  , Nimg*sizeof(float) , cudaMemcpyHostToDevice);
  															  //cudaMemcpy( DESTINATION , SOURCE , bytes              , cudaMemcpyHostToDevice);
	cudaMalloc ( &rf_dataGPU, Nt*Nsrc*sizeof(float) );			cudaMemcpy( rf_dataGPU , rf_data , Nt*Nsrc*sizeof(float) , cudaMemcpyHostToDevice);
	cudaMalloc ( &source_locationsGPU, 3*Nsrc*sizeof(float) );	cudaMemcpy( source_locationsGPU , source_locations , 3*Nsrc*sizeof(float) , cudaMemcpyHostToDevice);
	cudaMalloc ( &receiver_locationGPU, 3*sizeof(float) );		cudaMemcpy( receiver_locationGPU , receiver_location , 3*sizeof(float) , cudaMemcpyHostToDevice);
	cudaMalloc ( &image_coordinatesGPU, 3*Nimg*sizeof(float) );	cudaMemcpy( image_coordinatesGPU , image_coordinates , 3*Nimg*sizeof(float) , cudaMemcpyHostToDevice);

	// Define block size and number of blocks, and call function accordingly:
	//int blockSize = 1024;
	//int numBlocks = 1;
	//int numBlocks = (Nimg + blockSize - 1) / blockSize;
	int blockSize = CUDAparams[0];
	int numBlocks = CUDAparams[1];

	DnS_1rec_fixed_pos_GPU<<<numBlocks, blockSize>>>(rf_dataGPU,source_locationsGPU,receiver_locationGPU,image_coordinatesGPU,c,fsamp,Nsrc,Nt,Nimg,imageGPU);
	cudaStreamQuery(0); // Forces the WDDM to directly flush the kernel call to the GPU rather than first batching (and hence avoids call overhead)

	// Wait for all blocks/threads of the GPU to finish:
	cudaDeviceSynchronize();
	
	// Copy image from GPU to CPU and clear GPU memory:
	cudaMemcpy( image		, imageGPU , Nimg*sizeof(float) , cudaMemcpyDeviceToHost);
  //cudaMemcpy( DESTINATION , SOURCE   , bytes              , cudaMemcpyDeviceToHost);
	cudaFree(imageGPU);	
	cudaFree(rf_dataGPU);				cudaFree(source_locationsGPU);
	cudaFree(receiver_locationGPU);		cudaFree(image_coordinatesGPU);
	
	return;
}

void __declspec(dllexport) DnS_1rec_at_src_GPU_interface(
	float* rf_data, float* source_locations, float* image_coordinates,
	float c, float fsamp, int Nsrc, int Nt, int Nimg,
	int* CUDAparams,
	float* image)
{
	// preallocate space on the GPU for the image, and initialise to 0:
	float* imageGPU, * rf_dataGPU, * source_locationsGPU, * image_coordinatesGPU;
	cudaMalloc(&imageGPU, Nimg * sizeof(float));				cudaMemcpy(imageGPU, image, Nimg * sizeof(float), cudaMemcpyHostToDevice);
	cudaMalloc(&rf_dataGPU, Nt * Nsrc * sizeof(float));			cudaMemcpy(rf_dataGPU, rf_data, Nt * Nsrc * sizeof(float), cudaMemcpyHostToDevice);
	cudaMalloc(&source_locationsGPU, 3 * Nsrc * sizeof(float));	cudaMemcpy(source_locationsGPU, source_locations, 3 * Nsrc * sizeof(float), cudaMemcpyHostToDevice);
	cudaMalloc(&image_coordinatesGPU, 3 * Nimg * sizeof(float));	cudaMemcpy(image_coordinatesGPU, image_coordinates, 3 * Nimg * sizeof(float), cudaMemcpyHostToDevice);

	// Define block size and number of blocks, and call function accordingly:
	//int blockSize = 1024;
	//int numBlocks = 1;
	//int numBlocks = (Nimg + blockSize - 1) / blockSize;
	int blockSize = CUDAparams[0];
	int numBlocks = CUDAparams[1];

	DnS_1rec_at_src_GPU<<<numBlocks, blockSize>>>(rf_dataGPU, source_locationsGPU, image_coordinatesGPU, c, fsamp, Nsrc, Nt, Nimg, imageGPU);
	cudaStreamQuery(0); // Forces the WDDM to directly flush the kernel call to the GPU rather than first batching (and hence avoids call overhead)

	// Wait for all blocks/threads of the GPU to finish:
	cudaDeviceSynchronize();

	// Copy image from GPU to CPU and clear GPU memory:
	cudaMemcpy(image, imageGPU, Nimg * sizeof(float), cudaMemcpyDeviceToHost);
	//cudaMemcpy( DESTINATION , SOURCE   , bytes              , cudaMemcpyDeviceToHost);
	cudaFree(imageGPU);
	cudaFree(rf_dataGPU);				cudaFree(source_locationsGPU);
	cudaFree(image_coordinatesGPU);

	return;
}

void __declspec(dllexport) DnS_Nrec_arb_pos_GPU_interface(
	float* rf_data, float* source_locations, float* receiver_locations, float* image_coordinates,
	float c, float fsamp, int Nsrc, int Nt, int Nimg,
	int* CUDAparams,
	float* image)
{
	// preallocate space on the GPU for the image, and initialise to 0:
	float* imageGPU, * rf_dataGPU, * source_locationsGPU, * receiver_locationsGPU, * image_coordinatesGPU;
	cudaMalloc(&imageGPU, Nimg * sizeof(float));					cudaMemcpy(imageGPU, image, Nimg * sizeof(float), cudaMemcpyHostToDevice);
	cudaMalloc(&rf_dataGPU, Nt * Nsrc * sizeof(float));				cudaMemcpy(rf_dataGPU, rf_data, Nt * Nsrc * sizeof(float), cudaMemcpyHostToDevice);
	cudaMalloc(&source_locationsGPU, 3 * Nsrc * sizeof(float));		cudaMemcpy(source_locationsGPU, source_locations, 3 * Nsrc * sizeof(float), cudaMemcpyHostToDevice);
	cudaMalloc(&receiver_locationsGPU, 3 * Nsrc * sizeof(float));	cudaMemcpy(receiver_locationsGPU, receiver_locations, 3 * Nsrc * sizeof(float), cudaMemcpyHostToDevice);
	cudaMalloc(&image_coordinatesGPU, 3 * Nimg * sizeof(float));	cudaMemcpy(image_coordinatesGPU, image_coordinates, 3 * Nimg * sizeof(float), cudaMemcpyHostToDevice);

	// Define block size and number of blocks, and call function accordingly:
	int blockSize = CUDAparams[0];
	int numBlocks = CUDAparams[1];

	DnS_Nrec_arb_pos_GPU<<<numBlocks, blockSize>>>(rf_dataGPU, source_locationsGPU, receiver_locationsGPU, image_coordinatesGPU, c, fsamp, Nsrc, Nt, Nimg, imageGPU);
	cudaStreamQuery(0); // Forces the WDDM to directly flush the kernel call to the GPU rather than first batching (and hence avoids call overhead)

	// Wait for all blocks/threads of the GPU to finish:
	cudaDeviceSynchronize();

	// Copy image from GPU to CPU and clear GPU memory:
	cudaMemcpy(image, imageGPU, Nimg * sizeof(float), cudaMemcpyDeviceToHost);
	cudaFree(imageGPU);
	cudaFree(rf_dataGPU);				cudaFree(source_locationsGPU);
	cudaFree(receiver_locationsGPU);	cudaFree(image_coordinatesGPU);

	return;
}

void __declspec(dllexport) DnS_1rec_fixed_pos_GPU_chunks_interface(
	float *rf_data, float *source_locations, float *receiver_location, float *image_coordinates,
	float c, float fsamp, int Nsrc, int Nt, int Nimg, 
	int *CUDAparams,
	float *image)
{
	// preallocate space on the GPU for the image, and initialise to 0:
	float *imageGPU, *rf_dataGPU, *source_locationsGPU, *receiver_locationGPU, *image_coordinatesGPU;
	cudaMalloc(&rf_dataGPU,			 Nt*Nsrc  * sizeof(float));	cudaMemcpy(rf_dataGPU,			 rf_data,		   Nt*Nsrc	* sizeof(float), cudaMemcpyHostToDevice);
	cudaMalloc(&source_locationsGPU, 3 * Nsrc * sizeof(float));	cudaMemcpy(source_locationsGPU,  source_locations, 3 * Nsrc * sizeof(float), cudaMemcpyHostToDevice);
	cudaMalloc(&receiver_locationGPU,3		  * sizeof(float));	cudaMemcpy(receiver_locationGPU, receiver_location,3		* sizeof(float), cudaMemcpyHostToDevice);

	// Define block size and number of blocks, and define chunk size:
	int blockSize = CUDAparams[0];
	int numBlocks = CUDAparams[1];
	int chunk_size = 393216; // 2^17 * 3
	
	// Allocate and initialise array "subset" which will hold the chunks of image_coordinates:
	float* subset = (float*)malloc(3 * chunk_size * sizeof(float));
	memset(subset,		0.0f,	   3 * chunk_size * sizeof(float));
	// Allocate GPU memory for image_coordinatesGPU and imageGPU:
	cudaMalloc(&image_coordinatesGPU, 3 * chunk_size * sizeof(float));
	cudaMalloc(&imageGPU,				  chunk_size * sizeof(float));

	// The loop through the chunks:
	int offset = 0 , num2copy = 0;
	for (int chunkcnt = 0; chunkcnt <= Nimg/chunk_size; chunkcnt++){
	//for (int chunkcnt = 0; chunkcnt < 99; chunkcnt++) {
		// Compute pointer offset corresponding to the current chunk:
		offset = chunk_size * chunkcnt;
		// Compute the number of elements to consider (to account for an incomplete final chunk):
		num2copy = min(chunk_size, Nimg - offset);
		
		// Extract the current chunk from "image_coordinates", store to "subset", then copy to GPU:
		// (Note: in the final chunk, "num2copy" could be less than "chunk_size", which means that
		//  part of "subset" is not updated and hence that part of the image chunk contains bogus data.
		//  However, as only the first "num2copy" elements will be copied to "image", this automatically
		//  corrects itself. Bear in mind though that this might result in a small number of pixels
		//  being reconstructed twice for no reason.)
		memcpy(subset + 0*chunk_size, image_coordinates + offset + 0*Nimg, num2copy * sizeof(float));	// copy x coordinates of first chunk_size image coordinates
		memcpy(subset + 1*chunk_size, image_coordinates + offset + 1*Nimg, num2copy * sizeof(float));	// copy y coordinates of first chunk_size image coordinates
		memcpy(subset + 2*chunk_size, image_coordinates + offset + 2*Nimg, num2copy * sizeof(float));	// copy z coordinates of first chunk_size image coordinates
		cudaMemcpy(image_coordinatesGPU, subset, 3 * chunk_size * sizeof(float), cudaMemcpyHostToDevice);
		
		// Perform  image reconstruction on the current chunk and wait for all GPU blocks/threads to complete:
		cudaMemset(imageGPU, 0.0f, chunk_size * sizeof(float));
		DnS_1rec_fixed_pos_GPU<<<numBlocks, blockSize>>>(rf_dataGPU, source_locationsGPU, receiver_locationGPU, image_coordinatesGPU, c, fsamp, Nsrc, Nt, chunk_size, imageGPU);
		cudaDeviceSynchronize(); // Wait for all blocks/threads to complete

		// Copy "imageGPU" to the host and update the correct chunk of "image":
		cudaMemcpy(image + offset, imageGPU, num2copy * sizeof(float), cudaMemcpyDeviceToHost);
	}


	// Copy image from GPU to CPU and clear GPU memory:
	cudaFree(rf_dataGPU);				cudaFree(source_locationsGPU);
	cudaFree(receiver_locationGPU);		cudaFree(image_coordinatesGPU);
	cudaFree(imageGPU);
	free(subset);

	return;
}

void __declspec(dllexport) DnS_1rec_at_src_GPU_chunks_interface(
	float* rf_data, float* source_locations, float* image_coordinates,
	float c, float fsamp, int Nsrc, int Nt, int Nimg,
	int* CUDAparams,
	float* image)
{
	// preallocate space on the GPU for the image, and initialise to 0:
	float* imageGPU, * rf_dataGPU, * source_locationsGPU, * image_coordinatesGPU;
	cudaMalloc(&rf_dataGPU, Nt * Nsrc * sizeof(float));	cudaMemcpy(rf_dataGPU, rf_data, Nt * Nsrc * sizeof(float), cudaMemcpyHostToDevice);
	cudaMalloc(&source_locationsGPU, 3 * Nsrc * sizeof(float));	cudaMemcpy(source_locationsGPU, source_locations, 3 * Nsrc * sizeof(float), cudaMemcpyHostToDevice);

	// Define block size and number of blocks, and define chunk size:
	int blockSize = CUDAparams[0];
	int numBlocks = CUDAparams[1];
	int chunk_size = 393216; // 2^17 * 3

	// Allocate and initialise array "subset" which will hold the chunks of image_coordinates:
	float* subset = (float*)malloc(3 * chunk_size * sizeof(float));
	memset(subset, 0.0f, 3 * chunk_size * sizeof(float));
	// Allocate GPU memory for image_coordinatesGPU and imageGPU:
	cudaMalloc(&image_coordinatesGPU, 3 * chunk_size * sizeof(float));
	cudaMalloc(&imageGPU, chunk_size * sizeof(float));

	// The loop through the chunks:
	int offset = 0, num2copy = 0;
	for (int chunkcnt = 0; chunkcnt <= Nimg / chunk_size; chunkcnt++) {
		//for (int chunkcnt = 0; chunkcnt < 99; chunkcnt++) {
			// Compute pointer offset corresponding to the current chunk:
		offset = chunk_size * chunkcnt;
		// Compute the number of elements to consider (to account for an incomplete final chunk):
		num2copy = min(chunk_size, Nimg - offset);

		// Extract the current chunk from "image_coordinates", store to "subset", then copy to GPU:
		// (Note: in the final chunk, "num2copy" could be less than "chunk_size", which means that
		//  part of "subset" is not updated and hence that part of the image chunk contains bogus data.
		//  However, as only the first "num2copy" elements will be copied to "image", this automatically
		//  corrects itself. Bear in mind though that this might result in a small number of pixels
		//  being reconstructed twice for no reason.)
		memcpy(subset + 0 * chunk_size, image_coordinates + offset + 0 * Nimg, num2copy * sizeof(float));	// copy x coordinates of first chunk_size image coordinates
		memcpy(subset + 1 * chunk_size, image_coordinates + offset + 1 * Nimg, num2copy * sizeof(float));	// copy y coordinates of first chunk_size image coordinates
		memcpy(subset + 2 * chunk_size, image_coordinates + offset + 2 * Nimg, num2copy * sizeof(float));	// copy z coordinates of first chunk_size image coordinates
		cudaMemcpy(image_coordinatesGPU, subset, 3 * chunk_size * sizeof(float), cudaMemcpyHostToDevice);

		// Perform  image reconstruction on the current chunk and wait for all GPU blocks/threads to complete:
		cudaMemset(imageGPU, 0.0f, chunk_size * sizeof(float));
		DnS_1rec_at_src_GPU<<<numBlocks, blockSize>>>(rf_dataGPU, source_locationsGPU, image_coordinatesGPU, c, fsamp, Nsrc, Nt, chunk_size, imageGPU);
		cudaDeviceSynchronize(); // Wait for all blocks/threads to complete

		// Copy "imageGPU" to the host and update the correct chunk of "image":
		cudaMemcpy(image + offset, imageGPU, num2copy * sizeof(float), cudaMemcpyDeviceToHost);
	}


	// Copy image from GPU to CPU and clear GPU memory:
	cudaFree(rf_dataGPU);				cudaFree(source_locationsGPU);
	cudaFree(image_coordinatesGPU);		cudaFree(imageGPU);
	free(subset);

	return;
}

void __declspec(dllexport) DnS_Nrec_arb_pos_GPU_chunks_interface(
	float* rf_data, float* source_locations, float* receiver_locations, float* image_coordinates,
	float c, float fsamp, int Nsrc, int Nt, int Nimg,
	int* CUDAparams,
	float* image)
{
	// preallocate space on the GPU for the image, and initialise to 0:
	float* imageGPU, * rf_dataGPU, * source_locationsGPU, * receiver_locationsGPU, * image_coordinatesGPU;
	cudaMalloc(&rf_dataGPU, Nt * Nsrc * sizeof(float));	cudaMemcpy(rf_dataGPU, rf_data, Nt * Nsrc * sizeof(float), cudaMemcpyHostToDevice);
	cudaMalloc(&source_locationsGPU, 3 * Nsrc * sizeof(float));	cudaMemcpy(source_locationsGPU, source_locations, 3 * Nsrc * sizeof(float), cudaMemcpyHostToDevice);
	cudaMalloc(&receiver_locationsGPU, 3 * Nsrc * sizeof(float));cudaMemcpy(receiver_locationsGPU, receiver_locations, 3 * Nsrc * sizeof(float), cudaMemcpyHostToDevice);

	// Define block size and number of blocks, and define chunk size:
	int blockSize = CUDAparams[0];
	int numBlocks = CUDAparams[1];
	int chunk_size = 393216; // 2^17 * 3

	// Allocate and initialise array "subset" which will hold the chunks of image_coordinates:
	float* subset = (float*)malloc(3 * chunk_size * sizeof(float));
	memset(subset, 0.0f, 3 * chunk_size * sizeof(float));
	// Allocate GPU memory for image_coordinatesGPU and imageGPU:
	cudaMalloc(&image_coordinatesGPU, 3 * chunk_size * sizeof(float));
	cudaMalloc(&imageGPU, chunk_size * sizeof(float));

	// The loop through the chunks:
	int offset = 0, num2copy = 0;
	for (int chunkcnt = 0; chunkcnt <= Nimg / chunk_size; chunkcnt++) {
		//for (int chunkcnt = 0; chunkcnt < 99; chunkcnt++) {
			// Compute pointer offset corresponding to the current chunk:
		offset = chunk_size * chunkcnt;
		// Compute the number of elements to consider (to account for an incomplete final chunk):
		num2copy = min(chunk_size, Nimg - offset);

		// Extract the current chunk from "image_coordinates", store to "subset", then copy to GPU:
		// (Note: in the final chunk, "num2copy" could be less than "chunk_size", which means that
		//  part of "subset" is not updated and hence that part of the image chunk contains bogus data.
		//  However, as only the first "num2copy" elements will be copied to "image", this automatically
		//  corrects itself. Bear in mind though that this might result in a small number of pixels
		//  being reconstructed twice for no reason.)
		memcpy(subset + 0 * chunk_size, image_coordinates + offset + 0 * Nimg, num2copy * sizeof(float));	// copy x coordinates of first chunk_size image coordinates
		memcpy(subset + 1 * chunk_size, image_coordinates + offset + 1 * Nimg, num2copy * sizeof(float));	// copy y coordinates of first chunk_size image coordinates
		memcpy(subset + 2 * chunk_size, image_coordinates + offset + 2 * Nimg, num2copy * sizeof(float));	// copy z coordinates of first chunk_size image coordinates
		cudaMemcpy(image_coordinatesGPU, subset, 3 * chunk_size * sizeof(float), cudaMemcpyHostToDevice);

		// Perform  image reconstruction on the current chunk and wait for all GPU blocks/threads to complete:
		cudaMemset(imageGPU, 0.0f, chunk_size * sizeof(float));
		DnS_Nrec_arb_pos_GPU<<<numBlocks, blockSize>>>(rf_dataGPU, source_locationsGPU, receiver_locationsGPU, image_coordinatesGPU, c, fsamp, Nsrc, Nt, chunk_size, imageGPU);
		cudaDeviceSynchronize(); // Wait for all blocks/threads to complete

		// Copy "imageGPU" to the host and update the correct chunk of "image":
		cudaMemcpy(image + offset, imageGPU, num2copy * sizeof(float), cudaMemcpyDeviceToHost);
	}


	// Copy image from GPU to CPU and clear GPU memory:
	cudaFree(rf_dataGPU);				cudaFree(source_locationsGPU);
	cudaFree(receiver_locationsGPU);	cudaFree(image_coordinatesGPU);
	cudaFree(imageGPU);
	free(subset);

	return;
}






// Delay-multiply-and-sum (DMAS):
void __declspec(dllexport) DMnS_1rec_fixed_pos_GPU_interface(
	float *rf_data,	float *source_locations,	float *receiver_location,	float *image_coordinates,
	float c, float fsamp, int Nsrc, int Nt, int Nimg,
	int *CUDAparams,
	float *image)
{
	// preallocate space on the GPU for the image, and initialise to 0:
	float *imageGPU, *rf_dataGPU, *source_locationsGPU, *receiver_locationGPU, *image_coordinatesGPU;
	cudaMalloc(&imageGPU, Nimg * sizeof(float));				cudaMemcpy(imageGPU, image, Nimg * sizeof(float), cudaMemcpyHostToDevice);
	cudaMalloc(&rf_dataGPU, Nt*Nsrc * sizeof(float));			cudaMemcpy(rf_dataGPU, rf_data, Nt*Nsrc * sizeof(float), cudaMemcpyHostToDevice);
	cudaMalloc(&source_locationsGPU, 3 * Nsrc * sizeof(float));	cudaMemcpy(source_locationsGPU, source_locations, 3 * Nsrc * sizeof(float), cudaMemcpyHostToDevice);
	cudaMalloc(&receiver_locationGPU, 3 * sizeof(float));		cudaMemcpy(receiver_locationGPU, receiver_location, 3 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMalloc(&image_coordinatesGPU, 3 * Nimg * sizeof(float));cudaMemcpy(image_coordinatesGPU, image_coordinates, 3 * Nimg * sizeof(float), cudaMemcpyHostToDevice);
	
	// Define block size and number of blocks, and call function accordingly:
	int blockSize = CUDAparams[0];
	int numBlocks = CUDAparams[1];

	DMnS_1rec_fixed_pos_GPU<<<numBlocks, blockSize>>>(rf_dataGPU, source_locationsGPU, receiver_locationGPU, image_coordinatesGPU, c, fsamp, Nsrc, Nt, Nimg, imageGPU);
	cudaStreamQuery(0); // Forces the WDDM to directly flush the kernel call to the GPU rather than first batching (and hence avoids call overhead)

	// Wait for all blocks/threads of the GPU to finish:
	cudaDeviceSynchronize();

	// Copy image from GPU to CPU and clear GPU memory:
	cudaMemcpy(image, imageGPU, Nimg * sizeof(float), cudaMemcpyDeviceToHost);
	cudaFree(imageGPU);
	cudaFree(rf_dataGPU);				cudaFree(source_locationsGPU);
	cudaFree(receiver_locationGPU);		cudaFree(image_coordinatesGPU);

	return;
}

void __declspec(dllexport) DMnS_1rec_at_src_GPU_interface(
	float* rf_data, float* source_locations, float* image_coordinates,
	float c, float fsamp, int Nsrc, int Nt, int Nimg,
	int* CUDAparams,
	float* image)
{
	// preallocate space on the GPU for the image, and initialise to 0:
	float* imageGPU, * rf_dataGPU, * source_locationsGPU, * image_coordinatesGPU;
	cudaMalloc(&imageGPU, Nimg * sizeof(float));				cudaMemcpy(imageGPU, image, Nimg * sizeof(float), cudaMemcpyHostToDevice);
	cudaMalloc(&rf_dataGPU, Nt * Nsrc * sizeof(float));			cudaMemcpy(rf_dataGPU, rf_data, Nt * Nsrc * sizeof(float), cudaMemcpyHostToDevice);
	cudaMalloc(&source_locationsGPU, 3 * Nsrc * sizeof(float));	cudaMemcpy(source_locationsGPU, source_locations, 3 * Nsrc * sizeof(float), cudaMemcpyHostToDevice);
	cudaMalloc(&image_coordinatesGPU, 3 * Nimg * sizeof(float)); cudaMemcpy(image_coordinatesGPU, image_coordinates, 3 * Nimg * sizeof(float), cudaMemcpyHostToDevice);

	// Define block size and number of blocks, and call function accordingly:
	int blockSize = CUDAparams[0];
	int numBlocks = CUDAparams[1];

	DMnS_1rec_at_src_GPU<<<numBlocks, blockSize>>>(rf_dataGPU, source_locationsGPU, image_coordinatesGPU, c, fsamp, Nsrc, Nt, Nimg, imageGPU);
	cudaStreamQuery(0); // Forces the WDDM to directly flush the kernel call to the GPU rather than first batching (and hence avoids call overhead)

	// Wait for all blocks/threads of the GPU to finish:
	cudaDeviceSynchronize();

	// Copy image from GPU to CPU and clear GPU memory:
	cudaMemcpy(image, imageGPU, Nimg * sizeof(float), cudaMemcpyDeviceToHost);
	cudaFree(imageGPU);		cudaFree(image_coordinatesGPU);
	cudaFree(rf_dataGPU);	cudaFree(source_locationsGPU);
	
	return;
}

void __declspec(dllexport) DMnS_Nrec_arb_pos_GPU_interface(
	float* rf_data, float* source_locations, float* receiver_locations, float* image_coordinates,
	float c, float fsamp, int Nsrc, int Nt, int Nimg,
	int* CUDAparams,
	float* image)
{
	// preallocate space on the GPU for the image, and initialise to 0:
	float* imageGPU, * rf_dataGPU, * source_locationsGPU, * receiver_locationsGPU, * image_coordinatesGPU;
	cudaMalloc(&imageGPU, Nimg * sizeof(float));				cudaMemcpy(imageGPU, image, Nimg * sizeof(float), cudaMemcpyHostToDevice);
	cudaMalloc(&rf_dataGPU, Nt * Nsrc * sizeof(float));			cudaMemcpy(rf_dataGPU, rf_data, Nt * Nsrc * sizeof(float), cudaMemcpyHostToDevice);
	cudaMalloc(&source_locationsGPU, 3 * Nsrc * sizeof(float));	cudaMemcpy(source_locationsGPU, source_locations, 3 * Nsrc * sizeof(float), cudaMemcpyHostToDevice);
	cudaMalloc(&receiver_locationsGPU, 3 * Nsrc * sizeof(float));		cudaMemcpy(receiver_locationsGPU, receiver_locations, 3 * Nsrc * sizeof(float), cudaMemcpyHostToDevice);
	cudaMalloc(&image_coordinatesGPU, 3 * Nimg * sizeof(float));cudaMemcpy(image_coordinatesGPU, image_coordinates, 3 * Nimg * sizeof(float), cudaMemcpyHostToDevice);

	// Define block size and number of blocks, and call function accordingly:
	int blockSize = CUDAparams[0];
	int numBlocks = CUDAparams[1];

	DMnS_Nrec_arb_pos_GPU<<<numBlocks, blockSize>>>(rf_dataGPU, source_locationsGPU, receiver_locationsGPU, image_coordinatesGPU, c, fsamp, Nsrc, Nt, Nimg, imageGPU);
	cudaStreamQuery(0); // Forces the WDDM to directly flush the kernel call to the GPU rather than first batching (and hence avoids call overhead)

	// Wait for all blocks/threads of the GPU to finish:
	cudaDeviceSynchronize();

	// Copy image from GPU to CPU and clear GPU memory:
	cudaMemcpy(image, imageGPU, Nimg * sizeof(float), cudaMemcpyDeviceToHost);
	cudaFree(imageGPU);
	cudaFree(rf_dataGPU);				cudaFree(source_locationsGPU);
	cudaFree(receiver_locationsGPU);	cudaFree(image_coordinatesGPU);

	return;
}

void __declspec(dllexport) DMnS_1rec_fixed_pos_GPU_chunks_interface(
	float *rf_data, float *source_locations, float *receiver_location, float *image_coordinates,
	float c, float fsamp, int Nsrc, int Nt, int Nimg,
	int *CUDAparams,
	float *image)
{
	// preallocate space on the GPU for the image, and initialise to 0:
	float *imageGPU, *rf_dataGPU, *source_locationsGPU, *receiver_locationGPU, *image_coordinatesGPU;
	cudaMalloc(&rf_dataGPU, Nt*Nsrc * sizeof(float));	cudaMemcpy(rf_dataGPU, rf_data, Nt*Nsrc * sizeof(float), cudaMemcpyHostToDevice);
	cudaMalloc(&source_locationsGPU, 3 * Nsrc * sizeof(float));	cudaMemcpy(source_locationsGPU, source_locations, 3 * Nsrc * sizeof(float), cudaMemcpyHostToDevice);
	cudaMalloc(&receiver_locationGPU, 3 * sizeof(float));	cudaMemcpy(receiver_locationGPU, receiver_location, 3 * sizeof(float), cudaMemcpyHostToDevice);

	// Define block size and number of blocks, and define chunk size:
	int blockSize = CUDAparams[0];
	int numBlocks = CUDAparams[1];
	int chunk_size = 5120; // 1024 * 5

	// Allocate and initialise array "subset" which will hold the chunks of image_coordinates:
	float* subset = (float*)malloc(3 * chunk_size * sizeof(float));
	memset(subset, 0.0f, 3 * chunk_size * sizeof(float));
	// Allocate GPU memory for image_coordinatesGPU and imageGPU:
	cudaMalloc(&image_coordinatesGPU, 3 * chunk_size * sizeof(float));
	cudaMalloc(&imageGPU, chunk_size * sizeof(float));

	// The loop through the chunks:
	int offset = 0, num2copy = 0;
	for (int chunkcnt = 0; chunkcnt <= Nimg / chunk_size; chunkcnt++) {
		//for (int chunkcnt = 0; chunkcnt < 99; chunkcnt++) {
			// Compute pointer offset corresponding to the current chunk:
		offset = chunk_size * chunkcnt;
		// Compute the number of elements to consider (to account for an incomplete final chunk):
		num2copy = min(chunk_size, Nimg - offset);

		// Extract the current chunk from "image_coordinates", store to "subset", then copy to GPU:
		// (Note: in the final chunk, "num2copy" could be less than "chunk_size", which means that
		//  part of "subset" is not updated and hence that part of the image chunk contains bogus data.
		//  However, as only the first "num2copy" elements will be copied to "image", this automatically
		//  corrects itself. Bear in mind though that this might result in a small number of pixels
		//  being reconstructed twice for no reason.)
		memcpy(subset + 0 * chunk_size, image_coordinates + offset + 0 * Nimg, num2copy * sizeof(float));	// copy x coordinates of first chunk_size image coordinates
		memcpy(subset + 1 * chunk_size, image_coordinates + offset + 1 * Nimg, num2copy * sizeof(float));	// copy y coordinates of first chunk_size image coordinates
		memcpy(subset + 2 * chunk_size, image_coordinates + offset + 2 * Nimg, num2copy * sizeof(float));	// copy z coordinates of first chunk_size image coordinates
		cudaMemcpy(image_coordinatesGPU, subset, 3 * chunk_size * sizeof(float), cudaMemcpyHostToDevice);

		// Perform  image reconstruction on the current chunk and wait for all GPU blocks/threads to complete:
		cudaMemset(imageGPU, 0.0f, chunk_size * sizeof(float));
		DMnS_1rec_fixed_pos_GPU<<<numBlocks, blockSize>>>(rf_dataGPU, source_locationsGPU, receiver_locationGPU, image_coordinatesGPU, c, fsamp, Nsrc, Nt, chunk_size, imageGPU);
		cudaDeviceSynchronize(); // Wait for all blocks/threads to complete

		// Copy "imageGPU" to the host and update the correct chunk of "image":
		cudaMemcpy(image + offset, imageGPU, num2copy * sizeof(float), cudaMemcpyDeviceToHost);
	}


	// Copy image from GPU to CPU and clear GPU memory:
	cudaFree(rf_dataGPU);				cudaFree(source_locationsGPU);
	cudaFree(receiver_locationGPU);		cudaFree(image_coordinatesGPU);
	cudaFree(imageGPU);
	free(subset);

	return;
}

void __declspec(dllexport) DMnS_1rec_at_src_GPU_chunks_interface(
	float* rf_data, float* source_locations, float* image_coordinates,
	float c, float fsamp, int Nsrc, int Nt, int Nimg,
	int* CUDAparams,
	float* image)
{
	// preallocate space on the GPU for the image, and initialise to 0:
	float* imageGPU, * rf_dataGPU, * source_locationsGPU, * image_coordinatesGPU;
	cudaMalloc(&rf_dataGPU, Nt * Nsrc * sizeof(float));	cudaMemcpy(rf_dataGPU, rf_data, Nt * Nsrc * sizeof(float), cudaMemcpyHostToDevice);
	cudaMalloc(&source_locationsGPU, 3 * Nsrc * sizeof(float));	cudaMemcpy(source_locationsGPU, source_locations, 3 * Nsrc * sizeof(float), cudaMemcpyHostToDevice);

	// Define block size and number of blocks, and define chunk size:
	int blockSize = CUDAparams[0];
	int numBlocks = CUDAparams[1];
	int chunk_size = 5120; // 1024 * 5

	// Allocate and initialise array "subset" which will hold the chunks of image_coordinates:
	float* subset = (float*)malloc(3 * chunk_size * sizeof(float));
	memset(subset, 0.0f, 3 * chunk_size * sizeof(float));
	// Allocate GPU memory for image_coordinatesGPU and imageGPU:
	cudaMalloc(&image_coordinatesGPU, 3 * chunk_size * sizeof(float));
	cudaMalloc(&imageGPU, chunk_size * sizeof(float));

	// The loop through the chunks:
	int offset = 0, num2copy = 0;
	for (int chunkcnt = 0; chunkcnt <= Nimg / chunk_size; chunkcnt++) {
		//for (int chunkcnt = 0; chunkcnt < 99; chunkcnt++) {
			// Compute pointer offset corresponding to the current chunk:
		offset = chunk_size * chunkcnt;
		// Compute the number of elements to consider (to account for an incomplete final chunk):
		num2copy = min(chunk_size, Nimg - offset);

		// Extract the current chunk from "image_coordinates", store to "subset", then copy to GPU:
		// (Note: in the final chunk, "num2copy" could be less than "chunk_size", which means that
		//  part of "subset" is not updated and hence that part of the image chunk contains bogus data.
		//  However, as only the first "num2copy" elements will be copied to "image", this automatically
		//  corrects itself. Bear in mind though that this might result in a small number of pixels
		//  being reconstructed twice for no reason.)
		memcpy(subset + 0 * chunk_size, image_coordinates + offset + 0 * Nimg, num2copy * sizeof(float));	// copy x coordinates of first chunk_size image coordinates
		memcpy(subset + 1 * chunk_size, image_coordinates + offset + 1 * Nimg, num2copy * sizeof(float));	// copy y coordinates of first chunk_size image coordinates
		memcpy(subset + 2 * chunk_size, image_coordinates + offset + 2 * Nimg, num2copy * sizeof(float));	// copy z coordinates of first chunk_size image coordinates
		cudaMemcpy(image_coordinatesGPU, subset, 3 * chunk_size * sizeof(float), cudaMemcpyHostToDevice);

		// Perform  image reconstruction on the current chunk and wait for all GPU blocks/threads to complete:
		cudaMemset(imageGPU, 0.0f, chunk_size * sizeof(float));
		DMnS_1rec_at_src_GPU<<<numBlocks, blockSize>>> (rf_dataGPU, source_locationsGPU, image_coordinatesGPU, c, fsamp, Nsrc, Nt, chunk_size, imageGPU);
		cudaDeviceSynchronize(); // Wait for all blocks/threads to complete

		// Copy "imageGPU" to the host and update the correct chunk of "image":
		cudaMemcpy(image + offset, imageGPU, num2copy * sizeof(float), cudaMemcpyDeviceToHost);
	}


	// Copy image from GPU to CPU and clear GPU memory:
	cudaFree(rf_dataGPU);				cudaFree(source_locationsGPU);
	cudaFree(image_coordinatesGPU);		cudaFree(imageGPU);
	free(subset);

	return;
}

void __declspec(dllexport) DMnS_Nrec_arb_pos_GPU_chunks_interface(
	float* rf_data, float* source_locations, float* receiver_locations, float* image_coordinates,
	float c, float fsamp, int Nsrc, int Nt, int Nimg,
	int* CUDAparams,
	float* image)
{
	// preallocate space on the GPU for the image, and initialise to 0:
	float* imageGPU, * rf_dataGPU, * source_locationsGPU, * receiver_locationsGPU, * image_coordinatesGPU;
	cudaMalloc(&rf_dataGPU, Nt * Nsrc * sizeof(float));	cudaMemcpy(rf_dataGPU, rf_data, Nt * Nsrc * sizeof(float), cudaMemcpyHostToDevice);
	cudaMalloc(&source_locationsGPU, 3 * Nsrc * sizeof(float));	cudaMemcpy(source_locationsGPU, source_locations, 3 * Nsrc * sizeof(float), cudaMemcpyHostToDevice);
	cudaMalloc(&receiver_locationsGPU, 3 * Nsrc * sizeof(float));	cudaMemcpy(receiver_locationsGPU, receiver_locations, 3 * Nsrc * sizeof(float), cudaMemcpyHostToDevice);

	// Define block size and number of blocks, and define chunk size:
	int blockSize = CUDAparams[0];
	int numBlocks = CUDAparams[1];
	int chunk_size = 5120; // 1024 * 5

	// Allocate and initialise array "subset" which will hold the chunks of image_coordinates:
	float* subset = (float*)malloc(3 * chunk_size * sizeof(float));
	memset(subset, 0.0f, 3 * chunk_size * sizeof(float));
	// Allocate GPU memory for image_coordinatesGPU and imageGPU:
	cudaMalloc(&image_coordinatesGPU, 3 * chunk_size * sizeof(float));
	cudaMalloc(&imageGPU, chunk_size * sizeof(float));

	// The loop through the chunks:
	int offset = 0, num2copy = 0;
	for (int chunkcnt = 0; chunkcnt <= Nimg / chunk_size; chunkcnt++) {
		//for (int chunkcnt = 0; chunkcnt < 99; chunkcnt++) {
			// Compute pointer offset corresponding to the current chunk:
		offset = chunk_size * chunkcnt;
		// Compute the number of elements to consider (to account for an incomplete final chunk):
		num2copy = min(chunk_size, Nimg - offset);

		// Extract the current chunk from "image_coordinates", store to "subset", then copy to GPU:
		// (Note: in the final chunk, "num2copy" could be less than "chunk_size", which means that
		//  part of "subset" is not updated and hence that part of the image chunk contains bogus data.
		//  However, as only the first "num2copy" elements will be copied to "image", this automatically
		//  corrects itself. Bear in mind though that this might result in a small number of pixels
		//  being reconstructed twice for no reason.)
		memcpy(subset + 0 * chunk_size, image_coordinates + offset + 0 * Nimg, num2copy * sizeof(float));	// copy x coordinates of first chunk_size image coordinates
		memcpy(subset + 1 * chunk_size, image_coordinates + offset + 1 * Nimg, num2copy * sizeof(float));	// copy y coordinates of first chunk_size image coordinates
		memcpy(subset + 2 * chunk_size, image_coordinates + offset + 2 * Nimg, num2copy * sizeof(float));	// copy z coordinates of first chunk_size image coordinates
		cudaMemcpy(image_coordinatesGPU, subset, 3 * chunk_size * sizeof(float), cudaMemcpyHostToDevice);

		// Perform  image reconstruction on the current chunk and wait for all GPU blocks/threads to complete:
		cudaMemset(imageGPU, 0.0f, chunk_size * sizeof(float));
		DMnS_Nrec_arb_pos_GPU<<<numBlocks, blockSize>>>(rf_dataGPU, source_locationsGPU, receiver_locationsGPU, image_coordinatesGPU, c, fsamp, Nsrc, Nt, chunk_size, imageGPU);
		cudaDeviceSynchronize(); // Wait for all blocks/threads to complete

		// Copy "imageGPU" to the host and update the correct chunk of "image":
		cudaMemcpy(image + offset, imageGPU, num2copy * sizeof(float), cudaMemcpyDeviceToHost);
	}


	// Copy image from GPU to CPU and clear GPU memory:
	cudaFree(rf_dataGPU);				cudaFree(source_locationsGPU);
	cudaFree(receiver_locationsGPU);	cudaFree(image_coordinatesGPU);
	cudaFree(imageGPU);
	free(subset);

	return;
}






// Short lag spatial coherence (SLSC):
// NOTE - VARIABLE "CUDAparams" NOW CONTAINS "m" and "w" FOR SLSC!
void __declspec(dllexport) SLSC_1rec_fixed_pos_GPU_interface(
	float* rf_data, float* source_locations, float* receiver_location, float* image_coordinates,
	float c, float fsamp, int Nsrc, int Nt, int Nimg,
	int* CUDAparams,
	float* image)
{
	// preallocate space on the GPU for the image, and initialise to 0:
	float* imageGPU, * rf_dataGPU, * source_locationsGPU, * receiver_locationGPU, * image_coordinatesGPU, * rf_data_squaredGPU;
	cudaMalloc(&imageGPU, Nimg * sizeof(float));				cudaMemcpy(imageGPU, image, Nimg * sizeof(float), cudaMemcpyHostToDevice);
	cudaMalloc(&rf_dataGPU, Nt * Nsrc * sizeof(float));			cudaMemcpy(rf_dataGPU, rf_data, Nt * Nsrc * sizeof(float), cudaMemcpyHostToDevice);
	cudaMalloc(&source_locationsGPU, 3 * Nsrc * sizeof(float));	cudaMemcpy(source_locationsGPU, source_locations, 3 * Nsrc * sizeof(float), cudaMemcpyHostToDevice);
	cudaMalloc(&receiver_locationGPU, 3 * sizeof(float));		cudaMemcpy(receiver_locationGPU, receiver_location, 3 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMalloc(&image_coordinatesGPU, 3 * Nimg * sizeof(float)); cudaMemcpy(image_coordinatesGPU, image_coordinates, 3 * Nimg * sizeof(float), cudaMemcpyHostToDevice);

	// Define block size and number of blocks, and call function accordingly:
	int blockSize = CUDAparams[0];
	int numBlocks = CUDAparams[1];
	int m = CUDAparams[2];
	int w = CUDAparams[3];

	// Allocate, fill, and copy to GPU an array containing rf_data^2:
	float* rf_data_squared = (float*)malloc(Nt * Nsrc * sizeof(float));
	for (int cnt = 0; cnt < Nt * Nsrc; cnt++) { rf_data_squared[cnt] = rf_data[cnt] * rf_data[cnt]; }
	cudaMalloc(&rf_data_squaredGPU, Nt * Nsrc * sizeof(float)); cudaMemcpy(rf_data_squaredGPU, rf_data_squared, Nt * Nsrc * sizeof(float), cudaMemcpyHostToDevice);

	SLSC_1rec_fixed_pos_GPU <<<numBlocks, blockSize>>>(rf_dataGPU, source_locationsGPU, receiver_locationGPU, image_coordinatesGPU, rf_data_squaredGPU, c, fsamp, Nsrc, Nt, Nimg, m, w, imageGPU);
	cudaStreamQuery(0); // Forces the WDDM to directly flush the kernel call to the GPU rather than first batching (and hence avoids call overhead)

	// Wait for all blocks/threads of the GPU to finish:
	cudaDeviceSynchronize();

	// Copy image from GPU to CPU and clear GPU memory:
	cudaMemcpy(image, imageGPU, Nimg * sizeof(float), cudaMemcpyDeviceToHost);
	cudaFree(imageGPU);
	cudaFree(rf_dataGPU);				cudaFree(source_locationsGPU);
	cudaFree(receiver_locationGPU);		cudaFree(image_coordinatesGPU);

	return;
}

void __declspec(dllexport) SLSC_1rec_at_src_GPU_interface(
	float* rf_data, float* source_locations, float* image_coordinates,
	float c, float fsamp, int Nsrc, int Nt, int Nimg,
	int* CUDAparams,
	float* image)
{
	// preallocate space on the GPU for the image, and initialise to 0:
	float* imageGPU, * rf_dataGPU, * source_locationsGPU, * image_coordinatesGPU, * rf_data_squaredGPU;
	cudaMalloc(&imageGPU, Nimg * sizeof(float));				cudaMemcpy(imageGPU, image, Nimg * sizeof(float), cudaMemcpyHostToDevice);
	cudaMalloc(&rf_dataGPU, Nt * Nsrc * sizeof(float));			cudaMemcpy(rf_dataGPU, rf_data, Nt * Nsrc * sizeof(float), cudaMemcpyHostToDevice);
	cudaMalloc(&source_locationsGPU, 3 * Nsrc * sizeof(float));	cudaMemcpy(source_locationsGPU, source_locations, 3 * Nsrc * sizeof(float), cudaMemcpyHostToDevice);
	cudaMalloc(&image_coordinatesGPU, 3 * Nimg * sizeof(float)); cudaMemcpy(image_coordinatesGPU, image_coordinates, 3 * Nimg * sizeof(float), cudaMemcpyHostToDevice);

	// Define block size and number of blocks, and call function accordingly:
	int blockSize = CUDAparams[0];
	int numBlocks = CUDAparams[1];
	int m = CUDAparams[2];
	int w = CUDAparams[3];

	// Allocate, fill, and copy to GPU an array containing rf_data^2:
	float* rf_data_squared = (float*)malloc(Nt * Nsrc * sizeof(float));
	for (int cnt = 0; cnt < Nt * Nsrc; cnt++) { rf_data_squared[cnt] = rf_data[cnt] * rf_data[cnt]; }
	cudaMalloc(&rf_data_squaredGPU, Nt * Nsrc * sizeof(float)); cudaMemcpy(rf_data_squaredGPU, rf_data_squared, Nt * Nsrc * sizeof(float), cudaMemcpyHostToDevice);

	SLSC_1rec_at_src_GPU<<<numBlocks, blockSize>>>(rf_dataGPU, source_locationsGPU, image_coordinatesGPU, rf_data_squaredGPU, c, fsamp, Nsrc, Nt, Nimg, m, w, imageGPU);
	cudaStreamQuery(0); // Forces the WDDM to directly flush the kernel call to the GPU rather than first batching (and hence avoids call overhead)

	// Wait for all blocks/threads of the GPU to finish:
	cudaDeviceSynchronize();

	// Copy image from GPU to CPU and clear GPU memory:
	cudaMemcpy(image, imageGPU, Nimg * sizeof(float), cudaMemcpyDeviceToHost);
	cudaFree(imageGPU);
	cudaFree(rf_dataGPU);				cudaFree(source_locationsGPU);
	cudaFree(image_coordinatesGPU);

	return;
}

void __declspec(dllexport) SLSC_Nrec_arb_pos_GPU_interface(
	float* rf_data, float* source_locations, float* receiver_locations, float* image_coordinates,
	float c, float fsamp, int Nsrc, int Nt, int Nimg,
	int* CUDAparams,
	float* image)
{
	// preallocate space on the GPU for the image, and initialise to 0:
	float* imageGPU, * rf_dataGPU, * source_locationsGPU, * receiver_locationsGPU, * image_coordinatesGPU, * rf_data_squaredGPU;
	cudaMalloc(&imageGPU, Nimg * sizeof(float));				cudaMemcpy(imageGPU, image, Nimg * sizeof(float), cudaMemcpyHostToDevice);
	cudaMalloc(&rf_dataGPU, Nt * Nsrc * sizeof(float));			cudaMemcpy(rf_dataGPU, rf_data, Nt * Nsrc * sizeof(float), cudaMemcpyHostToDevice);
	cudaMalloc(&source_locationsGPU, 3 * Nsrc * sizeof(float));	cudaMemcpy(source_locationsGPU, source_locations, 3 * Nsrc * sizeof(float), cudaMemcpyHostToDevice);
	cudaMalloc(&receiver_locationsGPU, 3 * Nsrc * sizeof(float));cudaMemcpy(receiver_locationsGPU, receiver_locations, 3 * Nsrc * sizeof(float), cudaMemcpyHostToDevice);
	cudaMalloc(&image_coordinatesGPU, 3 * Nimg * sizeof(float)); cudaMemcpy(image_coordinatesGPU, image_coordinates, 3 * Nimg * sizeof(float), cudaMemcpyHostToDevice);

	// Define block size and number of blocks, and call function accordingly:
	int blockSize = CUDAparams[0];
	int numBlocks = CUDAparams[1];
	int m = CUDAparams[2];
	int w = CUDAparams[3];

	// Allocate, fill, and copy to GPU an array containing rf_data^2:
	float* rf_data_squared = (float*)malloc(Nt * Nsrc * sizeof(float));
	for (int cnt = 0; cnt < Nt * Nsrc; cnt++) { rf_data_squared[cnt] = rf_data[cnt] * rf_data[cnt]; }
	cudaMalloc(&rf_data_squaredGPU, Nt * Nsrc * sizeof(float)); cudaMemcpy(rf_data_squaredGPU, rf_data_squared, Nt * Nsrc * sizeof(float), cudaMemcpyHostToDevice);

	SLSC_Nrec_arb_pos_GPU<<<numBlocks, blockSize>>>(rf_dataGPU, source_locationsGPU, receiver_locationsGPU, image_coordinatesGPU, rf_data_squaredGPU, c, fsamp, Nsrc, Nt, Nimg, m, w, imageGPU);
	cudaStreamQuery(0); // Forces the WDDM to directly flush the kernel call to the GPU rather than first batching (and hence avoids call overhead)

	// Wait for all blocks/threads of the GPU to finish:
	cudaDeviceSynchronize();

	// Copy image from GPU to CPU and clear GPU memory:
	cudaMemcpy(image, imageGPU, Nimg * sizeof(float), cudaMemcpyDeviceToHost);
	cudaFree(imageGPU);
	cudaFree(rf_dataGPU);				cudaFree(source_locationsGPU);
	cudaFree(receiver_locationsGPU);		cudaFree(image_coordinatesGPU);

	return;
}

void __declspec(dllexport) SLSC_1rec_fixed_pos_GPU_chunks_interface(
	float* rf_data, float* source_locations, float* receiver_location, float* image_coordinates,
	float c, float fsamp, int Nsrc, int Nt, int Nimg,
	int* CUDAparams,
	float* image)
{
	// preallocate space on the GPU for the image, and initialise to 0:
	float* imageGPU, * rf_dataGPU, * source_locationsGPU, * receiver_locationGPU, * image_coordinatesGPU, * rf_data_squaredGPU;
	cudaMalloc(&rf_dataGPU, Nt * Nsrc * sizeof(float));	cudaMemcpy(rf_dataGPU, rf_data, Nt * Nsrc * sizeof(float), cudaMemcpyHostToDevice);
	cudaMalloc(&source_locationsGPU, 3 * Nsrc * sizeof(float));	cudaMemcpy(source_locationsGPU, source_locations, 3 * Nsrc * sizeof(float), cudaMemcpyHostToDevice);
	cudaMalloc(&receiver_locationGPU, 3 * sizeof(float));	cudaMemcpy(receiver_locationGPU, receiver_location, 3 * sizeof(float), cudaMemcpyHostToDevice);

	// Define block size and number of blocks, and define chunk size:
	int blockSize = CUDAparams[0];
	int numBlocks = CUDAparams[1];
	int chunk_size = 10240; // 1024*10
	int m = CUDAparams[2];
	int w = CUDAparams[3];

	// Allocate and initialise array "subset" which will hold the chunks of image_coordinates:
	float* subset = (float*)malloc(3 * chunk_size * sizeof(float));
	memset(subset, 0.0f, 3 * chunk_size * sizeof(float));

	// Allocate, fill, and copy to GPU an array containing rf_data^2:
	float* rf_data_squared = (float*)malloc(Nt * Nsrc * sizeof(float));
	for (int cnt = 0; cnt < Nt * Nsrc; cnt++) { rf_data_squared[cnt] = rf_data[cnt] * rf_data[cnt]; }
	cudaMalloc(&rf_data_squaredGPU, Nt * Nsrc * sizeof(float)); cudaMemcpy(rf_data_squaredGPU, rf_data_squared, Nt * Nsrc * sizeof(float), cudaMemcpyHostToDevice);

	// Allocate GPU memory for image_coordinatesGPU, rf_data_squaredGPU, and imageGPU:
	cudaMalloc(&image_coordinatesGPU, 3 * chunk_size * sizeof(float));
	cudaMalloc(&imageGPU, chunk_size * sizeof(float));
	
	// The loop through the chunks:
	int offset = 0, num2copy = 0;
	for (int chunkcnt = 0; chunkcnt <= Nimg / chunk_size; chunkcnt++) {
		//for (int chunkcnt = 0; chunkcnt < 99; chunkcnt++) {
			// Compute pointer offset corresponding to the current chunk:
		offset = chunk_size * chunkcnt;
		// Compute the number of elements to consider (to account for an incomplete final chunk):
		num2copy = min(chunk_size, Nimg - offset);

		// Extract the current chunk from "image_coordinates", store to "subset", then copy to GPU:
		// (Note: in the final chunk, "num2copy" could be less than "chunk_size", which means that
		//  part of "subset" is not updated and hence that part of the image chunk contains bogus data.
		//  However, as only the first "num2copy" elements will be copied to "image", this automatically
		//  corrects itself. Bear in mind though that this might result in a small number of pixels
		//  being reconstructed twice for no reason.)
		memcpy(subset + 0 * chunk_size, image_coordinates + offset + 0 * Nimg, num2copy * sizeof(float));	// copy x coordinates of first chunk_size image coordinates
		memcpy(subset + 1 * chunk_size, image_coordinates + offset + 1 * Nimg, num2copy * sizeof(float));	// copy y coordinates of first chunk_size image coordinates
		memcpy(subset + 2 * chunk_size, image_coordinates + offset + 2 * Nimg, num2copy * sizeof(float));	// copy z coordinates of first chunk_size image coordinates
		cudaMemcpy(image_coordinatesGPU, subset, 3 * chunk_size * sizeof(float), cudaMemcpyHostToDevice);

		// Perform  image reconstruction on the current chunk and wait for all GPU blocks/threads to complete:
		cudaMemset(imageGPU, 0.0f, chunk_size * sizeof(float));
		SLSC_1rec_fixed_pos_GPU<<<numBlocks, blockSize>>>(rf_dataGPU, source_locationsGPU, receiver_locationGPU, image_coordinatesGPU, rf_data_squaredGPU, c, fsamp, Nsrc, Nt, chunk_size, m, w, imageGPU);
		cudaDeviceSynchronize(); // Wait for all blocks/threads to complete

		// Copy "imageGPU" to the host and update the correct chunk of "image":
		cudaMemcpy(image + offset, imageGPU, num2copy * sizeof(float), cudaMemcpyDeviceToHost);
	}

	
	// Copy image from GPU to CPU and clear GPU memory:
	cudaFree(rf_dataGPU);				cudaFree(source_locationsGPU);
	cudaFree(receiver_locationGPU);		cudaFree(image_coordinatesGPU);
	cudaFree(imageGPU);					cudaFree(rf_data_squaredGPU);
	free(subset);
	
	//for (int imgcnt = 0; imgcnt < Nimg; imgcnt++) { image[imgcnt] = 1.0f; }
	return;
}

void __declspec(dllexport) SLSC_1rec_at_src_GPU_chunks_interface(
	float* rf_data, float* source_locations, float* image_coordinates,
	float c, float fsamp, int Nsrc, int Nt, int Nimg,
	int* CUDAparams,
	float* image)
{
	// preallocate space on the GPU for the image, and initialise to 0:
	float* imageGPU, * rf_dataGPU, * source_locationsGPU, * image_coordinatesGPU, * rf_data_squaredGPU;
	cudaMalloc(&rf_dataGPU, Nt * Nsrc * sizeof(float));	cudaMemcpy(rf_dataGPU, rf_data, Nt * Nsrc * sizeof(float), cudaMemcpyHostToDevice);
	cudaMalloc(&source_locationsGPU, 3 * Nsrc * sizeof(float));	cudaMemcpy(source_locationsGPU, source_locations, 3 * Nsrc * sizeof(float), cudaMemcpyHostToDevice);

	// Define block size and number of blocks, and define chunk size:
	int blockSize = CUDAparams[0];
	int numBlocks = CUDAparams[1];
	int chunk_size = 10240; // 1024*10
	int m = CUDAparams[2];
	int w = CUDAparams[3];

	// Allocate and initialise array "subset" which will hold the chunks of image_coordinates:
	float* subset = (float*)malloc(3 * chunk_size * sizeof(float));
	memset(subset, 0.0f, 3 * chunk_size * sizeof(float));

	// Allocate, fill, and copy to GPU an array containing rf_data^2:
	float* rf_data_squared = (float*)malloc(Nt * Nsrc * sizeof(float));
	for (int cnt = 0; cnt < Nt * Nsrc; cnt++) { rf_data_squared[cnt] = rf_data[cnt] * rf_data[cnt]; }
	cudaMalloc(&rf_data_squaredGPU, Nt * Nsrc * sizeof(float)); cudaMemcpy(rf_data_squaredGPU, rf_data_squared, Nt * Nsrc * sizeof(float), cudaMemcpyHostToDevice);

	// Allocate GPU memory for image_coordinatesGPU, rf_data_squaredGPU, and imageGPU:
	cudaMalloc(&image_coordinatesGPU, 3 * chunk_size * sizeof(float));
	cudaMalloc(&imageGPU, chunk_size * sizeof(float));

	// The loop through the chunks:
	int offset = 0, num2copy = 0;
	for (int chunkcnt = 0; chunkcnt <= Nimg / chunk_size; chunkcnt++) {
		//for (int chunkcnt = 0; chunkcnt < 99; chunkcnt++) {
			// Compute pointer offset corresponding to the current chunk:
		offset = chunk_size * chunkcnt;
		// Compute the number of elements to consider (to account for an incomplete final chunk):
		num2copy = min(chunk_size, Nimg - offset);

		// Extract the current chunk from "image_coordinates", store to "subset", then copy to GPU:
		// (Note: in the final chunk, "num2copy" could be less than "chunk_size", which means that
		//  part of "subset" is not updated and hence that part of the image chunk contains bogus data.
		//  However, as only the first "num2copy" elements will be copied to "image", this automatically
		//  corrects itself. Bear in mind though that this might result in a small number of pixels
		//  being reconstructed twice for no reason.)
		memcpy(subset + 0 * chunk_size, image_coordinates + offset + 0 * Nimg, num2copy * sizeof(float));	// copy x coordinates of first chunk_size image coordinates
		memcpy(subset + 1 * chunk_size, image_coordinates + offset + 1 * Nimg, num2copy * sizeof(float));	// copy y coordinates of first chunk_size image coordinates
		memcpy(subset + 2 * chunk_size, image_coordinates + offset + 2 * Nimg, num2copy * sizeof(float));	// copy z coordinates of first chunk_size image coordinates
		cudaMemcpy(image_coordinatesGPU, subset, 3 * chunk_size * sizeof(float), cudaMemcpyHostToDevice);

		// Perform  image reconstruction on the current chunk and wait for all GPU blocks/threads to complete:
		cudaMemset(imageGPU, 0.0f, chunk_size * sizeof(float));
		SLSC_1rec_at_src_GPU<<<numBlocks, blockSize>>>(rf_dataGPU, source_locationsGPU, image_coordinatesGPU, rf_data_squaredGPU, c, fsamp, Nsrc, Nt, chunk_size, m, w, imageGPU);
		cudaDeviceSynchronize(); // Wait for all blocks/threads to complete

		// Copy "imageGPU" to the host and update the correct chunk of "image":
		cudaMemcpy(image + offset, imageGPU, num2copy * sizeof(float), cudaMemcpyDeviceToHost);
	}


	// Copy image from GPU to CPU and clear GPU memory:
	cudaFree(rf_dataGPU);				cudaFree(source_locationsGPU);
	cudaFree(image_coordinatesGPU);
	cudaFree(imageGPU);					cudaFree(rf_data_squaredGPU);
	free(subset);

	//for (int imgcnt = 0; imgcnt < Nimg; imgcnt++) { image[imgcnt] = 1.0f; }
	return;
}

void __declspec(dllexport) SLSC_Nrec_arb_pos_GPU_chunks_interface(
	float* rf_data, float* source_locations, float* receiver_locations, float* image_coordinates,
	float c, float fsamp, int Nsrc, int Nt, int Nimg,
	int* CUDAparams,
	float* image)
{
	// preallocate space on the GPU for the image, and initialise to 0:
	float* imageGPU, * rf_dataGPU, * source_locationsGPU, * receiver_locationsGPU, * image_coordinatesGPU, * rf_data_squaredGPU;
	cudaMalloc(&rf_dataGPU, Nt * Nsrc * sizeof(float));	cudaMemcpy(rf_dataGPU, rf_data, Nt * Nsrc * sizeof(float), cudaMemcpyHostToDevice);
	cudaMalloc(&source_locationsGPU, 3 * Nsrc * sizeof(float));	cudaMemcpy(source_locationsGPU, source_locations, 3 * Nsrc * sizeof(float), cudaMemcpyHostToDevice);
	cudaMalloc(&receiver_locationsGPU, 3 * Nsrc * sizeof(float));	cudaMemcpy(receiver_locationsGPU, receiver_locations, 3 * Nsrc * sizeof(float), cudaMemcpyHostToDevice);

	// Define block size and number of blocks, and define chunk size:
	int blockSize = CUDAparams[0];
	int numBlocks = CUDAparams[1];
	int chunk_size = 10240; // 1024*10
	int m = CUDAparams[2];
	int w = CUDAparams[3];

	// Allocate and initialise array "subset" which will hold the chunks of image_coordinates:
	float* subset = (float*)malloc(3 * chunk_size * sizeof(float));
	memset(subset, 0.0f, 3 * chunk_size * sizeof(float));

	// Allocate, fill, and copy to GPU an array containing rf_data^2:
	float* rf_data_squared = (float*)malloc(Nt * Nsrc * sizeof(float));
	for (int cnt = 0; cnt < Nt * Nsrc; cnt++) { rf_data_squared[cnt] = rf_data[cnt] * rf_data[cnt]; }
	cudaMalloc(&rf_data_squaredGPU, Nt * Nsrc * sizeof(float)); cudaMemcpy(rf_data_squaredGPU, rf_data_squared, Nt * Nsrc * sizeof(float), cudaMemcpyHostToDevice);

	// Allocate GPU memory for image_coordinatesGPU, rf_data_squaredGPU, and imageGPU:
	cudaMalloc(&image_coordinatesGPU, 3 * chunk_size * sizeof(float));
	cudaMalloc(&imageGPU, chunk_size * sizeof(float));

	// The loop through the chunks:
	int offset = 0, num2copy = 0;
	for (int chunkcnt = 0; chunkcnt <= Nimg / chunk_size; chunkcnt++) {
		//for (int chunkcnt = 0; chunkcnt < 99; chunkcnt++) {
			// Compute pointer offset corresponding to the current chunk:
		offset = chunk_size * chunkcnt;
		// Compute the number of elements to consider (to account for an incomplete final chunk):
		num2copy = min(chunk_size, Nimg - offset);

		// Extract the current chunk from "image_coordinates", store to "subset", then copy to GPU:
		// (Note: in the final chunk, "num2copy" could be less than "chunk_size", which means that
		//  part of "subset" is not updated and hence that part of the image chunk contains bogus data.
		//  However, as only the first "num2copy" elements will be copied to "image", this automatically
		//  corrects itself. Bear in mind though that this might result in a small number of pixels
		//  being reconstructed twice for no reason.)
		memcpy(subset + 0 * chunk_size, image_coordinates + offset + 0 * Nimg, num2copy * sizeof(float));	// copy x coordinates of first chunk_size image coordinates
		memcpy(subset + 1 * chunk_size, image_coordinates + offset + 1 * Nimg, num2copy * sizeof(float));	// copy y coordinates of first chunk_size image coordinates
		memcpy(subset + 2 * chunk_size, image_coordinates + offset + 2 * Nimg, num2copy * sizeof(float));	// copy z coordinates of first chunk_size image coordinates
		cudaMemcpy(image_coordinatesGPU, subset, 3 * chunk_size * sizeof(float), cudaMemcpyHostToDevice);

		// Perform  image reconstruction on the current chunk and wait for all GPU blocks/threads to complete:
		cudaMemset(imageGPU, 0.0f, chunk_size * sizeof(float));
		SLSC_Nrec_arb_pos_GPU<<<numBlocks, blockSize>>>(rf_dataGPU, source_locationsGPU, receiver_locationsGPU, image_coordinatesGPU, rf_data_squaredGPU, c, fsamp, Nsrc, Nt, chunk_size, m, w, imageGPU);
		cudaDeviceSynchronize(); // Wait for all blocks/threads to complete

		// Copy "imageGPU" to the host and update the correct chunk of "image":
		cudaMemcpy(image + offset, imageGPU, num2copy * sizeof(float), cudaMemcpyDeviceToHost);
	}


	// Copy image from GPU to CPU and clear GPU memory:
	cudaFree(rf_dataGPU);				cudaFree(source_locationsGPU);
	cudaFree(receiver_locationsGPU);	cudaFree(image_coordinatesGPU);
	cudaFree(imageGPU);					cudaFree(rf_data_squaredGPU);
	free(subset);

	//for (int imgcnt = 0; imgcnt < Nimg; imgcnt++) { image[imgcnt] = 1.0f; }
	return;
}

