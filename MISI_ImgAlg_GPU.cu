// NOTES: 
// 1. DO NOT PRECOMPUTE DELAYS, as the memory operations required to apply these during DAS/DMAS actually
//    result in 2-fold SLOWER computation.
// 2. For chunk-wise computations, consider providing the chunk size as an input

#include "MISI_ImgAlg_GPU.h"
#include <math.h>

// Function "DnS_1rec_fixed_pos_GPU" computes, image pixel by image pixel, the delay-and-sum image amplitude. 
// Using CUDA, this computation is parallelised by distributing each image pixel to a different block/thread.
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
    
    //int index = threadIdx.x;
    //int stride = blockDim.x;
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

__global__
void DMnS_1rec_fixed_pos_GPU(
	float *rf_data,	float *source_locations,	float *receiver_location,	float *image_coordinates,
	float c, float fsamp, int Nsrc, int Nt, int Nimg,
	float *image)
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





// ==========================================================================================================


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


// Repeated 130k-pixel chunk-sized DAS reconstruction to avoid OS freezes or time-outs:
/* (Read up on WDDM to figure out why. In a nut shell: the WDDM driver model checks whether the GPU responds within a certain time frame; 
   if not, the GPU driver is reset to ensure system stability. Thing is, some image reconstructions can take longer than this wait time -> 
   run time errors result! So dividing the image reconstruction into chunks circumvents this problem at the expense of a little overhead.
*/
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
	int chunk_size = 130049; // 2^17 - 1023
	
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


// Repeated 1024-pixel chunk-sized DMAS reconstruction to avoid OS freezes or time-outs:
/* (Read up on WDDM to figure out why. In a nut shell: the WDDM driver model checks whether the GPU responds within a certain time frame;
   if not, the GPU driver is reset to ensure system stability. Thing is, some image reconstructions can take longer than this wait time ->
   run time errors result! So dividing the image reconstruction into chunks circumvents this problem at the expense of a little overhead.
*/
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
	int chunk_size = 1024;

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
