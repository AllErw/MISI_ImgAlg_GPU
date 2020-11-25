#ifndef KERNEL_H
#define KERNEL_H

#include "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.0\include\cuda_runtime.h"
#include "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.0\include\device_launch_parameters.h"

#ifdef __cplusplus
extern "C" {
#endif

	void __declspec(dllexport) DnS_1rec_fixed_pos_GPU_interface(
		float* rf_data, float* source_locations, float* receiver_location, 	float* image_coordinates, 
		float c, float fsamp, int Nsrc, int Nt, int Nimg,
		int* CUDAparams,
		float* image);

#ifdef __cplusplus
}
#endif


#ifdef __cplusplus
extern "C" {
#endif

	void __declspec(dllexport) DnS_1rec_fixed_pos_GPU_chunks_interface(
		float* rf_data, float* source_locations, float* receiver_location, float* image_coordinates, 
		float c, float fsamp, int Nsrc, int Nt, int Nimg,
		int* CUDAparams,
		float* image);

#ifdef __cplusplus
}
#endif


#ifdef __cplusplus
extern "C" {
#endif

	void __declspec(dllexport) DMnS_1rec_fixed_pos_GPU_interface(
		float* rf_data, float* source_locations, float* receiver_location, float* image_coordinates,
		float c, float fsamp, int Nsrc, int Nt, int Nimg,
		int* CUDAparams,
		float* image);

#ifdef __cplusplus
}
#endif


#ifdef __cplusplus
extern "C" {
#endif

	void __declspec(dllexport) DMnS_1rec_fixed_pos_GPU_chunks_interface(
		float* rf_data, float* source_locations, float* receiver_location, float* image_coordinates,
		float c, float fsamp, int Nsrc, int Nt, int Nimg,
		int* CUDAparams,
		float* image);

#ifdef __cplusplus
}
#endif



#ifdef __cplusplus
extern "C" {
#endif

	void __declspec(dllexport) SLSC_1rec_fixed_pos_GPU_interface(
		float* rf_data, float* source_locations, float* receiver_location, float* image_coordinates,
		float c, float fsamp, int Nsrc, int Nt, int Nimg,
		int* CUDAparams,
		float* image);

#ifdef __cplusplus
}
#endif


#ifdef __cplusplus
extern "C" {
#endif

	void __declspec(dllexport) SLSC_1rec_fixed_pos_GPU_chunks_interface(
		float* rf_data, float* source_locations, float* receiver_location, float* image_coordinates,
		float c, float fsamp, int Nsrc, int Nt, int Nimg,
		int* CUDAparams,
		float* image);

#ifdef __cplusplus
}
#endif


#endif  // KERNEL_H