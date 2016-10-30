#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "cublas_v2.h"
#include "cuda.h"
#include "cuda_runtime_api.h"
#include <time.h>
#include "pairwise_distance.h"
const int CUDA_NUM_THREADS = 512;
inline int GET_BLOCKS(const int N) {
  return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}

__global__ void calc_loss_matrix_A_step1(const float* distance, int* label,  int num_data, int num_extdata,
int k, int* triplet_index, const float margin){
   for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_data; i += blockDim.x * gridDim.x){
        int min_index;
        float min_value;
        int unique_label_index = (num_data - num_extdata) + label[i];
        float margin0 += distance[unique_label_index*num_data + i];
        for(int j = 0; j < k; j++){
            min_index = -1;
            min_value = FLT_MAX;

            for(int t = 0; t < i; t++){
                if(distance[t*num_data + i] < min_value){
                    min_value = distance[t*num_data + i];
                    min_index = t;
                }
            }
            for(int t = i+1; t < num_data; t++){
                if(distance[i*num_data + t] < min_value){
                    min_value = distance[t*num_data + i];
                    min_index = t;
                }
            }
            if(min_index != -1){
                if(label[i] != label[min_index] && min_value < margin0)
                    triplet_index[i*k + j] = min_index;
                else
                    triplet_index[i*k + j] = -1;

                if(min_index < i)
                    distance[t*num_data + i] = FLT_MAX;
                else
                    distance[i*num_data + t] = FLT_MAX;
            }
        }
   }
}

//Reuse the space of distance
__global__ void calc_loss_matrix_A_step2(int* label, int num_data, int num_extdata, int k, int* triplet_index, float* A){
   for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_data; i += blockDim.x * gridDim.x){
        int idx = i/num_data;
        int idy = i%num_data;
        A[i] = 0;
        if(idx == idy){
            for(int j = 0; j < num_data - num_extdata; j++){
                for(int t = 0; t < k; t++){
                    if(triplet_index[j*k + t] != -1){
                        if(j == idx)
                            A[i] += 0.75;
                        if(num_data - num_extdata + label[j] == idx)
                            A[i] += 0.75;
                        if(triplet_index[j*k + t] == idx)
                            A[i] -= 1.;
                    }
                }
            }
        }
        else{
            if(idx <idy){
                for(int j = 0; j < num_data; j++){
                    for(int t = 0; t < k; t++){
                        if(triplet_index[j*k + t] != -1){
                            if(j == idx)
                                A[i] += 0.75;
                            if(num_data - num_extdata + label[j] == idx)
                                A[i] += 0.75;
                            if(triplet_index[j*k + t] == idx)
                                A[i] -= 1.;
                        }
                    }
                }
            }
        }
   }
}

void gpu_syrk(cublasHandle_t handle, const int nrows, const int ncols, const float* x,
    float* out) {
	const float alpha=1.f;
	const float beta =0.f;
  	cublasSsyrk(handle, CUBLAS_FILL_MODE_UPPER,CUBLAS_OP_T, nrows, ncols, &alpha, x, ncols,  &beta, out, nrows);
}


void gpu_syr2k(cublasHandle_t handle, const int nrows, const float* diag, const float* ones, float* dist){
	const float alpha = 1.f;
    const float beta = -2.f;
	cublasSsyr2k(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T, nrows, 1, &alpha, diag, 1, ones, 1, &beta, dist, nrows);
}


int verify_loss_c(float* x, int* label,  int num_data, int num_feature, int num_extdata, float margin, int k,  float* grad, float* loss){
    cublasStatus_t stat;
	cublasHandle_t handle;
	cudaError_t cudaStat;
	stat = cublasCreate(&handle);
	float *dev_x, *dev_dist, *dev_ones, *diag, *dev_diag, *dev_A, *dev_grad;
	int *dev_label, *dev_triplet_index;
	diag = new float[num_data];

	cudaStat = cudaMalloc ((void**)&dev_x, num_data*num_feature*sizeof(float));
	cudaStat = cudaMalloc ((void**)&dev_ones, num_data*sizeof(float));
	cudaStat = cudaMalloc ((void**)&dev_diag, num_data*sizeof(float));;
	cudaStat = cudaMalloc ((void**)&dev_dist, num_data*num_data*sizeof(float));
	cudaStat = cudaMalloc ((void**)&dev_A, num_data*num_data*sizeof(float));
	cudaStat = cudaMalloc ((void**)&dev_triplet_index, num_data*k*sizeof(int));
	cudaStat = cudaMalloc ((void**)&dev_label, num_data*sizeof(int));
	cudaStat = cudaMalloc ((void**)&dev_grad, num_data*num_feature*sizeof(float));


	cudaStat = cudaMemcpy (dev_x, x, num_feature*num_data*sizeof(float), cudaMemcpyHostToDevice);
	cudaStat = cudaMemcpy (dev_label, label, num_data*sizeof(int), cudaMemcpyHostToDevice);

	gpu_syrk(handle, num_data, num_feature, dev_x, dev_dist);

	cudaStat = cudaMemcpy (dist, dev_dist, num_data*num_data*sizeof(float),cudaMemcpyDeviceToHost);

	for(int i = 0; i < num_data; i++)
		diag[i] = dist[i*num_data+i];
	cudaStat = cudaMemcpy (dev_diag, diag, num_data*sizeof(float), cudaMemcpyHostToDevice);
	for(int i = 0; i < num_data; i++)
        diag[i] = 1;
	cudaStat = cudaMemcpy (dev_ones, diag, num_data*sizeof(float), cudaMemcpyHostToDevice);
	gpu_syr2k(handle, num_data, dev_diag, dev_ones, dev_dist);

    //Now the dev_dist has stored the pairwise distance  in upper triangle matrix
    //We can start to calculate the loss and the gradient
    //Step1 : Calculate the triplets
    calc_loss_matrix_A_step1<<<GET_BLOCKS(num_data - num_extdata), CUDA_NUM_THREADS>>>(dev_dist, dev_label,  num_data, num_extdata, k, dev_triplet_index, margin);
    //Step2 : Accumulate the matrix A
    calc_loss_matrix_A_step2<<<GET_BLOCKS(num_data*num_data), CUDA_NUM_THREADS>>>(dev_label, num_data, num_extdata, k, dev_triplet_index, dev_A);

    cudaFree (dev_dist);
	cudaFree (dev_diag);
	cudaFree (dev_ones);
	cudaFree (dev_triplet_index);
    cudaFree (dev_label);

    float alpha = 1.;
    float beta = 0.;
    cublasSsymm(handle, CUBLAS_FILL_MODE_UPPER, n, &alpha, dev_A, lda, dev_x, 1, &beta, dev_grad)
    cudaMemcpy (grad, dev_grad, num_data*num_feature*sizeof(float),cudaMemcpyDeviceToHost);
    cublasSdot(handle, dev_grad,  dev_x, loss);

	cudaFree (dev_x);
	cudaFree (dev_A);
	cudaFree (dev_grad)

	delete[] diag;
	cublasDestroy(handle);
	return true;
}
