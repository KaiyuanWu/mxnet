#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "cublas_v2.h"
#include "cuda.h"
#include "cuda_runtime_api.h"
#include <time.h>
#include "verify_loss_c.h"
#include "limits.h"
#include "float.h"
const int CUDA_NUM_THREADS = 512;
inline int GET_BLOCKS(const int N) {
  return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}


__global__ void copy_upper_to_lower(float* data, int ncols){
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < ncols*ncols; i += blockDim.x * gridDim.x){
        int idy = i/ncols;
        int idx = i%ncols;
        if(idx > idy)
            data[i] = data[idx*ncols + idy];
    }
}

__global__ void calc_loss_matrix_A_step1(float* distance, int* label,  int num_data, int num_extdata,
int k, int* triplet_index, const float margin){
   for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_data - num_extdata; i += blockDim.x * gridDim.x){
        int min_index;
        float min_value;
        int unique_label_index = (num_data - num_extdata) + label[i];
        float margin0 = margin + distance[unique_label_index*num_data + i];
        for(int j = 0; j < k; j++){
            min_index = -1;
            min_value = FLT_MAX;

            for(int t = 0; t < num_data; t++){
                if(t==i)
                    continue;
                if(distance[t*num_data + i] < min_value){
                    min_value = distance[t*num_data + i];
                    min_index = t;
                }
            }

            if(min_index != -1){
                if(label[i] != label[min_index] && min_value < margin0)
                    triplet_index[i*k + j] = min_index;
                else
                    triplet_index[i*k + j] = -1;
                distance[min_index*num_data + i] = FLT_MAX;
            }
            else
                triplet_index[i*k + j] = -1;
        }
   }
}

//Reuse the space of distance
__global__ void calc_loss_matrix_A_step2(int* label, int num_data, int num_extdata, int k, int* triplet_index, float* A){
   for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_data*num_data; i += blockDim.x * gridDim.x){
        int idx = i%num_data;
        int idy = i/num_data;
        A[i] = 0;
        if(idx == idy){
            for(int j = 0; j < num_data - num_extdata; j++){
                for(int t = 0; t < k; t++){
                    int ti = j;
                    int tj = num_data - num_extdata + label[j];
                    int tk = triplet_index[j*k + t];
                    if(tk != -1){
                        if(ti == idx)
                            A[i] += 0.75;
                        else{
                            if(tj == idx)
                                A[i] += 0.75;
                            else{
                                if(tk == idx)
                                    A[i] -= 1.;
                            }
                        }
                    }
                }
            }
        }
        else{
            if(idx <idy){
                for(int j = 0; j < num_data; j++){
                    for(int t = 0; t < k; t++){
                        int ti = j;
                        int tj = num_data - num_extdata + label[j];
                        int tk = triplet_index[j*k + t];
                        if(tk != -1){
                            if((ti == idx && tj == idy)||(tj == idx && ti == idy))
                                A[i] -= 1.25;
                            else{
                                if((ti == idx && tk == idy)||(tk == idx && ti == idy))
                                    A[i] += .5;
                                else{
                                    if((tj == idx && tk == idy)||(tk == idx && tj == idy))
                                        A[i] += .5;
                                }
                            }
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


float verify_loss_c(float* x, int* label,  int num_data, int num_feature, int num_extdata, float margin, int k,  float* grad){
    printf("num_data = %d, num_feature = %d, num_extdata = %d, margin = %f, k = %d\n",
        num_data, num_feature, num_extdata, margin, k);
    cublasStatus_t stat;
	cublasHandle_t handle;
	cudaError_t cudaStat;
	stat = cublasCreate(&handle);

	    if (stat != CUBLAS_STATUS_SUCCESS) {
                printf ("CUBLAS initialization failed\n");
                return -1.;
        }
	float *dev_x, *dev_dist, *dev_ones, *diag, *dev_diag, *dev_A, *dev_grad, *dev_triplet_dist;
	int *dev_label, *dev_triplet_index, *dev_triplet_label;
	float* dist;
	diag = new float[num_data];
    dist = new float[num_data*num_data];

	cudaError_t cudaStat1 = cudaMalloc ((void**)&dev_x, num_data*num_feature*sizeof(float));
	cudaError_t cudaStat2 = cudaMalloc ((void**)&dev_ones, num_data*sizeof(float));
	cudaError_t cudaStat3 = cudaMalloc ((void**)&dev_diag, num_data*sizeof(float));;
	cudaError_t cudaStat4 = cudaMalloc ((void**)&dev_dist, num_data*num_data*sizeof(float));
	cudaError_t cudaStat5 = cudaMalloc ((void**)&dev_A, num_data*num_data*sizeof(float));
	cudaError_t cudaStat6 = cudaMalloc ((void**)&dev_triplet_index, num_data*k*sizeof(int));
	cudaError_t cudaStat7 = cudaMalloc ((void**)&dev_label, num_data*sizeof(int));
	cudaError_t cudaStat8 = cudaMalloc ((void**)&dev_grad, num_data*num_feature*sizeof(float));
    if (cudaStat1 != cudaSuccess || cudaStat2 != cudaSuccess || cudaStat3 != cudaSuccess || cudaStat4 != cudaSuccess||
        cudaStat5 != cudaSuccess || cudaStat6 != cudaSuccess || cudaStat7 != cudaSuccess || cudaStat8 != cudaSuccess) {
                printf ("device memory allocation failed x\n");
                return -1;
    }

	cudaStat1 = cudaMemcpy (dev_x, x, num_feature*num_data*sizeof(float), cudaMemcpyHostToDevice);
	cudaStat2 = cudaMemcpy (dev_label, label, num_data*sizeof(int), cudaMemcpyHostToDevice);


	if (cudaStat1 != cudaSuccess || cudaStat2 != cudaSuccess) {
                printf ("copy cpu data to device failed x\n");
                return -1;
    }

	gpu_syrk(handle, num_data, num_feature, dev_x, dev_dist);


	cudaStat1 = cudaMemcpy (dist, dev_dist, num_data*num_data*sizeof(float),cudaMemcpyDeviceToHost);
	if (cudaStat1 != cudaSuccess) {
                printf ("copy device data to cpu failed x\n");
                return -1;
    }

	for(int i = 0; i < num_data; i++)
		diag[i] = dist[i*num_data+i];
	cudaStat1 = cudaMemcpy (dev_diag, diag, num_data*sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStat1 != cudaSuccess) {
                printf ("copy cpu data to device failed x\n");
                return -1;
    }
	for(int i = 0; i < num_data; i++)
        diag[i] = 1;
	cudaStat = cudaMemcpy (dev_ones, diag, num_data*sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStat != cudaSuccess) {
                printf ("copy cpu data to device failed x\n");
                return -1;
    }
	gpu_syr2k(handle, num_data, dev_diag, dev_ones, dev_dist);

    copy_upper_to_lower<<<GET_BLOCKS(num_data*num_data), CUDA_NUM_THREADS>>>(dev_dist,  num_data);

    //Now the dev_dist has stored the pairwise distance  in upper triangle matrix mode
    //We can start to calculate the loss and the gradient
    //Step1 : Calculate the triplets
    calc_loss_matrix_A_step1<<<GET_BLOCKS(num_data - num_extdata), CUDA_NUM_THREADS>>>(dev_dist, dev_label,  num_data, num_extdata, k,
    dev_triplet_index, margin);

    /*
    int* triplet_index = new int[num_data*k];
    cudaMemcpy(triplet_index, dev_triplet_index, num_data*k*sizeof(int), cudaMemcpyDeviceToHost);
    for(int i = 0; i<num_data; i++){
        printf("#%d:", i);
        for(int j=0; j<k; j++){
            if(triplet_index[i*k+j] != -1)
                printf("%d, ", triplet_index[i*k+j]);
        }
        printf("\n");
    }*/



    //Step2 : Accumulate the matrix A
    calc_loss_matrix_A_step2<<<GET_BLOCKS(num_data*num_data), CUDA_NUM_THREADS>>>(dev_label, num_data, num_extdata, k, dev_triplet_index, dev_A);
    copy_upper_to_lower<<<GET_BLOCKS(num_data*num_data), CUDA_NUM_THREADS>>>(dev_A,  num_data);

    /*
    float* A = new float[num_data*num_data];
    cudaMemcpy(A, dev_A, num_data*num_data*sizeof(float),cudaMemcpyDeviceToHost);
    printf("A = \n");
    for(int i = 0; i < num_data; i++){
        for(int j = 0; j < num_data; j++)
            printf("%2.3f, ", A[j*num_data + i]);
        printf("\n");
    }*/

    float alpha = 1.;
    float beta = 0.;
    cublasSgemm(handle, CUBLAS_OP_N,  CUBLAS_OP_T, num_feature, num_data,  num_data,
        &alpha, dev_x, num_feature, dev_A, num_data, &beta, dev_grad, num_feature);
    cudaMemcpy (grad, dev_grad, num_data*num_feature*sizeof(float),cudaMemcpyDeviceToHost);

    /*
    printf("grad = \n");
    for(int i = 0; i < num_data; i++){
        for(int j = 0; j < num_feature; j++)
            printf("%2.3f, ", grad[i*num_feature + j ]);
        printf("\n");
    }*/

    float loss=0.;
    cublasSdot(handle, num_data*num_feature,  dev_x, 1, dev_grad, 1,  &loss);

    /*
    printf("loss = %f\n", loss);
    loss = 0.;
    for(int i=0; i<num_data; i++){
        for(int j=0; j<num_feature; j++)
            loss += x[i]*grad[i];
    }
    printf("loss = %f\n", loss);
    */
    cudaFree (dev_dist);
	cudaFree (dev_diag);
	cudaFree (dev_ones);
	cudaFree (dev_triplet_index);
    cudaFree (dev_label);
	cudaFree (dev_x);
	cudaFree (dev_A);
	cudaFree (dev_grad);
	cublasDestroy(handle);
	delete[] diag;
	delete[] dist;
	return 0.5*loss;
}
