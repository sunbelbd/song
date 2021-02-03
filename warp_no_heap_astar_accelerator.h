#pragma once

#include"data.h"
#include<vector>
#include"config.h"
#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<cuda_runtime.h>
#include"cublas_v2.h"

#include"smmh2.h"
#include"bin_heap.h"
#include"bloomfilter.h"
#include"vanilla_list.h"

#define FULL_MASK 0xffffffff
#define N_THREAD_IN_WARP 32

__global__
void warp_independent_search_kernel(value_t* d_data,value_t* d_query,idx_t* d_result,idx_t* d_graph,int num_query,int vertex_offset_shift){
	#define DIM 784
    int bid = blockIdx.x;
	const int step = N_THREAD_IN_WARP;
	if(bid >= num_query)
		return;
    int tid = threadIdx.x;
    //BloomFilter<256,8,7> bf;
    //BloomFilter<128,7,7> bf;
    //BloomFilter<64,6,7>* pbf;
    //BloomFilter<64,6,3> bf;
    //VanillaList bf;
    //KernelPair<dist_t,idx_t>* q;
    //KernelPair<dist_t,idx_t>* topk;
	value_t* dist_list;
	if(tid == 0){
		dist_list = new value_t[FIXED_DEGREE];
	//	q = new KernelPair<dist_t,idx_t>[QUEUE_SIZE + 2];
	//	topk = new KernelPair<dist_t,idx_t>[TOPK + 1];
    //	pbf = new BloomFilter<64,6,7>();
	}

    __shared__ value_t query_point[DIM];

	__shared__ KernelPair<dist_t,idx_t> now;
	__shared__ bool finished;
	value_t start_distance;
	__syncthreads();

	value_t tmp = 0;
	for(int i = tid;i < DIM;i += step){
		query_point[i] = d_query[bid * DIM + i];
		value_t diff = query_point[i] - d_data[i]; 
		tmp += diff * diff;
	}
	for (int offset = 16; offset > 0; offset /= 2)
    	tmp += __shfl_down_sync(FULL_MASK, tmp, offset);
	if(tid == 0)
		start_distance = tmp;
	__syncthreads();
	
	if(tid == 0){
		dist_t d = start_distance;
		now.first = d;
		now.second = 0;
		finished = false;
	}
	__syncthreads();
    while(!finished){
		auto offset = now.second << vertex_offset_shift;
		int degree = d_graph[offset];
		for(int i = 0;i < degree;++i){
			//TODO: replace this atomic with reduction in CUB
			value_t tmp = 0;
			for(int j = tid;j < DIM;j += step){
				value_t diff = query_point[j] - d_data[d_graph[offset + i + 1] * DIM + j]; 
				tmp += diff * diff;
			}
			for (int offset = 16; offset > 0; offset /= 2)
    			tmp += __shfl_down_sync(FULL_MASK, tmp, offset);
			if(tid == 0)
				dist_list[i] = tmp;
		}

		__syncthreads();
		if(tid == 0){
			finished = true;
			for(int i = 0;i < degree;++i){
				dist_t d = dist_list[i];
				if(now.first > d){
					now.first = d;
					now.second = d_graph[offset + i + 1];
					finished = false;
				}
			}
		}
		__syncthreads();
    }

	if(tid == 0){
		d_result[bid] = now.second;
    	delete[] dist_list;
	}
}

class WarpNoHeapAStarAccelerator{
private:

public:
    static void astar_multi_start_search_batch(const std::vector<std::vector<std::pair<int,value_t>>>& queries,int k,std::vector<std::vector<idx_t>>& results,value_t* h_data,idx_t* h_graph,int vertex_offset_shift,int num,int dim){
        value_t* d_data;
		value_t* d_query;
		idx_t* d_result;
		idx_t* d_graph;
		
		
		std::unique_ptr<value_t[]> h_query = std::unique_ptr<value_t[]>(new value_t[queries.size() * dim]);
		memset(h_query.get(),0,sizeof(value_t) * queries.size() * dim);
		for(int i = 0;i < queries.size();++i){
			for(auto p : queries[i]){
				*(h_query.get() + i * dim + p.first) = p.second;
			}
		}
		std::unique_ptr<idx_t[]> h_result = std::unique_ptr<idx_t[]>(new idx_t[queries.size() * TOPK]);

		cudaMalloc(&d_data,sizeof(value_t) * num * dim);
		cudaMalloc(&d_query,sizeof(value_t) * queries.size() * dim);
		cudaMalloc(&d_result,sizeof(idx_t) * queries.size() * TOPK);
		cudaMalloc(&d_graph,sizeof(idx_t) * (num << vertex_offset_shift));
		
		cudaMemcpy(d_data,h_data,sizeof(value_t) * num * dim,cudaMemcpyHostToDevice);
		cudaMemcpy(d_query,h_query.get(),sizeof(value_t) * queries.size() * dim,cudaMemcpyHostToDevice);
		cudaMemcpy(d_graph,h_graph,sizeof(idx_t) * (num << vertex_offset_shift),cudaMemcpyHostToDevice);
		warp_independent_search_kernel<<<queries.size(),32>>>(d_data,d_query,d_result,d_graph,queries.size(),vertex_offset_shift);
		cudaMemcpy(h_result.get(),d_result,sizeof(idx_t) * queries.size() * TOPK,cudaMemcpyDeviceToHost);
		results.clear();
		for(int i = 0;i < queries.size();++i){
			std::vector<idx_t> v(TOPK);
			for(int j = 0;j < TOPK;++j)
				v[j] = h_result[i * TOPK + j];
			results.push_back(v);
		}
    }
};

