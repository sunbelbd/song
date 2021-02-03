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
#define DIM 784
#define HASH_BITS 256
#define HASH_DIM (HASH_BITS / 32)

__global__
void hash_query(data_value_t* d_query,bithash_t* d_hash_matrix,value_t* d_hashquery){
    int bid = blockIdx.x;
    int tid = threadIdx.x;
	if(tid == 0){
		for(int i = 0;i < HASH_DIM;++i)
			d_hashquery[bid * HASH_DIM + i] = 0;
	}
	for(int i = 0;i < HASH_BITS;++i){
		float sum = 0;
		for(int j = tid;j < DIM;j += N_THREAD_IN_WARP){
			sum += d_query[bid * DIM + j] * d_hash_matrix[i * DIM + j];
		}
		for (int offset = N_THREAD_IN_WARP; offset > 0; offset /= 2)
    		sum += __shfl_down_sync(FULL_MASK, sum, offset);
		if(tid == 0){
			d_hashquery[bid * HASH_DIM + (i / 32)] |= (sum >= 0) << (i & 31);
		}
	}
}

__global__
void warp_independent_search_kernel(value_t* d_data,value_t* d_query,idx_t* d_result,idx_t* d_graph,int num_query,int vertex_offset_shift){
    int bid = blockIdx.x;
	const int step = N_THREAD_IN_WARP;//blockDim.x;
	if(bid >= num_query)
		return;
    int tid = threadIdx.x;
    //BloomFilter<256,8,7> bf;
    //BloomFilter<128,7,7> bf;
    //BloomFilter<64,6,7>* pbf;
    //BloomFilter<64,6,3> bf;
   // VanillaList bf;
    //KernelPair<dist_t,idx_t>* q;
    //KernelPair<dist_t,idx_t>* topk;
	value_t* dist_list;
	if(tid == 0){
		dist_list = new value_t[FIXED_DEGREE];
	//	q = new KernelPair<dist_t,idx_t>[QUEUE_SIZE + 2];
	//	topk = new KernelPair<dist_t,idx_t>[TOPK + 1];
    //	pbf = new BloomFilter<64,6,7>();
	}

    __shared__ value_t query_point[HASH_DIM];

	__shared__ KernelPair<dist_t,idx_t> now;
	__shared__ bool finished;
	///*__shared__*/ value_t* dist_list;//[N_THREAD_IN_WARP];
	value_t start_distance;
	__syncthreads();

	value_t tmp = 0;
	for(int i = tid;i < HASH_DIM;i += step){
		query_point[i] = d_query[bid * HASH_DIM + i];
		tmp += __popc(query_point[i] ^ d_data[i]);
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
			for(int j = tid;j < HASH_DIM;j += step){
				tmp += __popc(query_point[j] ^ d_data[d_graph[offset + i + 1] * HASH_DIM + j]); 
			}
			for (int offset = 16; offset > 0; offset /= 2)
    			tmp += __shfl_down_sync(FULL_MASK, tmp, offset);
			if(tid == 0){
				dist_list[i] = tmp;
			}
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

__global__
void warp_independent_search_kernel_with_heap(value_t* d_data,value_t* d_query,idx_t* d_result,idx_t* d_graph,int num_query,int vertex_offset_shift){
	const int QUEUE_SIZE = TOPK;
    int bid = blockIdx.x;
	const int step = N_THREAD_IN_WARP;
	if(bid >= num_query)
		return;
    int tid = threadIdx.x;
    //BloomFilter<256,8,7> bf;
    //BloomFilter<128,7,7> bf;
    BloomFilter<64,6,7>* pbf;
    //BloomFilter<64,6,3> bf;
   // VanillaList bf;
    KernelPair<dist_t,idx_t>* q;
    KernelPair<dist_t,idx_t>* topk;
	value_t* dist_list;
	if(tid == 0){
		dist_list = new value_t[FIXED_DEGREE];
		q = new KernelPair<dist_t,idx_t>[QUEUE_SIZE + 2];
		topk = new KernelPair<dist_t,idx_t>[TOPK + 1];
    	pbf = new BloomFilter<64,6,7>();
	}
    int heap_size;
	int topk_heap_size;

    __shared__ value_t query_point[HASH_DIM];

	__shared__ bool finished;
	__shared__ idx_t index_list[FIXED_DEGREE];
	__shared__ char index_list_len;
	value_t start_distance;
	__syncthreads();

	value_t tmp = 0;
	for(int i = tid;i < HASH_DIM;i += step){
		query_point[i] = d_query[bid * HASH_DIM + i];
		tmp += __popc(query_point[i] ^ d_data[i]);
	}
	for (int offset = 16; offset > 0; offset /= 2)
    	tmp += __shfl_down_sync(FULL_MASK, tmp, offset);
	if(tid == 0)
		start_distance = tmp;
	__syncthreads();
	
	if(tid == 0){
    	heap_size = 1;
		topk_heap_size = 0;
		finished = false;
		dist_t d = start_distance;
		KernelPair<dist_t,idx_t> kp;
		kp.first = d;
		kp.second = 0;
		smmh2::insert(q,heap_size,kp);
		pbf->add(0);
	}
	__syncthreads();
    while(heap_size > 1){
		KernelPair<dist_t,idx_t> now;
		if(tid == 0){
			now = smmh2::pop_min(q,heap_size);
			if(topk_heap_size == TOPK && (topk[0].first <= now.first)){
				finished = true;
			}
		}
		__syncthreads();
		if(finished)
			break;
		if(tid == 0){
			topk[topk_heap_size++] = now;
			push_heap(topk,topk + topk_heap_size);
			if(topk_heap_size > TOPK){
				pop_heap(topk,topk + topk_heap_size);
				--topk_heap_size;
			}
        	auto offset = now.second << vertex_offset_shift;
			index_list_len = 0;
			int degree = d_graph[offset];
			for(int i = 1;i <= degree;++i){
				auto idx = d_graph[offset + i];
				if(tid == 0){
					if(pbf->test(idx)){
						continue;
					}
					pbf->add(idx);
					index_list[index_list_len++] = idx;
				}
			}
		}
		__syncthreads();
		for(int i = 0;i < index_list_len;++i){
			//TODO: replace this atomic with reduction in CUB
			value_t tmp = 0;
			for(int j = tid;j < HASH_DIM;j += step){
				tmp += __popc(query_point[j] ^ d_data[index_list[i] * HASH_DIM + j]); 
			}
			for (int offset = 16; offset > 0; offset /= 2)
    			tmp += __shfl_down_sync(FULL_MASK, tmp, offset);
			if(tid == 0)
				dist_list[i] = tmp;

		}

		__syncthreads();
		if(tid == 0){
			for(int i = 0;i < index_list_len;++i){
				dist_t d = dist_list[i];
				KernelPair<dist_t,idx_t> kp;
				kp.first = d;
				kp.second = index_list[i];
				smmh2::insert(q,heap_size,kp);
				if(heap_size >= QUEUE_SIZE + 2){
					smmh2::pop_max(q,heap_size);
				}
			}
		}
		__syncthreads();
    }

	if(tid == 0){
		for(int i = 0;i < TOPK;++i){
			auto now = pop_heap(topk,topk + topk_heap_size - i);
			d_result[bid * TOPK + TOPK - 1 - i] = now.second;
		}
		delete[] q;
		delete[] topk;
    	delete pbf;
    	delete[] dist_list;
	}
}

class HashWarpNoHeapAStarAccelerator{
private:

public:
    static void astar_multi_start_search_batch(const std::vector<std::vector<std::pair<int,data_value_t>>>& queries,int k,std::vector<std::vector<idx_t>>& results,value_t* h_data,idx_t* h_graph,int vertex_offset_shift,int num,bithash_t* d_hash_matrix){
        value_t* d_data;
		data_value_t* d_query;
		value_t* d_hashquery;
		idx_t* d_result;
		idx_t* d_graph;
		const int dim = DIM;
		
		
		std::unique_ptr<data_value_t[]> h_query = std::unique_ptr<data_value_t[]>(new data_value_t[queries.size() * dim]);
		memset(h_query.get(),0,sizeof(data_value_t) * queries.size() * dim);
		for(int i = 0;i < queries.size();++i){
			for(auto p : queries[i]){
				*(h_query.get() + i * dim + p.first) = p.second;
			}
		}
		std::unique_ptr<idx_t[]> h_result = std::unique_ptr<idx_t[]>(new idx_t[queries.size() * TOPK]);

		cudaMalloc(&d_data,sizeof(value_t) * num * HASH_DIM);
		cudaMalloc(&d_query,sizeof(data_value_t) * queries.size() * dim);
		cudaMalloc(&d_hashquery,queries.size() * HASH_BITS / 8);
		cudaMalloc(&d_result,sizeof(idx_t) * queries.size());
		cudaMalloc(&d_graph,sizeof(idx_t) * (num << vertex_offset_shift));
		
		cudaMemcpy(d_data,h_data,sizeof(value_t) * num * HASH_DIM,cudaMemcpyHostToDevice);
		cudaMemcpy(d_query,h_query.get(),sizeof(data_value_t) * queries.size() * dim,cudaMemcpyHostToDevice);
		cudaMemcpy(d_graph,h_graph,sizeof(idx_t) * (num << vertex_offset_shift),cudaMemcpyHostToDevice);
		hash_query<<<queries.size(),32>>>(d_query,d_hash_matrix,d_hashquery);
		warp_independent_search_kernel<<<queries.size(),32>>>(d_data,d_hashquery,d_result,d_graph,queries.size(),vertex_offset_shift);
		cudaMemcpy(h_result.get(),d_result,sizeof(idx_t) * queries.size() ,cudaMemcpyDeviceToHost);
		results.clear();
		for(int i = 0;i < queries.size();++i){
			std::vector<idx_t> v(1,h_result[i]);
			results.push_back(v);
		}
    }
    
	static void astar_multi_start_search_batch_with_heap(const std::vector<std::vector<std::pair<int,data_value_t>>>& queries,int k,std::vector<std::vector<idx_t>>& results,value_t* h_data,idx_t* h_graph,int vertex_offset_shift,int num,bithash_t* d_hash_matrix){
        value_t* d_data;
		data_value_t* d_query;
		value_t* d_hashquery;
		idx_t* d_result;
		idx_t* d_graph;
		const int dim = DIM;
		
		
		std::unique_ptr<data_value_t[]> h_query = std::unique_ptr<data_value_t[]>(new data_value_t[queries.size() * dim]);
		memset(h_query.get(),0,sizeof(data_value_t) * queries.size() * dim);
		for(int i = 0;i < queries.size();++i){
			for(auto p : queries[i]){
				*(h_query.get() + i * dim + p.first) = p.second;
			}
		}
		std::unique_ptr<idx_t[]> h_result = std::unique_ptr<idx_t[]>(new idx_t[queries.size() * TOPK]);

		cudaMalloc(&d_data,sizeof(value_t) * num * HASH_DIM);
		cudaMalloc(&d_query,sizeof(data_value_t) * queries.size() * dim);
		cudaMalloc(&d_hashquery,queries.size() * HASH_BITS / 8);
		cudaMalloc(&d_result,sizeof(idx_t) * queries.size() * TOPK);
		cudaMalloc(&d_graph,sizeof(idx_t) * (num << vertex_offset_shift));
		
		cudaMemcpy(d_data,h_data,sizeof(value_t) * num * HASH_DIM,cudaMemcpyHostToDevice);
		cudaMemcpy(d_query,h_query.get(),sizeof(data_value_t) * queries.size() * dim,cudaMemcpyHostToDevice);
		cudaMemcpy(d_graph,h_graph,sizeof(idx_t) * (num << vertex_offset_shift),cudaMemcpyHostToDevice);
		hash_query<<<queries.size(),32>>>(d_query,d_hash_matrix,d_hashquery);
		warp_independent_search_kernel_with_heap<<<queries.size(),32>>>(d_data,d_hashquery,d_result,d_graph,queries.size(),vertex_offset_shift);
		cudaMemcpy(h_result.get(),d_result,sizeof(idx_t) * queries.size() * TOPK,cudaMemcpyDeviceToHost);
		
		cudaFree(d_data);
		cudaFree(d_query);
		cudaFree(d_hashquery);
		cudaFree(d_result);
		cudaFree(d_graph);
		results.clear();
		for(int i = 0;i < queries.size();++i){
			std::vector<idx_t> v(TOPK);
			for(int j = 0;j < TOPK;++j)
				v[j] = h_result[i * TOPK + j];
			results.push_back(v);
		}
    }
};

