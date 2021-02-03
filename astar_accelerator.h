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

#define TOPK 10

template<class A,class B>
struct KernelPair{
    A first;
    B second;
	
	__device__
	KernelPair(){}


	__device__
    bool operator <(KernelPair& kp) const{
        return first < kp.first;
    }


	__device__
    bool operator >(KernelPair& kp) const{
        return first > kp.first;
    }
};



__device__
dist_t device_l2_distance(value_t* d_data,idx_t idx,value_t* d_query,int qid){
	const int dim = 784;
	dist_t ret = 0;
	for(int i = 0;i < dim;++i){
		dist_t diff = d_data[idx * dim + i] - d_query[qid * dim + i];
		ret += diff * diff;
	}
	return ret;
}

__device__
dist_t device_distance(value_t* d_data,idx_t idx,value_t* d_query,int qid){
	return device_l2_distance(d_data,idx,d_query,qid);
}

__global__
void independent_search_kernel(value_t* d_data,value_t* d_query,idx_t* d_result,idx_t* d_graph,int num_query,int vertex_offset_shift){
	const int QUEUE_SIZE = TOPK;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if(tid >= num_query)
		return;
    //BloomFilter<256,8,7> bf;
    //BloomFilter<128,7,7> bf;
    BloomFilter<64,6,7> bf;
    //BloomFilter<64,6,3> bf;
    //VanillaList bf;
    KernelPair<dist_t,idx_t> q[QUEUE_SIZE + 2];
    const idx_t start_point = 0;
    dist_t d = device_distance(d_data,start_point,d_query,tid);
    int heap_size = 1;
	KernelPair<dist_t,idx_t> kp;
	kp.first = d;
	kp.second = start_point;
	smmh2::insert(q,heap_size,kp);
    bf.add(start_point);

    KernelPair<dist_t,idx_t> topk[TOPK + 1];
	int topk_heap_size = 0;
    while(heap_size > 1){
        auto now = smmh2::pop_min(q,heap_size);
		if(topk_heap_size == TOPK && topk[0].first < now.first){
        	break;
        }
       	topk[topk_heap_size++] = now;
		push_heap(topk,topk + topk_heap_size);
        if(topk_heap_size > TOPK){
        	pop_heap(topk,topk + topk_heap_size);
			--topk_heap_size;
		}
		

        idx_t offset = now.second << vertex_offset_shift;
        int degree = d_graph[offset];
        for(int i = 1;i <= degree;++i){
            auto idx = d_graph[offset + i];
            if(bf.test(idx)){
                continue;
			}
            bf.add(idx);
            dist_t d = device_distance(d_data,idx,d_query,tid);
			KernelPair<dist_t,idx_t> kp;
			kp.first = d;
			kp.second = idx;
			smmh2::insert(q,heap_size,kp);
			if(heap_size >= QUEUE_SIZE + 2){
				smmh2::pop_max(q,heap_size);
			}
        }
    }
	for(int i = 0;i < TOPK;++i){
		auto now = pop_heap(topk,topk + topk_heap_size - i);
		d_result[tid * TOPK + TOPK - 1 - i] = now.second;
	}
}

class AStarAccelerator{
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
		independent_search_kernel<<<600,128>>>(d_data,d_query,d_result,d_graph,queries.size(),vertex_offset_shift);
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

