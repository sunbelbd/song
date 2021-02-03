#pragma once

#include"data.h"
#include<vector>
#include"config.h"
#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<cuda_runtime.h>
#include"cublas_v2.h"
#include<chrono>

#include"smmh2.h"
#include"bin_heap.h"
#include"cuckoofilter.h"
#include"bloomfilter.h"
#include"blocked_bloomfilter.h"
#include"vanilla_list.h"
#include"fixhash.h"

#ifndef __ENABLE_BLOCKED_BLOOM_FILTER
	#define BlockedBloomFilter BloomFilter
#endif

#define FULL_MASK 0xffffffff
#define N_THREAD_IN_WARP 32
#define N_MULTIQUERY 1
#define CRITICAL_STEP (N_THREAD_IN_WARP/N_MULTIQUERY)
#define N_MULTIPROBE 1
#define FINISH_CNT 1

#define __ENABLE_MEASURE

struct Measure{
	unsigned long long stage1 = 0;
	unsigned long long stage2 = 0;
	unsigned long long stage3 = 0;
};

__global__
void warp_independent_search_kernel(value_t* d_data,value_t* d_query,idx_t* d_result,idx_t* d_graph,int num_query,int vertex_offset_shift
#ifdef __ENABLE_MEASURE
,Measure* measure
#endif
){
	const int QUEUE_SIZE = TOPK;
	#define DIM 960
    int bid = blockIdx.x * N_MULTIQUERY;
	const int step = N_THREAD_IN_WARP;
    int tid = threadIdx.x;
	int cid = tid / CRITICAL_STEP;
	int subtid = tid % CRITICAL_STEP;
#define BLOOM_FILTER_BIT64 8
#define BLOOM_FILTER_BIT_SHIFT 3
#define BLOOM_FILTER_NUM_HASH 7

#ifndef __ENABLE_VISITED_DEL
#define HASH_TABLE_CAPACITY (TOPK*4*16)
#else
#define HASH_TABLE_CAPACITY (TOPK*4*2)
#endif

#ifdef __DISABLE_SELECT_INSERT
#undef HASH_TABLE_CAPACITY
#define HASH_TABLE_CAPACITY (TOPK*4*16+500)
#endif

    //BloomFilter<256,8,7> bf;
    //BloomFilter<128,7,7> bf;
    //BloomFilter<64,6,7>* pbf;
    //BloomFilter<64,6,3> bf;
    //VanillaList* pbf;
#ifdef __ENABLE_FIXHASH
    FixHash<int,HASH_TABLE_CAPACITY>* pbf;
#elif __ENABLE_CUCKOO_FILTER
	#define CUCKOO_CAPACITY (BLOOM_FILTER_BIT64 * 2)
	CuckooFilter<CUCKOO_CAPACITY>* pbf;
#else
    //BloomFilter<BLOOM_FILTER_BIT64,BLOOM_FILTER_BIT_SHIFT,BLOOM_FILTER_NUM_HASH>* pbf;
    BlockedBloomFilter<BLOOM_FILTER_BIT64,BLOOM_FILTER_BIT_SHIFT,BLOOM_FILTER_NUM_HASH>* pbf;
#endif
    KernelPair<dist_t,idx_t>* q;
    KernelPair<dist_t,idx_t>* topk;
	value_t* dist_list;
	if(subtid == 0){
		dist_list = new value_t[FIXED_DEGREE * N_MULTIPROBE];
		q= new KernelPair<dist_t,idx_t>[QUEUE_SIZE + 2];
		topk = new KernelPair<dist_t,idx_t>[TOPK + 1];
    	//pbf = new BloomFilter<64,6,7>();
#ifdef __ENABLE_FIXHASH
		pbf = new FixHash<int,HASH_TABLE_CAPACITY>();
#elif __ENABLE_CUCKOO_FILTER
		pbf = new CuckooFilter<CUCKOO_CAPACITY>();
#else
    	//pbf = new BloomFilter<BLOOM_FILTER_BIT64,BLOOM_FILTER_BIT_SHIFT,BLOOM_FILTER_NUM_HASH>();
    	pbf = new BlockedBloomFilter<BLOOM_FILTER_BIT64,BLOOM_FILTER_BIT_SHIFT,BLOOM_FILTER_NUM_HASH>();
#endif
	//pbf = new VanillaList();
	}
    __shared__ int heap_size[N_MULTIQUERY];
	int topk_heap_size;

    __shared__ value_t query_point[N_MULTIQUERY][DIM];

	__shared__ int finished[N_MULTIQUERY];
	__shared__ idx_t index_list[N_MULTIQUERY][FIXED_DEGREE * N_MULTIPROBE];
	__shared__ char index_list_len[N_MULTIQUERY];
	value_t start_distance;
	__syncthreads();

	value_t tmp[N_MULTIQUERY];
	#ifdef __USE_COS_DIST
	value_t tmp_data_len[N_MULTIQUERY];
	#endif
	for(int j = 0;j < N_MULTIQUERY;++j){
		tmp[j] = 0;
		#ifdef __USE_COS_DIST
		tmp_data_len[j] = 0;
		#endif
		for(int i = tid;i < DIM;i += step){
			query_point[j][i] = d_query[(bid + j) * DIM + i];
			#ifdef __USE_L2_DIST
			tmp[j] += (query_point[j][i] - d_data[i]) * (query_point[j][i] - d_data[i]); 
			#elif __USE_IP_DIST
			tmp[j] += query_point[j][i] * d_data[i]; 
			#elif __USE_COS_DIST
			//negative cosine
			tmp[j] += query_point[j][i] * d_data[i]; 
			tmp_data_len[j] += d_data[i] * d_data[i];
			#else
			#error No distance type found. It must be [__USE_L2_DIST|__USE_IP_DIST|__USE_COS_DIST]!
			#endif
		}
		for (int offset = 16; offset > 0; offset /= 2){
				#ifdef __USE_L2_DIST
				tmp[j] += __shfl_xor_sync(FULL_MASK, tmp[j], offset);
				#elif __USE_IP_DIST
				tmp[j] += __shfl_xor_sync(FULL_MASK, tmp[j], offset);
				#elif __USE_COS_DIST
				//negative cosine
				tmp[j] += __shfl_xor_sync(FULL_MASK, tmp[j], offset);
				tmp_data_len[j] += __shfl_xor_sync(FULL_MASK, tmp_data_len[j], offset);
				#else
				#error No distance type found. It must be [__USE_L2_DIST|__USE_IP_DIST|__USE_COS_DIST]!
				#endif
		}
	}
	if(subtid == 0){
		#ifdef __USE_L2_DIST
		start_distance = tmp[cid];
		#elif __USE_IP_DIST
		start_distance = -tmp[cid];
		#elif __USE_COS_DIST
		//negative cosine
       	int sign = tmp[cid] < 0 ? 1 : -1;
		if(tmp_data_len[cid] != 0)
			start_distance = sign * tmp[cid] * tmp[cid] / tmp_data_len[cid];
		else
			start_distance = 0;
		#else
		#error No distance type found. It must be [__USE_L2_DIST|__USE_IP_DIST|__USE_COS_DIST]!
		#endif
	}
	__syncthreads();
	
	if(subtid == 0){
    	heap_size[cid] = 1;
		topk_heap_size = 0;
		finished[cid] = false;
		dist_t d = start_distance;
		KernelPair<dist_t,idx_t> kp;
		kp.first = d;
		kp.second = 0;
		smmh2::insert(q,heap_size[cid],kp);
		pbf->add(0);
	}
	__syncthreads();
    while(heap_size[cid] > 1){
#ifdef __ENABLE_MEASURE
		auto stage1_start = clock64();
#endif
		index_list_len[cid] = 0;
		int current_heap_elements = heap_size[cid] - 1;
		for(int k = 0;k < N_MULTIPROBE && k < current_heap_elements;++k){
			KernelPair<dist_t,idx_t> now;
			if(subtid == 0){
				now = smmh2::pop_min(q,heap_size[cid]);
#ifdef __ENABLE_VISITED_DEL
				pbf->del(now.second);
#endif
				if(k == 0 && topk_heap_size == TOPK && (topk[0].first <= now.first)){
					++finished[cid];
				}
			}
			__syncthreads();
			if(finished[cid] >= FINISH_CNT)
				break;
			if(subtid == 0){
				topk[topk_heap_size++] = now;
				push_heap(topk,topk + topk_heap_size);
#ifdef __ENABLE_VISITED_DEL
				pbf->add(now.second);
#endif
				if(topk_heap_size > TOPK){
#ifdef __ENABLE_VISITED_DEL
					pbf->del(topk[0].second);
#endif
					pop_heap(topk,topk + topk_heap_size);
					--topk_heap_size;
				}
				auto offset = now.second << vertex_offset_shift;
				int degree = d_graph[offset];
				for(int i = 1;i <= degree;++i){
					auto idx = d_graph[offset + i];
					if(subtid == 0){
						if(pbf->test(idx)){
							continue;
						}
#ifdef __DISABLE_SELECT_INSERT
						pbf->add(idx);
#endif
						index_list[cid][index_list_len[cid]++] = idx;
					}
				}
			}
		}
		if(finished[cid] >= FINISH_CNT)
			break;
		__syncthreads();

#ifdef __ENABLE_MEASURE
		auto stage1_end = clock64();
		if(tid == 0)
			atomicAdd(&measure->stage1,stage1_end - stage1_start);	
		auto stage2_start = clock64();
#endif
		for(int nq = 0;nq < N_MULTIQUERY;++nq){
			for(int i = 0;i < index_list_len[nq];++i){
				//TODO: replace this atomic with reduction in CUB
				value_t tmp = 0;
				#ifdef __USE_COS_DIST
				value_t tmp_data_len = 0;
				#endif
				for(int j = tid;j < DIM;j += step){
					#ifdef __USE_L2_DIST
					tmp += (query_point[nq][j] - d_data[index_list[nq][i] * DIM + j]) * (query_point[nq][j] - d_data[index_list[nq][i] * DIM + j]); 
					#elif __USE_IP_DIST
					tmp += query_point[nq][j] * d_data[index_list[nq][i] * DIM + j]; 
					#elif __USE_COS_DIST
					//negative cosine
					tmp += query_point[nq][j] * d_data[index_list[nq][i] * DIM + j]; 
					tmp_data_len += d_data[index_list[nq][i] * DIM + j] * d_data[index_list[nq][i] * DIM + j]; 
					#else
					#error No distance type found. It must be [__USE_L2_DIST|__USE_IP_DIST|__USE_COS_DIST]!
					#endif
				}
				for (int offset = 16; offset > 0; offset /= 2){
					#ifdef __USE_L2_DIST
					tmp += __shfl_xor_sync(FULL_MASK, tmp, offset);
					#elif __USE_IP_DIST
					tmp += __shfl_xor_sync(FULL_MASK, tmp, offset);
					#elif __USE_COS_DIST
					//negative cosine
					tmp += __shfl_xor_sync(FULL_MASK, tmp, offset);
					tmp_data_len += __shfl_xor_sync(FULL_MASK, tmp_data_len, offset);
					#else
					#error No distance type found. It must be [__USE_L2_DIST|__USE_IP_DIST|__USE_COS_DIST]!
					#endif
				}
				if(tid == nq * CRITICAL_STEP){
					#ifdef __USE_L2_DIST
					dist_list[i] = tmp;
					#elif __USE_IP_DIST
					dist_list[i] = -tmp;
					#elif __USE_COS_DIST
					//negative cosine
					int sign = tmp < 0 ? 1 : -1;
					if(tmp_data_len != 0)
						dist_list[i] = sign * tmp * tmp / tmp_data_len;
					else
						dist_list[i] = 0;
					#else
					#error No distance type found. It must be [__USE_L2_DIST|__USE_IP_DIST|__USE_COS_DIST]!
					#endif
				}
			}
		}

		__syncthreads();
#ifdef __ENABLE_MEASURE
		auto stage2_end = clock64();
		if(tid == 0)
			atomicAdd(&measure->stage2,stage2_end - stage2_start);	
		auto stage3_start = clock64();
#endif

		if(subtid == 0){
			for(int i = 0;i < index_list_len[cid];++i){
				dist_t d = dist_list[i];
				KernelPair<dist_t,idx_t> kp;
				kp.first = d;
				kp.second = index_list[cid][i];

				if(heap_size[cid] >= QUEUE_SIZE + 1 && q[2].first < kp.first){
					continue;
				}
#ifdef __ENABLE_MULTIPROBE_DOUBLE_CHECK
				if(pbf->test(kp.second))
					continue;
#endif
				smmh2::insert(q,heap_size[cid],kp);
#ifndef __DISABLE_SELECT_INSERT
				pbf->add(kp.second);
#endif
				if(heap_size[cid] >= QUEUE_SIZE + 2){
#ifdef __ENABLE_VISITED_DEL
					pbf->del(q[2].second);
#endif
					smmh2::pop_max(q,heap_size[cid]);
				}
			}
		}
		__syncthreads();
#ifdef __ENABLE_MEASURE
		auto stage3_end = clock64();
		if(tid == 0)
			atomicAdd(&measure->stage3,stage3_end - stage3_start);	
#endif
    }

	if(subtid == 0){
		for(int i = 0;i < TOPK;++i){
			auto now = pop_heap(topk,topk + topk_heap_size - i);
			d_result[(bid + cid) * TOPK + TOPK - 1 - i] = now.second;
		}
		delete[] q;
		delete[] topk;
    	delete pbf;
    	delete[] dist_list;
	}
}

class WarpAStarAccelerator{
private:

public:
    static void astar_multi_start_search_batch(const std::vector<std::vector<std::pair<int,value_t>>>& queries,int k,std::vector<std::vector<idx_t>>& results,value_t* h_data,idx_t* h_graph,int vertex_offset_shift,int num,int dim){
        value_t* d_data;
		value_t* d_query;
		idx_t* d_result;
		idx_t* d_graph;
		
		cudaMalloc(&d_data,sizeof(value_t) * num * dim);
		cudaMalloc(&d_graph,sizeof(idx_t) * (num << vertex_offset_shift));
		cudaMemcpy(d_data,h_data,sizeof(value_t) * num * dim,cudaMemcpyHostToDevice);
		cudaMemcpy(d_graph,h_graph,sizeof(idx_t) * (num << vertex_offset_shift),cudaMemcpyHostToDevice);

#ifdef __ENABLE_MEASURE
		Measure* d_measure;
		Measure h_measure;
		cudaMalloc(&d_measure,sizeof(Measure));
		cudaMemcpy(d_measure,&h_measure,sizeof(Measure),cudaMemcpyHostToDevice);
#endif

		auto time_begin = std::chrono::steady_clock::now();
		std::unique_ptr<value_t[]> h_query = std::unique_ptr<value_t[]>(new value_t[queries.size() * dim]);
		memset(h_query.get(),0,sizeof(value_t) * queries.size() * dim);
		for(int i = 0;i < queries.size();++i){
			for(auto p : queries[i]){
				*(h_query.get() + i * dim + p.first) = p.second;
			}
		}
		std::unique_ptr<idx_t[]> h_result = std::unique_ptr<idx_t[]>(new idx_t[queries.size() * TOPK]);

		cudaMalloc(&d_query,sizeof(value_t) * queries.size() * dim);
		cudaMalloc(&d_result,sizeof(idx_t) * queries.size() * TOPK);
		
		cudaMemcpy(d_query,h_query.get(),sizeof(value_t) * queries.size() * dim,cudaMemcpyHostToDevice);

#ifdef __ENABLE_MEASURE
		std::chrono::steady_clock::time_point mem_transfer = std::chrono::steady_clock::now();
		fprintf(stderr,"mem transfer %ld microseconds\n",std::chrono::duration_cast<std::chrono::microseconds>(mem_transfer - time_begin).count());
		std::chrono::steady_clock::time_point kernel_begin = std::chrono::steady_clock::now();
#endif

		warp_independent_search_kernel<<<queries.size()/N_MULTIQUERY,32>>>(d_data,d_query,d_result,d_graph,queries.size(),vertex_offset_shift
#ifdef __ENABLE_MEASURE
, d_measure
#endif
		);

#ifdef __ENABLE_MEASURE
		cudaDeviceSynchronize();
		std::chrono::steady_clock::time_point kernel_end = std::chrono::steady_clock::now();
		fprintf(stderr,"kernel takes %ld microseconds\n",std::chrono::duration_cast<std::chrono::microseconds>(kernel_end - kernel_begin).count());
		std::chrono::steady_clock::time_point back_begin = std::chrono::steady_clock::now();
#endif
		cudaMemcpy(h_result.get(),d_result,sizeof(idx_t) * queries.size() * TOPK,cudaMemcpyDeviceToHost);

#ifdef __ENABLE_MEASURE
		std::chrono::steady_clock::time_point back_end = std::chrono::steady_clock::now();
		fprintf(stderr,"transfer back result takes %ld microseconds\n",std::chrono::duration_cast<std::chrono::microseconds>(back_end - back_begin).count());

		cudaMemcpy(&h_measure,d_measure,sizeof(Measure),cudaMemcpyDeviceToHost);
		auto stage_sum = h_measure.stage1 + h_measure.stage2 + h_measure.stage3;
		fprintf(stderr,"stages percentage %.2f %.2f %.2f\n", h_measure.stage1 * 100.0 / stage_sum,
			h_measure.stage2 * 100.0 / stage_sum,h_measure.stage3 * 100.0 / stage_sum);
#endif
		results.clear();
		for(int i = 0;i < queries.size();++i){
			std::vector<idx_t> v(TOPK);
			for(int j = 0;j < TOPK;++j)
				v[j] = h_result[i * TOPK + j];
			results.push_back(v);
		}
		std::chrono::steady_clock::time_point time_end = std::chrono::steady_clock::now();
		fprintf(stderr,"using %ld microseconds\n",std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_begin).count());
		//printf("using %ld microseconds\n",std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_begin).count());
		cudaFree(d_data);
		cudaFree(d_query);
		cudaFree(d_result);
		cudaFree(d_graph);
    }
};

