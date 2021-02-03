#!/usr/bin/env bash

make auto_tune_bloom || true

mkdir build_template || true

cp *.h build_template
cp *.cc build_template
cp *.cu build_template
cp Makefile build_template
cp fill_parameters.sh build_template
cp auto_tune_bloom build_template

cd build_template

sed -i -e "s/#define N_MULTIQUERY.*/#define N_MULTIQUERY PLACE_HOLDER_N_MULTIQUERY/g" *.h
sed -i -e "s/#define N_MULTIPROBE.*/#define N_MULTIPROBE PLACE_HOLDER_N_MULTIPROBE/g" *.h
sed -i -e "s/#define FINISH_CNT.*/#define FINISH_CNT PLACE_HOLDER_FINISH_CNT/g" *.h

sed -i -e "s/#define BLOOM_FILTER_BIT64.*/#define BLOOM_FILTER_BIT64 PLACE_HOLDER_BLOOM_FILTER_BIT64/g" *.h
sed -i -e "s/#define BLOOM_FILTER_BIT_SHIFT.*/#define BLOOM_FILTER_BIT_SHIFT PLACE_HOLDER_BLOOM_FILTER_BIT_SHIFT/g" *.h
sed -i -e "s/#define BLOOM_FILTER_NUM_HASH.*/#define BLOOM_FILTER_NUM_HASH PLACE_HOLDER_BLOOM_FILTER_NUM_HASH/g" *.h
sed -i -e "s/#define HASH_BITS.*/#define HASH_BITS PLACE_HOLDER_HASH_BITS/g" *.h
sed -i -e "s/#define HASH_BITS.*/#define HASH_BITS PLACE_HOLDER_HASH_BITS/g" *.cu
sed -i -e "s/#define HASH_BITS.*/#define HASH_BITS PLACE_HOLDER_HASH_BITS/g" *.cc
sed -i -e "s/#define N_TABLES.*/#define N_TABLES PLACE_HOLDER_N_TABLES/g" *.h
sed -i -e "s/#define N_TABLES.*/#define N_TABLES PLACE_HOLDER_N_TABLES/g" *.cu
sed -i -e "s/#define N_TABLES.*/#define N_TABLES PLACE_HOLDER_N_TABLES/g" *.cc
sed -i -e "s/#define TOPK.*/#define TOPK PLACE_HOLDER_TOPK/g" *.h
sed -i -e "s/#define TOPK.*/#define TOPK PLACE_HOLDER_TOPK/g" *.cu
sed -i -e "s/#define TOPK.*/#define TOPK PLACE_HOLDER_TOPK/g" *.cc
sed -i -e "s/#define DIM.*/#define DIM PLACE_HOLDER_DIM/g" *.h
sed -i -e "s/#define DIM.*/#define DIM PLACE_HOLDER_DIM/g" *.cu
sed -i -e "s/#define DIM.*/#define DIM PLACE_HOLDER_DIM/g" *.cc

