#!/usr/bin/env bash

if [ "$#" -ne 3 ]; then
	echo "Usage: $0 <pq_size> <dim> <cos/l2/ip>" >&2
	echo "For example: $0 100 26 cos" >&2
	exit 1
fi


hash_bits=128 #Not used in default
n_tables=10   #Not used in default
topk=$1
dim=$2
dis=$3


multi_query=1
multi_probe=1
finish_cnt=1

output=`./auto_tune_bloom $topk`
bf_bit64=`echo ${output} | awk '{print $2}'`
bf_bit_shift=`echo ${output} | awk '{print $3}'`
bf_num_hash=`echo ${output} | awk '{print $5}'`

mkdir build_template/tmp || true

cp build_template/*.h build_template/tmp
cp build_template/*.cc build_template/tmp
cp build_template/*.cu build_template/tmp
cp build_template/Makefile build_template/tmp

cd build_template/tmp

sed -i -e "s/PLACE_HOLDER_N_MULTIQUERY/${multi_query}/g" *.h
sed -i -e "s/PLACE_HOLDER_N_MULTIPROBE/${multi_probe}/g" *.h
sed -i -e "s/PLACE_HOLDER_FINISH_CNT/${finish_cnt}/g" *.h

sed -i -e "s/PLACE_HOLDER_BLOOM_FILTER_BIT64/${bf_bit64}/g" *.h
sed -i -e "s/PLACE_HOLDER_BLOOM_FILTER_BIT_SHIFT/${bf_bit_shift}/g" *.h
sed -i -e "s/PLACE_HOLDER_BLOOM_FILTER_NUM_HASH/${bf_num_hash}/g" *.h
sed -i -e "s/PLACE_HOLDER_HASH_BITS/${hash_bits}/g" *.h
sed -i -e "s/PLACE_HOLDER_HASH_BITS/${hash_bits}/g" *.cu
sed -i -e "s/PLACE_HOLDER_HASH_BITS/${hash_bits}/g" *.cc
sed -i -e "s/PLACE_HOLDER_N_TABLES/${n_tables}/g" *.h
sed -i -e "s/PLACE_HOLDER_N_TABLES/${n_tables}/g" *.cu
sed -i -e "s/PLACE_HOLDER_N_TABLES/${n_tables}/g" *.cc
sed -i -e "s/PLACE_HOLDER_TOPK/${topk}/g" *.h
sed -i -e "s/PLACE_HOLDER_TOPK/${topk}/g" *.cu
sed -i -e "s/PLACE_HOLDER_TOPK/${topk}/g" *.cc
sed -i -e "s/PLACE_HOLDER_DIM/${dim}/g" *.h
sed -i -e "s/PLACE_HOLDER_DIM/${dim}/g" *.cu
sed -i -e "s/PLACE_HOLDER_DIM/${dim}/g" *.cc

if [ "${dis}" = "cos" ]; then
	make song
elif [ "${dis}" = "l2" ]; then
	make song DISTTYPE=__USE_L2_DIST
elif [ "${dis}" = "ip" ]; then
	make song DISTTYPE=__USE_IP_DIST
else
	echo "Usage: $0 <pq_size> <dim> <cos/l2/ip>" >&2
	echo "For example: $0 100 26 cos" >&2
	exit 1
fi

cp song ../..
