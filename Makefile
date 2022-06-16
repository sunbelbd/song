CXX=g++
NVCC=nvcc
DISTTYPE=__USE_COS_DIST

all : song

song.cpu : main.cc config.h graph.h parser_dense.h parser.h data.h logger.h
	$(CXX) main.cc -o song.cpu -std=c++11 -O3 -march=native

song : main.cu kernelgraph.h
	$(NVCC) -arch=sm_50 \
	-gencode=arch=compute_50,code=sm_50 \
        -gencode=arch=compute_52,code=sm_52 \
        -gencode=arch=compute_60,code=sm_60 \
        -gencode=arch=compute_61,code=sm_61 \
        -gencode=arch=compute_70,code=sm_70 \
        -gencode=arch=compute_75,code=sm_75 \
        -gencode=arch=compute_75,code=compute_75 \
        -std=c++11 main.cu -L/opt/ohpc/pub/cuda/cuda-11.2/lib64 -lcublas -g -O3 -ccbin=$(CXX) -o song -Xptxas -O3,-v \
        -D$(DISTTYPE) \
        -D__ENABLE_FIXHASH #hashtablenodelsel
        #-D__ENABLE_FIXHASH -D__ENABLE_VISITED_DEL #-D__ENABLE_FIXHASH -D__ENABLE_VISITED_DEL #-D__ENABLE_FIXHASH -D__ENABLE_VISITED_DEL #.-D__ENABLE_CUCKOO_FILTER -D_$

        #-D__ENABLE_FIXHASH -D__DISABLE_SELECT_INSERT # hashtablenodelnosel
        #-D__ENABLE_FIXHASH -D__ENABLE_VISITED_DEL  # hashtabledelsel
        #-D__ENABLE_FIXHASH #hashtablenodelsel
        #-D__ENABLE_CUCKOO_FILTER -D__ENABLE_VISITED_DEL #cuckoofilter

auto_tune_bloom : auto_tune_bloom.cc
	$(CXX) auto_tune_bloom.cc -std=c++11 -o auto_tune_bloom


