#include<stdio.h>
#include<string.h>
#include"logger.h"
#include"parser_dense.h"
#include"parser.h"
#include"data.h"
#include"kernelgraph.h"
#include"config.h"

std::unique_ptr<Data> data;
std::unique_ptr<GraphWrapper> graph; 
int topk = 0;
int display_topk = 1;

void build_callback(idx_t idx,std::vector<std::pair<int,value_t>> point){
    data->add(idx,point);
    graph->add_vertex(idx,point);
}

std::vector<std::vector<std::pair<int,value_t>>> batch_queries;
std::vector<std::vector<idx_t>> results(ACC_BATCH_SIZE);

void flush_queries(){
	results.resize(batch_queries.size());
	const int repeat = 1; // NOTICE: You can repeat multiple times to have an average search performance
	for(int i = 0;i < repeat;++i)
	    graph->search_top_k_batch(batch_queries,topk,results);
    for(int i = 0;i < batch_queries.size();++i){
        auto& result = results[i];
        for(int i = 0;i < result.size() && i < display_topk;++i)
            printf("%zu ",result[i]);
        printf("\n");
    }
    batch_queries.clear();
}

void query_callback(idx_t idx,std::vector<std::pair<int,value_t>> point){
    batch_queries.push_back(point);
	// Uncomment the following lines to have a finer granularity batch processing
    //if(batch_queries.size() == ACC_BATCH_SIZE){
    //    flush_queries();
    //}
	/////////////////////
}




void usage(char** argv){
    printf("Usage: %s <build/test> <build_data> <query_data> <search_top_k> <row> <dim> <return_top_k> <l2/ip/cos>\n",argv[0]);
}

int main(int argc,char** argv){
    if(argc != 9){
        usage(argv);
        return 1;
    }
	// You may need to increase this parameter for some new GPUs
    cudaDeviceSetLimit(cudaLimitMallocHeapSize,800*1024*1024);
	//////////////////////
	size_t row = atoll(argv[5]);
	int dim = atoi(argv[6]);
	display_topk = atoi(argv[7]);
	std::string dist_type = argv[8];
	data = std::unique_ptr<Data>(new Data(row,dim));
	if(dist_type == "l2"){
		graph = std::unique_ptr<GraphWrapper>(new KernelFixedDegreeGraph<0>(data.get())); 
	}else if(dist_type == "ip"){
		graph = std::unique_ptr<GraphWrapper>(new KernelFixedDegreeGraph<1>(data.get())); 
	}else if(dist_type == "cos"){
		graph = std::unique_ptr<GraphWrapper>(new KernelFixedDegreeGraph<2>(data.get())); 
	}else{
		usage(argv);
		return 1;
	}
    std::string mode = std::string(argv[1]);
    topk = atoi(argv[4]);
    if(mode == "build"){
        //std::unique_ptr<ParserDense> build_parser(new ParserDense(argv[2],build_callback));
        std::unique_ptr<Parser> build_parser(new Parser(argv[2],build_callback));
        fprintf(stderr,"Writing the graph and data...");    
        data->dump();
        fprintf(stderr,"...");    
        graph->dump();
        fprintf(stderr,"done\n");    
    }else if(mode == "test"){
        fprintf(stderr,"Loading the graph and data...");    
        data->load();
        fprintf(stderr,"...");    
        graph->load();
        fprintf(stderr,"done\n");    
        //std::unique_ptr<ParserDense> query_parser(new ParserDense(argv[3],query_callback));
        std::unique_ptr<Parser> query_parser(new Parser(argv[3],query_callback));
		flush_queries();	
    }else{
        usage(argv);
        return 1;
    }
    return 0;
}
