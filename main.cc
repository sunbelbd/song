#include<stdio.h>
#include<string.h>
#include <iostream>
#include <fstream>
#include <queue>
#include <chrono>
#include <unordered_set>
#include <unordered_map>
#include"logger.h"
#include"parser_dense.h"
#include"parser.h"
#include"data.h"
#include"graph.h"
#include<stdlib.h>
#include<memory>
#include<vector>
#include<functional>
#include"bithash.h"
#include <sstream>

std::unique_ptr<Data> data;
std::unique_ptr<GraphWrapper> graph; 
int topk = 0;
int display_topk = 1;
BitHash bithash;

std::vector<std::vector<idx_t>> resultss;

void build_callback(idx_t idx,std::vector<std::pair<int,value_t>> point){
	#ifdef __ENABLE_HASH
	point = bithash.hash2kv(point);
	#endif
    data->add(idx,point);
    graph->add_vertex(idx,point);
}

void query_callback(idx_t idx,std::vector<std::pair<int,value_t>> point){
	#ifdef __ENABLE_HASH
    //std::cout<<"here"<<std::endl;
	point = bithash.hash2kv(point);
	#endif
    resultss.push_back({});
    graph->search_top_k(point,topk,resultss[idx]);
    //for(int i = 0;i < resultss[idx].size() && i < display_topk;++i)
        //printf("%zu ",resultss[idx][i]);
    //printf("\n");
}

void flush_queries(char* groundtruth){
    std::string gt(groundtruth);
    std::ifstream fptr(gt);
    int counter=0;
    size_t num;
    for(int i = 0;i < resultss.size();++i){
        auto& result = resultss[i];
        std::unordered_set<size_t> sett;
        for(int i = 0;i < result.size() && i < display_topk;++i) {
            sett.insert(result[i]);
            printf("%zu ",result[i]);
        }
        printf("\n");

        std::string line;
        getline(fptr, line);
        std::stringstream ss(line);
        for(int j = 0;j < display_topk;++j){    // Does not depend on the truthsize
            ss>>num;
            if(sett.find(num)!=sett.end())
                counter++;
        }
        //std::cout<<counter<<std::endl;
    }
    //fprintf(stderr,"Accuracy = %f %%\n",counter*100.0/(display_topk*batch_queries.size()));
    fprintf(stdout,"Accuracy = %f %%\n",counter*100.0/(display_topk*resultss.size()));
    resultss.clear();
}

void usage(char** argv){
    printf("Usage: %s <build/test> <build_data> <query_data> <search_top_k> <row> <dim> <return_top_k> <l2/ip/cos>\n",argv[0]);
}

int main(int argc,char** argv){
    if(argc != 9 && argc!=10){
        usage(argv);
        return 1;
    }
	size_t row = atoll(argv[5]);
	int dim = atoi(argv[6]);
	display_topk = atoi(argv[7]);
	std::string dist_type = argv[8];
	
	#ifdef __ENABLE_HASH
	const int HASH_DIM = 128;
	bithash = BitHash(dim,HASH_DIM);    // (16,512) //Commented by Saim
	int old_dim = dim;
	dim = HASH_DIM / sizeof(value_t) / 8;
	dist_type = "hash";
	#endif

	data = std::unique_ptr<Data>(new Data(row,dim));


	if(dist_type == "l2"){
		graph = std::unique_ptr<GraphWrapper>(new FixedDegreeGraph<0>(data.get())); 
	}else if(dist_type == "ip"){
		graph = std::unique_ptr<GraphWrapper>(new FixedDegreeGraph<1>(data.get())); 
	}else if(dist_type == "cos"){
		graph = std::unique_ptr<GraphWrapper>(new FixedDegreeGraph<2>(data.get())); 
	}else if(dist_type == "hash"){
		graph = std::unique_ptr<GraphWrapper>(new FixedDegreeGraph<3>(data.get())); 
	}else{
		usage(argv);
		return 1;
	}
    std::string mode = std::string(argv[1]);
    topk = atoi(argv[4]);
    if(mode == "build"){
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
        std::unique_ptr<Parser> query_parser(new Parser(argv[3],query_callback));
        flush_queries(argv[9]);
    }else{
        usage(argv);
        return 1;
    }


    return 0;
}
