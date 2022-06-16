#pragma once

#include"config.h"
#include"logger.h"
#include<stdlib.h>
#include<memory>
#include<vector>
#include<functional>
#include<iostream>



class Parser{
private:
	const int ONE_BASED_LIBSVM = 1;
    const int MAX_LINE = 10000000;
    std::function<void(idx_t,std::vector<std::pair<int,value_t>>)> consume;

    std::vector<int> tokenize(char* buff){
        std::vector<int> ret;
        int i = 0;
        while(*(buff + i) != '\0'){
            if(*(buff + i) == ':' || *(buff + i) == ' ')
                ret.push_back(i);
            ++i;
        }
        return ret;
    }

    std::vector<std::pair<int,value_t>> parse(std::vector<int> tokens,char* buff){
        std::vector<std::pair<int,value_t>> ret;
        ret.reserve(tokens.size() / 2);
        for(int i = 0;i + 1 < tokens.size();i += 2){
            int index;
            value_t val;
            sscanf(buff + tokens[i] + 1,"%d",&index);
			index -= ONE_BASED_LIBSVM;
			double tmp;
            sscanf(buff + tokens[i + 1] + 1,"%lf",&tmp);
			val = tmp;
            ret.push_back(std::make_pair(index,val));
        }
        return ret;
    }


public:

    Parser(const char* path,std::function<void(idx_t,std::vector<std::pair<int,value_t>>)> consume) : consume(consume){
        auto fp = fopen(path,"r");
        if(fp == NULL){
            Logger::log(Logger::ERROR,"File not found at (%s)\n",path);
            exit(1);
        }
        std::unique_ptr<char[]> buff(new char[MAX_LINE]);
        std::vector<std::string> buffers;
        idx_t idx = 0;
        while(fgets(buff.get(),MAX_LINE,fp)){
//            for(int i=0;i<MAX_LINE;i++)
//                printf("%c",*(buff.get()+i));
            auto tokens = tokenize(buff.get());
            auto values = parse(tokens,buff.get());
//            for(auto p:values)
//                std::cout<<p.first<<"**"<<p.second<<std::endl;
            consume(idx,values);
            ++idx;

        }
        fclose(fp);
    }
    
};
