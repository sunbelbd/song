#pragma once

#include"config.h"
#include"logger.h"
#include<stdlib.h>
#include<memory>
#include<vector>
#include<functional>



class ParserSparse{
private:
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
        for(int i = 0;i < tokens.size();i += 2){
            int index;
            value_t val;
            sscanf(buff + tokens[i] + 1,"%d",&index);
            sscanf(buff + tokens[i + 1] + 1,"%lf",&val);
            ret.push_back(std::make_pair(index,val));
        }
        return ret;
    }


public:

    ParserSparse(const char* path,std::function<void(idx_t,std::vector<std::pair<int,value_t>>)> consume) : consume(consume){
        auto fp = fopen(path,"r");
        if(fp == NULL){
            Logger::log(Logger::ERROR,"File not found at (%s)\n",path);
            exit(1);
        }
        std::unique_ptr<char[]> buff(new char[MAX_LINE]);
        std::vector<std::string> buffers;
        idx_t idx = 0;
        while(fgets(buff.get(),MAX_LINE,fp)){
            auto tokens = tokenize(buff.get());
            auto values = parse(tokens,buff.get());
            consume(idx,values);
            ++idx;
        }
        fclose(fp);
    }
    
};

