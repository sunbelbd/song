#pragma once

#include"config.h"
#include"logger.h"
#include<stdlib.h>
#include<memory>
#include<vector>
#include<functional>
#include<algorithm>

class StringRef {
private:
    char const *begin_;
    int size_;

public:
    int size() const { return size_; }

    char const *begin() const { return begin_; }

    char const *end() const { return begin_ + size_; }

    StringRef(char const *const begin, int const size)
            : begin_(begin), size_(size) {}
};

class ParserDense{
private:
    const int MAX_LINE = 10000000;
    std::function<void(idx_t,std::vector<std::pair<int,data_value_t>>)> consume;
    std::vector<StringRef> split(const char* str, char delimiter = ' ') = delete; 
    std::vector<StringRef> split(char* str, char delimiter = ' ') = delete; 
    std::vector<StringRef> split(std::string const &str, char delimiter = ' ') {
            std::vector<StringRef> result;

            enum State {
                inSpace, inToken
            };

            State state = inSpace;
            char const *pTokenBegin = 0;    // Init to satisfy compiler.
            for (const char &it : str) {
                State const newState = (it == delimiter ? inSpace : inToken);
                if (newState != state) {
                    switch (newState) {
                        case inSpace:
                            result.emplace_back(pTokenBegin, &it - pTokenBegin);
                            break;
                        case inToken:
                            pTokenBegin = &it;
                    }
                }
                state = newState;
            }
            if (state == inToken) {
                result.emplace_back(pTokenBegin, &*str.end() - pTokenBegin);
            }
            return result;
        }
        void ltrim(std::string &s) {
            s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](int ch) {
                return !std::isspace(ch);
            }));
        }
        std::string ltrim_copy(std::string s) {
            ltrim(s);
            return s;
        }

public:

    ParserDense(const char* path,std::function<void(idx_t,std::vector<std::pair<int,data_value_t>>)> consume) : consume(consume){
        auto fp = fopen(path,"r");
        if(fp == NULL){
            Logger::log(Logger::ERROR,"File not found at (%s)\n",path);
            exit(1);
        }
        std::unique_ptr<char[]> buff(new char[MAX_LINE]);
        std::vector<std::string> buffers;
        idx_t idx = 0;
        std::string field_buffer;
        while(fgets(buff.get(),MAX_LINE,fp)){
            std::vector<std::pair<int,data_value_t>> vec_sample;
            vec_sample.reserve(200);
            int index=0;
			std::string tmp_str = std::string(buff.get());
            for (const auto &it : split(tmp_str, ' ')) {
                field_buffer = ltrim_copy(std::string(it.begin(), it.end()));
                // value_t val;
                // sscanf(buff + tokens[i] + 1,"%d",&index);
                // sscanf(field_buffer,"%lf",&val);
                vec_sample.push_back(std::make_pair(index,atof(field_buffer.c_str())));
                index++;
                
            }
            consume(idx,vec_sample);
            ++idx;
        }
        fclose(fp);
    }
};
