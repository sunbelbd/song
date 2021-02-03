#pragma once

#include<memory>
#include<vector>
#include"config.h"

#define _SCALE_WORLD_DENSE_DATA

#ifdef _SCALE_WORLD_DENSE_DATA
//dense data
class Data{
private:
    std::unique_ptr<value_t[]> data;
    size_t num;
    size_t curr_num = 0;
    int dim;

public:
    Data(size_t num, int dim) : num(num),dim(dim){
        data = std::unique_ptr<value_t[]>(new value_t[num * dim]);
        memset(data.get(),0,sizeof(value_t) * num * dim);
    }
    
    value_t* get(idx_t idx) const{
        return data.get() + idx * dim;
    }

    template<class T>
    dist_t l2_distance(idx_t a,T& v) const{
        auto pa = get(a);
        dist_t ret = 0;
        for(int i = 0;i < dim;++i){
            auto diff = *(pa + i) - v[i];
            ret += diff * diff;
        }
        return ret;
    }
    
    template<class T>
    dist_t negative_inner_prod_distance(idx_t a,T& v) const{
        auto pa = get(a);
        dist_t ret = 0;
        for(int i = 0;i < dim;++i){
            ret -= (*(pa + i)) * v[i];
        }
        return ret;
    }
    
    template<class T>
    dist_t negative_cosine_distance(idx_t a,T& v) const{
        auto pa = get(a);
        dist_t ret = 0;
        value_t lena = 0,lenv = 0;
        for(int i = 0;i < dim;++i){
            ret += (*(pa + i)) * v[i];
            lena += (*(pa + i)) * (*(pa + i));
            lenv += v[i] * v[i];
        }
        int sign = ret < 0 ? 1 : -1;
//        return sign * (ret * ret / lena);// / lenv);
        return sign * (ret * ret / lena / lenv);
    }
    
	#ifdef __ENABLE_HASH
	dist_t test_hamming(int a,int b){
		auto pa = get(a),
			 pb = get(b);
		dist_t ret = 0;
		for(int i = 0;i < dim;++i){
			auto diff = (*(pa + i)) ^ (*(pb + i));
			ret += __builtin_popcount(diff);
		}
		return ret;
	}
	template<class T>
    dist_t bit_hamming_distance(idx_t a,T& v) const{
        auto pa = get(a);
        dist_t ret = 0;
        for(int i = 0;i < dim;++i){
			auto diff = (*(pa + i)) ^ (v[i]);
			ret += __builtin_popcount(diff);
        }
        return ret;
    }
	#else
	template<class T>
    dist_t bit_hamming_distance(idx_t a,T& v) const{
		return 0;
    }

	#endif

    template<class T>
    dist_t real_nn(T& v) const{
        dist_t minn = 1e100;
        for(size_t i = 0;i < curr_num;++i){
            auto res = l2_distance(i,v);
            if(res < minn){
                minn = res;
            }
        }
        return minn;
    }
    
    
    std::vector<value_t> organize_point(const std::vector<std::pair<int,value_t>>& v){
        std::vector<value_t> ret(dim,0);
        for(const auto& p : v){
            if(p.first >= dim)
                printf("error %d %d\n",p.first,dim);
            ret[p.first] = p.second;
        }
        return std::move(ret);
    }

    value_t vec_sum2(const std::vector<std::pair<int,value_t>>& v){
        value_t ret = 0;
        for(const auto& p : v){
            if(p.first >= dim)
                printf("error %d %d\n",p.first,dim);
            ret += p.second * p.second;
        }
        return std::move(ret);
    }

	#ifdef __ENABLE_HASH
    void add(idx_t idx, std::vector<bool>& value){
        curr_num = std::max(curr_num,idx);
        auto p = get(idx);
		for(int i = 0;i < value.size();i += sizeof(value_t) * 8){
			value_t tmp = 0;
			for(int j = 0;j < sizeof(value_t) * 8;++j)
				tmp |= value[i + j] ? (1 << j) : 0;
			*(p + i / sizeof(value_t) / 8) = tmp;
		}
    }
	#endif

    void add(idx_t idx, std::vector<std::pair<int,value_t>>& value){
        curr_num = std::max(curr_num,idx);
        auto p = get(idx);
        for(const auto& v : value)
            *(p + v.first) = v.second;
    }

    inline size_t max_vertices(){
        return num;
    }

    inline size_t curr_vertices(){
        return curr_num;
    }

    void print(){
        for(int i = 0;i < num && i < 10;++i)
            printf("%f ",*(data.get() + i));
        printf("\n");
    }

    int get_dim(){
        return dim;
    }

    void dump(std::string file = "bfsg.data"){
        FILE* fp = fopen(file.c_str(),"wb");
        fwrite(data.get(),sizeof(value_t) * num * dim,1,fp);
        fclose(fp);
    }
    
    void load(std::string file = "bfsg.data"){
        curr_num = num;
        FILE* fp = fopen(file.c_str(),"rb");
        auto cnt = fread(data.get(),sizeof(value_t) * num * dim,1,fp);
        fclose(fp);
    }

};
template<>
dist_t Data::l2_distance(idx_t a,idx_t& b) const{
    auto pa = get(a),
         pb = get(b);
    dist_t ret = 0;
    for(int i = 0;i < dim;++i){
        auto diff = *(pa + i) - *(pb + i);
        ret += diff * diff;
    }
    return ret;
}

template<>
dist_t Data::negative_inner_prod_distance(idx_t a,idx_t& b) const{
    auto pa = get(a),
         pb = get(b);
    dist_t ret = 0;
    for(int i = 0;i < dim;++i){
        ret -= (*(pa + i)) * (*(pb + i));
    }
    return ret;
}

template<>
dist_t Data::negative_cosine_distance(idx_t a,idx_t& b) const{
    auto pa = get(a),
         pb = get(b);
    dist_t ret = 0;
    value_t lena = 0,lenv = 0;
    for(int i = 0;i < dim;++i){
        ret += (*(pa + i)) * (*(pb + i));
        lena += (*(pa + i)) * (*(pa + i));
        lenv += (*(pb + i)) * (*(pb + i));
    }
    int sign = ret < 0 ? 1 : -1;
//    return sign * (ret * ret / lena);
    return sign * (ret * ret / lena / lenv);
}

#ifdef __ENABLE_HASH
template<>
dist_t Data::bit_hamming_distance(idx_t a,idx_t& b) const{
    auto pa = get(a),
         pb = get(b);
    dist_t ret = 0;
    for(int i = 0;i < dim;++i){
        auto diff = (*(pa + i)) ^ (*(pb + i));
		ret += __builtin_popcount(diff);
    }
    return ret;
}
#endif

#else
//sparse data
class Data{
public:
    //TODO

};
#endif


