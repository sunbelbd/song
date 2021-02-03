#pragma once

struct VanillaList{
	const static int capacity = 1000;
    idx_t data[capacity];
    int len = 0;

	__device__
    void add(idx_t x){
		if(len == capacity)
			return;
		data[len++] = x;
    }

	__device__
    bool test(idx_t x){
        for(int i = 0;i < len;++i){
			if(x == data[i])
				return true;
		}
        return false;
    }
    
};
