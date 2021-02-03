#pragma once

// [begin,end)
template<class T>
__device__
void push_heap(T* begin,T* end){
    T* now = end - 1;
    int parent = (now - begin - 1) / 2;
    while(parent >= 0){
        if(*(begin + parent) < *now){
            auto tmp = *now;
            *now = *(begin + parent);
            *(begin + parent) = tmp;
            now = begin + parent;
    		parent = (parent - 1) / 2;
        }else{
            break;
        }
    }
}
template<class T>
__device__
T pop_heap(T* begin,T* end){
    T ret = *begin;
    *begin = *(end - 1);
    int len = end - begin;
    T* now = begin;
    while(now + 1 < end){
        int left = (now - begin) * 2 + 1;
        int right = (now - begin) * 2 + 2;
        int next = -1;
        if(right < len){
			next = *(begin + left) < *(begin + right) ? right : left;
        } else if(left < len){
			next = left;
		}
        if(next == -1 || !(*now < *(begin + next))){
            break;
        }else{
			T tmp = *now;
			*now = *(begin + next);
			*(begin + next) = tmp;
            now = begin + next;
        }
    }
    return ret;
}
