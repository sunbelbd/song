#pragma once

namespace smmh2{

template<class T>
__device__
void inline swap(T& a, T& b){
    T c(a); a=b; b=c;
}

template<class T>
__device__
int adjust_sibling(T* smmh,int Y,int max_size){
	int s;
	if(Y & 1){ // Y is left child)
		s = Y + 1;
		if(s >= max_size)
			return Y;
		if(smmh[Y] > smmh[s]){
			swap(smmh[Y],smmh[s]);
			return s;
		}
	}else{ // Y is right child
		s = Y - 1;
		if(smmh[Y] < smmh[s]){
			swap(smmh[Y],smmh[s]);
			return s;
		}
	}
	return Y;
}

__device__
int inline parent(int x){
	return ((x - 1) >> 1);
}

__device__
int inline grandparent(int x){
	return ((x - 3) >> 2);
}

__device__
int inline leftchild(int x){
	return ((x << 1) + 1);
}

__device__
int inline rightchild(int x){
	return ((x << 1) + 2);
}

__device__
bool inline is_leaf(unsigned int x,int max_size){
	return leftchild(x) >= max_size;
}

template<class T>
__device__
int adjust_grandparent(T* smmh,int Y,int max_size){
	if(Y <= 2)
		return Y;
	int G = grandparent(Y);
	int GL = leftchild(G),GR = rightchild(G);
	if(smmh[GL] > smmh[Y]){
		swap(smmh[GL],smmh[Y]);
		return GL;
	}else if(smmh[GR] < smmh[Y]){
		swap(smmh[GR],smmh[Y]);
		return GR;
	}
	return Y;
}

template<class T>
__device__
void insert(T* smmh,int& max_size,T& entry){
	int Y = max_size;
	smmh[max_size++] = entry;
	//printf("T %d %f\n",(int)entry.second,entry.first);
	while(1){
		Y = adjust_sibling(smmh,Y,max_size);
		int X = adjust_grandparent(smmh,Y,max_size);
		if(X == Y)
			break;
		Y = X;
	}
}

template<class T>
__device__
int adjust_grandchild(T* smmh,int Y,int max_size){
	if(Y & 1){
		if(is_leaf(Y,max_size))
			return Y;
		int CL = leftchild(Y),CR = leftchild(Y + 1);
		int C = CL;
		if(CR < max_size && smmh[CR] < smmh[CL])
			C = CR;
		if(smmh[C] < smmh[Y]){
			swap(smmh[C], smmh[Y]);
			return C;
		}
	}else{ // Y is a rightchild
		int CL = rightchild(Y - 1),CR = rightchild(Y);
		if(CL >= max_size)
			return Y;
		int C = CL;
		if(CR < max_size && smmh[CR] > smmh[CL])
			C = CR;
		if(smmh[C] > smmh[Y]){
			swap(smmh[C], smmh[Y]);
			return C;
		}
	}
	return Y;
}

template<class T>
__device__
void deletion(T* smmh,int idx,int& max_size){
	smmh[idx] = smmh[--max_size];
	int Y = idx;
	while(1){
		Y = adjust_sibling(smmh,Y,max_size);
		int X = adjust_grandchild(smmh,Y,max_size);
		if(X == Y)
			break;
		Y = X;
	}
}

template<class T>
__device__
T pop_min(T* smmh,int& max_size){
	T ret = smmh[1];
	deletion(smmh,1,max_size);
	return ret;
}

// NOTICE: must ensure max_size > 2
template<class T>
__device__
T pop_max(T* smmh,int& max_size){
	T ret = smmh[2];
	deletion(smmh,2,max_size);
	return ret;
}

template<class T>
__device__
void pretty_print(T* smmh,int& max_size){
	int border = 2;
	for(int i = 0;i < max_size;++i){
		printf("%d\t",smmh[i]);
		if(i + 2 == border){
			printf("\n");
			border <<= 1;
		}
	}
	printf("\n");
}

};
