#pragma once

template<typename T,const int max_size,const int EMPTY = -1,const int DELETION = -2>
struct FixHash{
    T data[max_size];
    //const static int num_hash = 7;
	
	const uint64_t random_number[8 * 2] = {
0x4bcb391f924ed183ULL,0xa0ab69ccd854fc0aULL,0x91086b9cecf5e3b7ULL,
0xc68e01641bead407ULL,0x3a7b976128a30449ULL,0x6d122efabfc4d99fULL,
0xe6700ef8715030e2ULL,0x80dd0c3bffcfb45bULL,0xe80f45af6e4ce166ULL,
0x6cf43e5aeb53c362ULL,0x31a27265a93c4f40ULL,0x743de943cecde0a4ULL,
0x5ed25dba0288592dULL,0xa69eb51a362c37bcULL,0x9a558fed9d4824f0ULL,0xf75678c2fdbdd68bULL//,0x34423f0963258c85ULL,0x3532778d6726905cULL,0x6fef7cbe609500f9ULL,0xb4419d54de48422ULL,0xda2157c5b12f41b6ULL,0xb315fbc927cae57eULL,0x4a6a38aaa5dcc71cULL,0x86b8c876df8a93f1ULL,0x20ee1d11467a102aULL,0x181399179bae820dULL,0x754794ac0581f2deULL,0xbb7dd7b268a1b05fULL,0x51f3f6b9061423e7ULL,0x2bc1feada8d098c0ULL,0x9629581689d33379ULL,0xa7db527f1e730387ULL,0x5d84ff10cd4d94d6ULL,0x86bc263fccb53eb7ULL,0xca1c3c264474cf4ULL,0x67eea94e006ddd46ULL,0x71d965ad9969018aULL,0xaf497940b2a58b9dULL,0x666c1a4a0bfb7d2eULL,0x13e52fdfab38213cULL,0x5aecd595110f8dfcULL,0xce3bb15c0334a4a8ULL,0xbdd3dbe329975051ULL,0xbb905e5237d4d0caULL,0xb07a1f2382567678ULL,0xc532f79af3352014ULL,0x6b7e603d5948f57bULL,0xc4c91c988f2a874fULL,0xed8c88a357a7e631ULL,0x83e7044453e44307ULL,0x58d175e98509c816ULL,0x5e0b9a22c7cb3beULL,0x2b391d3377c181eaULL,0x41e2b6d7fd610dd8ULL,0x15545fc7f219b48eULL,0x63baf917fa36f69eULL,0xa091555b086fc61eULL,0xda72de0a0625ef02ULL,0x70a6739cae181b68ULL,0x3a306eeb92f0dc4bULL,0xaab82d42e889cf80ULL,0x7fd20e629628bfacULL,0x22c09f4593f19b27ULL,0x74e124cbfe6a12f8ULL
        };
    
	__device__
	FixHash(){
        for(int i = 0;i < max_size;++i)
            data[i] = EMPTY;
    }

	__device__
    short hash(int h,idx_t x){
        return (x ^ (x >> 16) * random_number[h << 1] ^ random_number[(h << 1) + 1]) % max_size;
        //return (x ^ (x >> 32) * random_number[h << 1] ^ random_number[(h << 1) + 1]) & ((size64 << 6) - 1);
    }

	__device__
    void add(T x){
		auto code = hash(0,x);
		while(data[code] != EMPTY
#ifdef __ENABLE_VISITED_DEL
			 && data[code] != DELETION
#endif
									  )
			code = (code + 1) % max_size;
		data[code] = x;
    }
	
	__device__
    bool test(T x){
		auto code = hash(0,x);
		while(data[code] != EMPTY){
			if(data[code] == x)
				return true;
			code = (code + 1) % max_size;
		}
		return false;
    }

	__device__
	void del(T x){
		auto code = hash(0,x);
		auto remove_idx = code;
		while(data[remove_idx] != EMPTY){
			if(data[remove_idx] == x)
				break;
			remove_idx = (remove_idx + 1) % max_size;
		}
		if(data[remove_idx] == EMPTY)
			return;

		int next_idx = (remove_idx + 1) % max_size;
		while(data[next_idx] != EMPTY){
			auto new_code = hash(0,data[next_idx]);
// code remove_idx next_idx
// next_idx code remove_idx
// remove_idx next_idx code
			bool cond1 = code <= remove_idx;
			bool cond2 = remove_idx < next_idx;
			if(cond1 && cond2){
				if(new_code <= remove_idx){
					data[remove_idx] = data[next_idx];
					remove_idx = next_idx;
					code = new_code;
				}
			}else if(cond1){
				if(next_idx < new_code && new_code <= remove_idx){
					data[remove_idx] = data[next_idx];
					remove_idx = next_idx;
					code = new_code;
				}
			}else{
				if(next_idx < new_code || new_code <= remove_idx){
					data[remove_idx] = data[next_idx];
					remove_idx = next_idx;
					code = new_code;
				}
			}
			next_idx = (next_idx + 1) % max_size;
		}
		data[remove_idx] = EMPTY;
	}

/*
	__device__
	void del(T x){
		auto code = hash(0,x);
		while(data[code] != EMPTY){
			if(data[code] == x){
				data[code] = DELETION;
				break;
			}
			code = (code + 1) % max_size;
		}
		return;
	}*/
    
};

