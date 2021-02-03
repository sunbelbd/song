#pragma once

#define bucketSize 4
#define bucket_t uint8_t
#define BUCKET_T_MOD 255

template<const int capacity>
class CuckooFilter{
public:
	bucket_t buckets[capacity/bucketSize][bucketSize];
	int count = 0;

	const int maxCuckooCount = capacity / bucketSize;

	__device__
	bucket_t hash2fp(idx_t x){
		return (x % BUCKET_T_MOD) + 1;
	}

	// Lookup returns true if data is in the counter
	__device__
	bool test(idx_t x){
		int i1,i2;
		bucket_t fp;
		getIndicesFingerprint(x,i1,i2,fp);
		return getFingerprintIndex(buckets[i1],fp) > -1 || getFingerprintIndex(buckets[i2],fp) > -1;
	}

	__device__
	int randi(int i1,int i2){
		return ((i1 * (i2 >> 5)) ^ i2 ^ buckets[(i1 + i2) / 2][0]) % 2 == 0 ? i1 : i2;
	}

	// Insert inserts data into the counter and returns true upon success
	__device__
	bool Insert(idx_t x){
		int i1,i2;
		bucket_t fp;
		getIndicesFingerprint(x,i1,i2,fp);
		if (insert(fp, i1) || insert(fp, i2)) {
			return true;
		}
		//return false;
		return reinsert(fp, randi(i1, i2));
	}

	// InsertUnique inserts data into the counter if not exists and returns true upon success
	__device__
	bool add(idx_t x) {
		if(test(x)){
			return false;
		}
		return Insert(x);
	}

	__device__
	bool insert(bucket_t x, int i){
		if(bucket_insert(buckets[i],x)){
			++count;
			return true;
		}
		return false;
	}

	__device__
	bool reinsert(bucket_t x, int i){
		for(int k = 0; k < maxCuckooCount; ++k) {
			int j = hash(x + k * 156722 + 1034311351) % bucketSize;
			idx_t oldfp = x;
			x = buckets[i][j];
			buckets[i][j] = oldfp;

			// look in the alternate location for that random element
			i = getAltIndex(x, i);
			if(insert(x, i)){
				return true;
			}
		}
		return false;
	}

	// Delete data from counter if exists and return if deleted or not
	__device__
	bool del(idx_t x) {
		int i1,i2;
		bucket_t fp;
		getIndicesFingerprint(x,i1,i2,fp);
		return internal_del(fp, i1) || internal_del(fp, i2);
	}

	__device__
	bool internal_del(bucket_t x, int i){
		if(bucket_delete(buckets[i],x)){
			--count;
			return true;
		}
		return false;
	}

	__device__
	bool bucket_insert(bucket_t* bucket,bucket_t x) {
		for(int i = 0;i < bucketSize;++i){
			if(bucket[i] == 0) {
				bucket[i] = x;
				return true;
			}
		}
		return false;
	}

	__device__
	bool bucket_delete(bucket_t* bucket,bucket_t x){
		for(int i = 0;i < bucketSize;++i){
			if(bucket[i] == x) {
				bucket[i] = 0;
				return true;
			}
		}
		return false;
	}

	__device__
	int getFingerprintIndex(bucket_t* bucket,bucket_t x) {
		for(int i = 0;i < bucketSize;++i){
			if(bucket[i] == x) {
				return i;
			}
		}
		return -1;
	}

	__device__
	int getAltIndex(bucket_t fp, int i) {
		uint32_t h = hash(fp);
		return (i ^ h) % (capacity / bucketSize);
	}

	// getIndicesAndFingerprint returns the 2 bucket indices and fingerprint to be used
	__device__
	void getIndicesFingerprint(idx_t x,int &i1,int &i2,bucket_t& fp) {
		uint32_t h = hash(x);
		fp = hash2fp(x);
		i1 = h % (capacity / bucketSize);
		i2 = getAltIndex(fp, i1);
		return;
	}
	
	__device__
    int hash(idx_t x){ 
		x ^= x >> 33;
		x *= 0x2b391d3377c181eaULL;
		x ^= x >> 33;
		x *= 0x41e2b6d7fd610dd8ULL;
		x ^= x >> 33;
		return x;
	}

};

