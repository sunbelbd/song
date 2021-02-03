#include<stdio.h>
#include<string.h>
#include<algorithm>
#include<math.h>

double _p_from_kr(double k, double r) {
	return pow(1 - exp(-k / r), k);
}

double _k_from_r(double r) {
	return round(log(2) * r);
}

double _r_from_pk(double p,double  k) {
	return -k / log(1 - exp(log(p) / k));
}

double _r_from_mn(double m,double  n) {
	return m / n;
}

std::pair<double,double> km_from_np(double n,double p) {
	double m = ceil(n * log(p) / log(1 / pow(2, log(2))));

	double r = _r_from_mn(m, n);
	double k = _k_from_r(r);
	p = _p_from_kr(k, r);
	
	return std::make_pair(k,m);
}

std::pair<double,double> kp_from_mn(double m,double n) {
	double r = _r_from_mn(m, n);
	double k = _k_from_r(r);
	double p = _p_from_kr(k, r);
	return std::make_pair(k,p);
}

std::pair<double,double> kn_from_mp(double m,double p) {
	double n = ceil((m * log(pow(1 / 2, log(2))) / log(p)));
	double r = _r_from_mn(m, n);
	double k = _k_from_r(r);
	p = _p_from_kr(k, r);
	return std::make_pair(k,n);
}

double p_from_kmn(double k,double m,double n) {
	double p = _p_from_kr(k, _r_from_mn(m, n));
	return p;
}

double n_from_kmp(double k,double m,double p) {
	double r = _r_from_pk(p, k);
	double n = ceil(m / r);
	return n;
}

double m_from_knp(double k,double n,double p) {
	double r = _r_from_pk(p, k);
	double m = ceil(n * r);
	return m;
}


int main(int argc,char** argv){
	if(argc != 2){
		fprintf(stderr,"Usage: %s <topk>\n",argv[0]);
		return 1;
	}
	double topk = atof(argv[1]);
	double n = topk * 12.897;
	int maxn = 15;
	for(int i = 1;i <= maxn;++i){
		double m = 64 * (1 << i);
		auto tmp = kp_from_mn(m,n);
		double k = tmp.first;
		double p = tmp.second;
		if(p < 0.03){
			printf("bit %d %d hash %d p %f\n",1 << i,i,(int)round(k),p);
			break;
		}
		if(i == maxn)
			printf("bit %d %d hash %d p %f\n",1 << i,i,(int)round(k),p);
	}
	return 0;
}

