
#ifndef AUCTION_H
#define AUCTION_H

#include <iostream>
#include <vector>
#include <limits>
#include <stdlib.h>
#include <math.h>
#include <algorithm>

#define INF std::numeric_limits<int>::max()
#define VERBOSE false

/* Pre-declare functions to allow arbitrary call ordering  */
void auction(int N, double **cost, std::vector<int>& assignment);
void auctionRound(std::vector<int>* assignment, std::vector<double>* prices, std::vector<double>* C, double epsilon);
std::vector<int> getIndicesWithVal(std::vector<int>* v, int val);
void reset(std::vector<int>* v, int val);



#endif