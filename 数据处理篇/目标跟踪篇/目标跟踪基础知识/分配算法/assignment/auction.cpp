#include "auction.h"

void auction(int N, double **cost_c, std::vector<int> &assignment)
{
	std::vector<double> C(N * N);

	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N; j++)
		{
			C[i * N + j] = cost_c[i][j]; // rand() % size + 1;
		}
	}

	assignment.resize(N);
	std::vector<double> prices(N, 1);
	double epsilon = 1;
	int iter = 1;

	while (epsilon > 1.0 / N)
	{
		reset(&assignment, INF);
		while (find(assignment.begin(), assignment.end(), INF) != assignment.end())
		{
			iter++;
			auctionRound(&assignment, &prices, &C, epsilon);
		}
		epsilon = epsilon * .05;
	}

	// std::cout << "Num Iterations:\t" << iter << std::endl;
	// std::cout << "Total CPU time:\t" << time << std::endl;
	// std::cout << std::endl << std::endl << "Solution: "  << std::endl;
	// for (int i = 0; i < assignment.size(); i++)
	// {
	// 	std::cout << "Person " << i << " gets object " << assignment[i] << " " << prices[i] << std::endl;
	// }
}

void auctionRound(std::vector<int> *assignment, std::vector<double> *prices, std::vector<double> *C, double epsilon)
{
	int N = prices->size();

	/*
		These are meant to be kept in correspondance such that bidded[i]
		and bids[i] correspond to person i bidding for bidded[i] with bid bids[i]
	*/
	std::vector<int> tmpBidded;
	std::vector<double> tmpBids;
	std::vector<int> unAssig;

	/* Compute the bids of each unassigned individual and store them in temp */
	for (int i = 0; i < assignment->size(); i++)
	{
		if (assignment->at(i) == INF)
		{
			unAssig.push_back(i);

			/*
				Need the best and second best value of each object to this person
				where value is calculated row_{j} - prices{j}
			*/
			double optValForI = -INF;
			double secOptValForI = -INF;
			int optObjForI, secOptObjForI;
			for (int j = 0; j < N; j++)
			{
				double curVal = C->at(j + i * N) - prices->at(j);
				if (curVal > optValForI)
				{
					secOptValForI = optValForI;
					secOptObjForI = optObjForI;
					optValForI = curVal;
					optObjForI = j;
				}
				else if (curVal > secOptValForI)
				{
					secOptValForI = curVal;
					secOptObjForI = j;
				}
			}

			/* Computes the highest reasonable bid for the best object for this person */
			double bidForI = optValForI - secOptValForI + epsilon;

			/* Stores the bidding info for future use */
			tmpBidded.push_back(optObjForI);
			tmpBids.push_back(bidForI);
		}
	}

	/*
		Each object which has received a bid determines the highest bidder and
		updates its price accordingly
	*/
	for (int j = 0; j < N; j++)
	{
		std::vector<int> indices = getIndicesWithVal(&tmpBidded, j);
		if (indices.size() != 0)
		{
			/* Need the highest bid for object j */
			double highestBidForJ = -INF;
			int i_j;
			for (int i = 0; i < indices.size(); i++)
			{
				double curVal = tmpBids.at(indices.at(i));
				if (curVal > highestBidForJ)
				{
					highestBidForJ = curVal;
					i_j = indices.at(i);
				}
			}

			/* Find the other person who has object j and make them unassigned */
			for (int i = 0; i < assignment->size(); i++)
			{
				if (assignment->at(i) == j)
				{
					assignment->at(i) = INF;
					break;
				}
			}

			/* Assign object j to i_j and update the price vector */
			assignment->at(unAssig[i_j]) = j;
			prices->at(j) = prices->at(j) + highestBidForJ;
		}
	}
}

/*<--------------------------------------   Utility Functions   -------------------------------------->*/

std::vector<int> makeRandC(int size)
{
	srand(time(NULL));
	std::vector<int> mat(size * size, 2);
	for (int i = 0; i < size; i++)
	{
		for (int j = 0; j < size; j++)
		{
			mat[i + j * size] = rand() % size + 1;
		}
	}
	return mat;
}

/* Returns a vector of indices from v which have the specified value val */
std::vector<int> getIndicesWithVal(std::vector<int> *v, int val)
{
	std::vector<int> out;
	for (int i = 0; i < v->size(); i++)
	{
		if (v->at(i) == val)
		{
			out.push_back(i);
		}
	}
	return out;
}

void reset(std::vector<int> *v, int val)
{
	for (int i = 0; i < v->size(); i++)
	{
		v->at(i) = val;
	}
}