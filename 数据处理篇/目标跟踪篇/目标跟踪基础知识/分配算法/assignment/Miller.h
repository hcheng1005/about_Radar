/*
 * MurtyMiller.h
 *
 *  Created on: 15.08.2013
 *      Author: fb
 *
 * Murty's algorithm implementation according to
 * Miller's pseudo-code formulation in "Optimizing Murty's ranked assignment method"
 *
 * Miller, M.L.; Stone, H.S.; Cox, Ingemar J., "Optimizing Murty's ranked assignment method,"
 * Aerospace and Electronic Systems, IEEE Transactions on , vol.33, no.3, pp.851,862, July 1997
 * doi: 10.1109/7.599256
 */

#ifndef MILLER_H_
#define MILLER_H_

#include "AuctionAlgorithm.h"
#include <queue>

#include "Lapjv.h"

template <typename Scalar = double>
class MurtyMiller
{
public:
    typedef Eigen::Matrix<Scalar, -1, -1> WeightMatrix;
    typedef Eigen::Matrix<size_t, -1, -1> AssignmentMatrix;
    typedef typename Auction<Scalar>::Edge Edge;
    typedef typename Auction<Scalar>::Edges Edges;

    /**
     * a partition represents an assignment matrix (i.e. edges)
     * with it's weight matrix
     * see Murty's algorithm for details
     */
    class Partition
    {
    public:
        Partition() : value(0)
        {
            w = WeightMatrix::Zero(w.rows(), w.cols());
        }

        Partition(const Edges &edges, const WeightMatrix &w, const Scalar v) : edges(edges), w(w), value(v)
        {
        }

        Edges edges;
        WeightMatrix w;
        Scalar value;
    };

    struct ComparePartition : std::binary_function<Partition, Partition, bool>
    {
        bool operator()(const Partition &lhs, const Partition &rhs) const
        {
            return (lhs.value < rhs.value);
        }
    };

    /**
     * list of partitions
     */
    typedef typename std::vector<Partition> Partitions;

    /**
     * sum up values of edges, i.e. objective function value
     * @param edges
     * @return
     */
    static Scalar objectiveFunctionValue(const Edges &edges)
    {
        Scalar v = 0;
        for (const auto &e : edges)
            v += e.v;

        return v;
    }

    static typename std::vector<Edges> getMBestAssignments(const WeightMatrix &w, const size_t mBest = 5)
    {
#define LAPJV_ALG
        clock_t start1 = clock();
        const size_t rows = w.rows(), cols = w.cols();

        double **cost_ptr; // used by lapjv

        cost_ptr = new double *[sizeof(double *) * rows];
        for (int i = 0; i < rows; i++)
        {
            cost_ptr[i] = new double[sizeof(double) * rows];
        }

        double totalTime = 0;

        assert(rows != 0 && cols != 0 && cols >= rows);

        int x_c[10];
        int y_c[10];

        typename std::vector<Edges> resultingEdges;

        // special case if rows = cols = 1
        if (cols == 1 && rows == 1)
        {
            if (w(0, 0) == 0)
                return resultingEdges;

            Edges edges;
            edges.emplace_back(Edge(0, 0, w(0, 0)));
            resultingEdges.emplace_back(edges);
            return resultingEdges;
        }

        size_t kBest = 0;

        const size_t maxComb = (rows > cols) ? rows : cols;
        // if rows! < mBest ...
        switch (maxComb)
        {
        case 1:
            kBest = 1;
            break;
        case 2:
            kBest = 2;
            break;
        case 3:
            kBest = 6;
            break;
        case 4:
            kBest = 24;
            break;
        default:
            kBest = mBest;
            break;
        }
        if (mBest < kBest)
            kBest = mBest;

        std::cout << "kBest = " << kBest << std::endl;
#ifndef LAPJV_ALG
        Edges edges = Auction<Scalar>::solve(w); // make initial (best) assignment
#else
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < rows; j++)
            {
                // NOTE: different alg with different cost_mat array
                cost_ptr[i][j] = w(i, j) * -1.0;
            }
        }
        int ret = lapjv_internal(rows, cost_ptr, &x_c[0], &y_c[0]);
        Edges edges;
        if (!ret)
        {
            for (int i = 0; i < rows; i++)
            {
                Edge e;
                e.x = i;
                e.y = x_c[i];
                e.v = w(i, x_c[i]);
                edges.push_back(e);
            }
        }
        else
        {
            std::cout << "Error" << std::endl;
        }
#endif

        // sort edges by row
        std::sort(edges.begin(), edges.end(), [](const Edge &e1, const Edge &e2)
                  { return e1.x < e2.x; });

        // initial partition, i.e. best solution
        Partition init(edges, w, objectiveFunctionValue(edges));

        typedef std::priority_queue<Partition, std::vector<Partition>, ComparePartition> PartitionsPriorityQueue;

        // create answer-list with initial partition
        PartitionsPriorityQueue priorityQueue, answerList;
        priorityQueue.push(init);

        // assume values between 0 and 1 !
        const Scalar lockingValue = 0.;

        while (!priorityQueue.empty() && answerList.size() < kBest)
        {
            clock_t start_time = clock();
            // take first element from queue
            Partition currentPartition = priorityQueue.top();
            priorityQueue.pop();

            answerList.push(currentPartition);

            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < rows; j++)
                {
                    // NOTE: different alg with different cost_mat array
                    cost_ptr[i][j] = currentPartition.w(i, j) * -1.0;
                }
            }

            // for all triplets in this solution
            for (size_t e = 0; e < currentPartition.edges.size(); ++e)
            {
                auto &triplet = currentPartition.edges[e];

                WeightMatrix P_ = currentPartition.w; // P' = P

                // exclude edge by setting weight in matrix to lockingValue -> NOT (x, y)
                P_(triplet.x, triplet.y) = lockingValue;

                cost_ptr[triplet.x][triplet.y] = lockingValue;

                // determine solution for changed matrix and create partition
#ifndef LAPJV_ALG
                clock_t start_time = clock();
                Edges S_ = Auction<Scalar>::solve(P_);
                clock_t end_time = clock();
#else

                // USING LAPJV ALG
                int ret = lapjv_internal(rows, cost_ptr, &x_c[0], &y_c[0]);
                Edges S_;
                if (!ret)
                {
                    for (int i = 0; i < rows; i++)
                    {
                        Edge e;
                        e.x = i;
                        e.y = x_c[i];
                        e.v = P_(i, x_c[i]);
                        S_.push_back(e);
                    }
                }
                else
                {
                    std::cout << "Error" << std::endl;
                }
#endif

#ifdef __ASSOCIATON_FINDER_DEBUG
                for (const auto &t : currentPartition.edges)
                {
                    if (t.x == triplet.x && t.y == triplet.y)
                        std::cout << "NOT ";
                    std::cout << "(" << t.x << ", " << t.y << ") ";
                }
                std::cout << "sum = " << objectiveFunctionValue(S_) << std::endl;
#endif

                if (S_.size() == P_.rows()) // solution found? (rows >= cols!)
                {
                    // sort edges by row
                    std::sort(S_.begin(), S_.end(), [](const Edge &e1, const Edge &e2)
                              { return e1.x < e2.x; });

                    priorityQueue.emplace(Partition(S_, P_, objectiveFunctionValue(S_)));
                }
                // remove all vertices that include row and column of current node
                // i.e. force using this edge
                for (size_t r = 0; r < rows; ++r)
                {
#ifndef LAPJV_ALG
                    currentPartition.w(r, triplet.y) = lockingValue;
#else
                    cost_ptr[r][triplet.y] = lockingValue;
#endif
                }

                for (size_t c = 0; c < cols; ++c)
                {
#ifndef LAPJV_ALG
                    currentPartition.w(triplet.x, c) = lockingValue;
#else
                    cost_ptr[triplet.x][c] = lockingValue;
#endif
                }

                // set edge back to original value
                currentPartition.w(triplet.x, triplet.y) = triplet.v = w(triplet.x, triplet.y);
                cost_ptr[triplet.x][triplet.y] = triplet.v = w(triplet.x, triplet.y) * -1.0;
            }

            clock_t end_time = clock();
            double time = (double)(end_time - start_time) / CLOCKS_PER_SEC * 1000.0;

            totalTime += time;
        }
        // create return list
        while (!answerList.empty())
        {
            resultingEdges.emplace_back(answerList.top().edges);
            answerList.pop();
        }

        std::cout << "Total CPU time:\t" << totalTime << std::endl;
        clock_t end1 = clock();
        double time2 = (double)(end1 - start1) / CLOCKS_PER_SEC * 1000.0;
        std::cout << "Total CPU time:\t" << time2 << std::endl;
        return resultingEdges;
    }
};

#endif
