/******************************************************************************
 * This is an example comparing the efficiency of various assignment algorithms.
 *****************************************************************************/

#include "../../include/assignment/HungarianAlg.h" //min cost
#include "../../include/assignment/Lapjv.h"        //min cost
#include "../../include/assignment/auction.h"      //max cost
#include "../../include/assignment/hungarian_optimizer.h"

using namespace assign;

// define cost_mat = [trace * dets]
cost_t cost_c[10][10] = {
    1.74816487, 2.38917233, 1.71076069, 3.39056081, 8.97918614, 8.8463572,
    5.72492498, 3.51112043, 2.83719919, 1.47646523, 6.90482939, 6.54823362,
    2.97598861, 8.58927493, 1.27471997, 6.63336473, 8.00192929, 5.53644708,
    8.17686098, 6.53984023, 5.12970743, 5.15610536, 6.76563599, 3.63906049,
    9.5657066, 0.9938076, 4.65050956, 9.40180311, 7.40438347, 2.76392061,
    0.14804986, 2.46669343, 7.33323472, 5.8211227, 7.97660068, 4.25715621,
    8.70762212, 3.84921524, 3.19890027, 2.28689383, 8.34067808, 2.06432393,
    5.28740296, 4.65337427, 7.83300603, 8.53227427, 5.38427513, 1.03191784,
    6.6057736, 7.68792094, 4.62337316, 9.95950717, 7.65598475, 2.33958583,
    4.71393984, 8.73278614, 5.13445941, 8.88417655, 5.28262101, 1.08137045,
    6.5017676, 3.71347059, 8.90070478, 6.89908671, 9.3396071, 9.69360009,
    8.39359751, 9.25831462, 9.28462701, 4.67101498, 0.19922305, 8.61400931,
    4.97661521, 2.94110684, 4.14077323, 4.74816978, 4.42211109, 3.70811997,
    2.46486932, 6.42482562, 7.4039233, 3.37486973, 0.27083053, 0.18565782,
    5.25106232, 2.51429459, 8.12555989, 2.01157174, 9.21221066, 2.54981598,
    7.40352095, 7.36382558, 0.7780371, 1.78572676, 1.72834597, 8.56007773,
    8.72230221, 7.66976083, 7.88648666, 0.24672};

int main(void)
{
    double start_time, end_time;
    int n = 10;
    int x_c[10];
    int y_c[10];

    double **cost_ptr;         // used by lapjv
    double **cost_ptr2;        // used by lapjv
    distMatrix_t distMatrixIn; // used by hungarian

    cost_ptr = new double *[sizeof(double *) * n];
    cost_ptr2 = new double *[sizeof(double *) * n];
    for (int i = 0; i < n; i++)
    {
        cost_ptr[i] = new double[sizeof(double) * n];
        cost_ptr2[i] = new double[sizeof(double) * n];
    }

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            // NOTE: different alg with different cost_mat array
            cost_ptr[i][j] = cost_c[i][j] * -1.0;
            distMatrixIn.push_back(cost_c[j][i] * -1.0);
            cost_ptr2[i][j] = cost_c[i][j];
        }
    }

    // run "lapjv alg"
    start_time = clock();
    int_t ret = lapjv_internal(n, cost_ptr, &x_c[0], &y_c[0]);
    end_time = clock();

    std::cout << "Time-Lapjv: " << (end_time - start_time) / CLOCKS_PER_SEC * 1000.0 << "ms" << std::endl;
    if (ret != 0)
    {
        std::cout << "Calculate Wrong!" << std::endl;
    }

    for (int i = 0; i < n; i++)
    {
        std::cout << x_c[i] << std::endl;
    }

    // run "hungarian/km alg"
    AssignmentProblemSolver *munkres = new AssignmentProblemSolver();
    std::vector<int> assignment;

    start_time = clock();
    munkres->Solve(distMatrixIn, n, n, assignment, munkres->optimal);
    end_time = clock();
    std::cout << "Time-hungarian: " << (end_time - start_time) / CLOCKS_PER_SEC * 1000.0 << "ms" << std::endl;

    // display result
    for (const auto sub_ : assignment)
    {
        std::cout << sub_ << std::endl;
    }

    // run "auction alg"
    std::vector<int> assignment2;
    start_time = clock();
    auction(n, cost_ptr2, assignment2);
    end_time = clock();
    std::cout << "Time-Auction: " << (end_time - start_time) / CLOCKS_PER_SEC * 1000.0 << "ms" << std::endl;
    // display result
    for (const auto sub_ : assignment2)
    {
        std::cout << sub_ << std::endl;
    }

    // run "apollo hungarian Alg"
    HungarianOptimizer<float> *optimizer_ = new HungarianOptimizer<float>();
    optimizer_->costs()->Reserve(100, 100);
    std::vector<std::pair<size_t, size_t>> assignments;

    optimizer_->costs()->Resize(n, n);
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            (*optimizer_->costs())(i, j) = cost_c[i][j] * -1.0;
        }
    }

    // 执行分配
    start_time = clock();
    optimizer_->Minimize(&assignments);
    // optimizer_->PrintMatrix();
    end_time = clock();
    std::cout << "Time-Apollo-Hungarian: " << (end_time - start_time) / CLOCKS_PER_SEC * 1000.0 << "ms" << std::endl;

    // display result
    for (const auto &assignment : assignments)
    {
        std::cout << assignment.first << ", " << assignment.second << std::endl;
    }

    return 1;
}