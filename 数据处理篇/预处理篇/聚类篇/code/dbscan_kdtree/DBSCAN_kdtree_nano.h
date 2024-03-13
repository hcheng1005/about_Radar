#ifndef DBSCAN_KDTREE_NANO_H
#define DBSCAN_KDTREE_NANO_H

#include <pcl/point_types.h>
#include "nanoflann.hpp"
#include <vector>


typedef struct Point_
{
    float x, y, z;  // X, Y, Z position
}Point;


struct adaptor {
    std::vector<Point> m_points;

    explicit adaptor(std::vector<Point>& points)
        : m_points(points)
    {
    }

    [[nodiscard]] inline size_t kdtree_get_point_count() const
    {
        return m_points.size();
    }

    [[nodiscard]] inline float kdtree_get_pt(
        const size_t index, const size_t dim) const
    {
        switch (dim) {
        case 0:
            return m_points[index].x;
        case 1:
            return m_points[index].y;
        default:
            return m_points[index].z;
        }
    }

    template <class BBOX> bool kdtree_get_bbox(BBOX& /*bb*/) const
    {
        return false;
    }
};


class DBSCANNanoCluster
{
public:
    DBSCANNanoCluster(){}
    ~DBSCANNanoCluster(){}

    std::vector<std::vector<unsigned long>> extract(std::vector<Point> &points, float eps, size_t min_pts)
    {
        eps *= eps;
        const auto adapt = adaptor(points);
        using index_t = nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<float, decltype(adapt)>, decltype(adapt),3>;

        index_t indexAdaptor(3, adapt, nanoflann::KDTreeSingleIndexAdaptorParams(10));

        indexAdaptor.buildIndex();

        auto visited = std::vector<bool>(points.size());
        auto clusters = std::vector<std::vector<size_t>>();
        // auto matches = std::vector<std::pair<size_t, float>>();
        // auto sub_matches = std::vector<std::pair<size_t, float>>();

        std::vector<nanoflann::ResultItem<uint32_t, float>> matches;
        std::vector<nanoflann::ResultItem<uint32_t, float>> sub_matches;

        for (size_t i = 0; i < points.size(); i++)
        {
            if (visited[i])
            {
                continue;
            }

            indexAdaptor.radiusSearch(get_query_point(points, i).data(), eps, matches);
            if (matches.size() < static_cast<size_t>(min_pts))
                continue;
            visited[i] = true;

            std::vector<size_t> cluster = {i};

            while (!matches.empty())
            {
                auto nb_idx = matches.back().first;
                matches.pop_back();
                if (visited[nb_idx])
                    continue;
                visited[nb_idx] = true;

                indexAdaptor.radiusSearch(get_query_point(points, nb_idx).data(), eps, sub_matches);

                if (sub_matches.size() >= static_cast<size_t>(min_pts))
                {
                    std::copy(sub_matches.begin(), sub_matches.end(), std::back_inserter(matches));
                }
                cluster.push_back(nb_idx);
            }
            clusters.emplace_back(std::move(cluster));
        }
        return clusters;
    }

    std::array<float, 3> get_query_point(std::vector<Point> &points, size_t index)
    {
        return std::array<float, 3>({(float)points[index].x,
                                     (float)points[index].y, 
                                     (float)points[index].z});
    }

}; // class DBSCANCluster

#endif // DBSCAN_KDTREE_H