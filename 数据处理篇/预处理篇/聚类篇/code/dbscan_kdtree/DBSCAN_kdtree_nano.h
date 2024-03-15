#ifndef DBSCAN_KDTREE_NANO_H
#define DBSCAN_KDTREE_NANO_H

#include <pcl/point_types.h>
#include "nanoflann.hpp"
#include <vector>

template <typename T>
struct PointCloud
{
    struct Point
    {
        T x, y, z;
    };

    using coord_t = T;  //!< The type of each coordinate

    std::vector<Point> pts;

    // Must return the number of data points
    inline size_t kdtree_get_point_count() const { return pts.size(); }

    // Returns the dim'th component of the idx'th point in the class:
    // Since this is inlined and the "dim" argument is typically an immediate
    // value, the
    //  "if/else's" are actually solved at compile time.
    inline T kdtree_get_pt(const size_t idx, const size_t dim) const
    {
        if (dim == 0)
            return pts[idx].x;
        else if (dim == 1)
            return pts[idx].y;
        else
            return pts[idx].z;
    }

    // Optional bounding-box computation: return false to default to a standard
    // bbox computation loop.
    //   Return true if the BBOX was already computed by the class and returned
    //   in "bb" so it can be avoided to redo it again. Look at bb.size() to
    //   find out the expected dimensionality (e.g. 2 or 3 for point clouds)
    template <class BBOX>
    bool kdtree_get_bbox(BBOX& /* bb */) const
    {
        return false;
    }
};

template <typename T>
class DBSCANNanoCluster
{
public:
    DBSCANNanoCluster(){}
    ~DBSCANNanoCluster(){}

    std::vector<std::vector<unsigned long>> extract(PointCloud<T> &points, float eps, size_t min_pts)
    {
        eps *= eps;
        using KDtree = nanoflann::KDTreeSingleIndexAdaptor< nanoflann::L2_Simple_Adaptor< double, PointCloud<double> >,
                                                      PointCloud<double>, 3 >;

        KDtree indexAdaptor(3, points, nanoflann::KDTreeSingleIndexAdaptorParams(15));

        indexAdaptor.buildIndex();

        auto visited = std::vector<bool>(points.pts.size());
        auto clusters = std::vector<std::vector<size_t>>();

        std::vector<nanoflann::ResultItem<uint32_t, double>> matches;
        std::vector<nanoflann::ResultItem<uint32_t, double>> sub_matches;

        for (size_t i = 0; i < points.pts.size(); i++)
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

    std::array<double, 3> get_query_point(PointCloud<T>&points, size_t index)
    {
        return std::array<double, 3>({(float)points.pts.at(index).x,
                                     (float)points.pts.at(index).y, 
                                     (float)points.pts.at(index).z});
    }

}; // class DBSCANCluster

#endif // DBSCAN_KDTREE_H