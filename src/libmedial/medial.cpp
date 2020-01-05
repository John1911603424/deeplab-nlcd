/*
 * The MIT License (MIT)
 * =====================
 *
 * Copyright © 2019-2020 Azavea
 *
 * Permission is hereby granted, free of charge, to any person
 * obtaining a copy of this software and associated documentation
 * files (the “Software”), to deal in the Software without
 * restriction, including without limitation the rights to use,
 * copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following
 * conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
 * OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
 * HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
 * WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 */

#include <cstdio>
#include <cstdint>

#include <set>
#include <vector>

#include <boost/polygon/segment_data.hpp>
#include <boost/polygon/voronoi.hpp>
#include <boost/geometry.hpp>
#include <boost/geometry/geometries/point.hpp>
#include <boost/geometry/geometries/box.hpp>
#include <boost/geometry/index/rtree.hpp>

namespace bp = boost::polygon;
namespace bg = boost::geometry;
namespace bgi = boost::geometry::index;

typedef int64_t integral_coordinate_t;

struct integral_point_t
{
    integral_coordinate_t x;
    integral_coordinate_t y;
} __attribute__((packed));

bool operator<(const integral_point_t &a, const integral_point_t &b)
{
    if (a.x == b.x)
    {
        return (a.y < b.y);
    }
    else
    {
        return (a.x < b.x);
    }
}

struct integral_segment_t
{
    integral_point_t v0;
    integral_point_t v1;
} __attribute__((packed));

bool operator<(const integral_segment_t &a, const integral_segment_t &b)
{
    if (a.v0.x == b.v0.x)
    {
        return (a.v0.y <= b.v0.y);
    }
    else
    {
        return (a.v0.x <= b.v0.x);
    }
}

typedef bp::voronoi_edge<double> voronoi_edge_t;
typedef bp::voronoi_diagram<double> voronoi_diagram_t;

typedef std::pair<const integral_segment_t *, uint32_t> integral_rtree_value_t;
typedef bgi::rtree<integral_rtree_value_t, bgi::linear<16>> integral_rtree_t;

typedef bg::model::point<double, 2, bg::cs::cartesian> real_point_t;
typedef bg::model::segment<real_point_t> real_segment_t;
typedef std::pair<real_segment_t *, uint32_t> real_rtree_value_t;
typedef bgi::rtree<real_rtree_value_t, bgi::linear<16>> real_rtree_t;

// Geometry concepts
namespace boost::geometry::traits
{
template <>
struct tag<integral_point_t>
{
    typedef point_tag type;
};

template <>
struct coordinate_type<integral_point_t>
{
    typedef integral_coordinate_t type;
};

template <>
struct coordinate_system<integral_point_t>
{
    typedef cs::cartesian type;
};

template <>
struct dimension<integral_point_t> : boost::mpl::int_<2>
{
};

template <std::size_t Dimension>
struct access<integral_point_t, Dimension>
{
    static inline integral_coordinate_t get(integral_point_t const &p)
    {
        if (Dimension == 0)
        {
            return p.x;
        }
        else if (Dimension == 1)
        {
            return p.y;
        }
        else
        {
            throw __LINE__;
        }
    }

    static inline void set(integral_point_t &p, integral_coordinate_t c)
    {
        if (Dimension == 0)
        {
            p.x = c;
        }
        else if (Dimension == 1)
        {
            p.y = c;
        }
        else
        {
            throw __LINE__;
        }
    }
};

template <>
struct tag<integral_segment_t>
{
    typedef segment_tag type;
};

template <>
struct point_type<integral_segment_t>
{
    typedef integral_point_t type;
};

template <std::size_t Index, std::size_t Dimension>
struct indexed_access<integral_segment_t, Index, Dimension>
{
    static inline integral_coordinate_t get(integral_segment_t const &s)
    {
        if (Index == 0 && Dimension == 0)
        {
            return s.v0.x;
        }
        else if (Index == 0 && Dimension == 1)
        {
            return s.v0.y;
        }
        else if (Index == 1 && Dimension == 0)
        {
            return s.v1.x;
        }
        else if (Index == 1 && Dimension == 1)
        {
            return s.v1.y;
        }
        else
        {
            throw __LINE__;
        }
    }
};
} // namespace boost::geometry::traits

// Polygon concepts
namespace boost::polygon
{
template <>
struct geometry_concept<integral_point_t>
{
    typedef point_concept type;
};

template <>
struct point_traits<integral_point_t>
{
    typedef int64_t coordinate_type;

    static coordinate_type get(const integral_point_t &p, orientation_2d orient)
    {
        if (orient.to_int() == 0)
        {
            return p.x;
        }
        else
        {
            return p.y;
        }
    }
};

template <>
struct geometry_concept<integral_segment_t>
{
    typedef segment_concept type;
};

template <>
struct segment_traits<integral_segment_t>
{
    typedef point_traits<integral_point_t>::coordinate_type coordinate_type;
    typedef integral_point_t point_type;

    static inline point_type get(const integral_segment_t &s, direction_1d dir)
    {
        if (dir.to_int() == 0)
        {
            return s.v0;
        }
        else
        {
            return s.v1;
        }
    }
};
} // namespace boost::polygon

#define MAGIC_COLOR (33)

extern "C" int get_skeleton(int n, void *segment_data, double **return_data)
{
    auto segments = static_cast<integral_segment_t *>(segment_data);
    std::vector<real_segment_t> axis_vector;
    std::vector<double> sorted_axis_vector;
    voronoi_diagram_t vd;
    integral_rtree_t input_segment_rtree;
    real_rtree_t voronoi_segment_rtree;

    // construct voronoi diagram
    bp::construct_voronoi(segments, segments + (n >> 2), &vd);

    // construct rtree over input segments
    for (int i = 0; i < (n >> 2); ++i)
    {
        auto segment = static_cast<const integral_segment_t *>(segment_data) + i;
        input_segment_rtree.insert(std::make_pair(segment, i));
    }

    // iterate through the voronoi vertices looking for boundary vertices
    for (auto vit = vd.vertices().cbegin(); vit != vd.vertices().cend(); ++vit)
    {
        auto starting_edge = vit->incident_edge()->rot_next();
        for (; starting_edge->cell() != vit->incident_edge()->cell(); starting_edge = starting_edge->rot_next())
        {
            if (starting_edge->is_primary())
            {
                break;
            }
        }
        if (!starting_edge->is_primary())
        {
            break;
        }
        auto starting_source_segment = segments[starting_edge->cell()->source_index()];
        auto shared_endpoints = std::set<integral_point_t>();

        // Initialization
        shared_endpoints.insert(static_cast<integral_point_t>(starting_source_segment.v0));
        shared_endpoints.insert(static_cast<integral_point_t>(starting_source_segment.v1));

        for (auto edge = starting_edge->rot_next(); edge->cell() != starting_edge->cell(); edge = edge->rot_next())
        {
            if (!edge->is_primary())
            {
                continue;
            }
            auto associated_cell = edge->cell();
            auto source_segment = segments[associated_cell->source_index()];
            auto old_shared_endpoints = shared_endpoints;

            // Remove non-shared endpoints
            shared_endpoints.clear();
            if (old_shared_endpoints.count(source_segment.v0) > 0)
            {
                shared_endpoints.insert(source_segment.v0);
            }
            if (old_shared_endpoints.count(source_segment.v1) > 0)
            {
                shared_endpoints.insert(source_segment.v1);
            }

            // If the number of shared endpoints has dropped to zero, leave
            if (shared_endpoints.size() == 0)
            {
                break;
            }
        }

        // If there is a shared endpoint between the source segments
        // of all of the cells that meet this voronoi vertex, then it
        // is a boundary vertex (see
        // http://boost.2283326.n4.nabble.com/voronoi-medial-axis-tp4651161p4651225.html )
        if (shared_endpoints.size() > 0)
        {
            vit->color(MAGIC_COLOR);
        }
    }

    // Loop over voronoi edges, recording those that do not meet a boundary vertex
    for (auto eit = vd.edges().cbegin(); eit != vd.edges().cend(); ++eit)
    {
        if (eit->is_primary() && eit->is_finite() && eit->vertex0()->color() != MAGIC_COLOR && eit->vertex1()->color() != MAGIC_COLOR)
        {
            auto index = eit->cell()->source_index();

            double x1, y1, x2, y2;

            x1 = eit->vertex0()->x();
            y1 = eit->vertex0()->y();
            x2 = eit->vertex1()->x();
            y2 = eit->vertex1()->y();
            if (x1 <= x2)
            {
                axis_vector.emplace_back(real_point_t(x1, y1), real_point_t(x2, y2));
                voronoi_segment_rtree.insert(std::make_pair(&(axis_vector.back()), axis_vector.size() - 1));
            }
        }
    }

    // Get segments in "sorted" order
    real_point_t &current_point = axis_vector.front().first;
    while (voronoi_segment_rtree.size() > 0)
    {
        auto results = std::vector<real_rtree_value_t>();
        voronoi_segment_rtree.query(bgi::nearest(current_point, 1), std::back_inserter(results));
        auto result = results.front();
        const auto & current_segment = axis_vector[result.second];

        if (bg::distance(current_point, current_segment.first) < bg::distance(current_point, current_segment.second))
        {
            // Insert segment in original orientation
            sorted_axis_vector.push_back(current_segment.first.get<0>());
            sorted_axis_vector.push_back(current_segment.first.get<1>());
            sorted_axis_vector.push_back(current_segment.second.get<0>());
            sorted_axis_vector.push_back(current_segment.second.get<1>());
        }
        else
        {
            // Insert flipped segment
            sorted_axis_vector.push_back(current_segment.second.get<0>());
            sorted_axis_vector.push_back(current_segment.second.get<1>());
            sorted_axis_vector.push_back(current_segment.first.get<0>());
            sorted_axis_vector.push_back(current_segment.first.get<1>());
        }

        current_point = current_segment.second;
        voronoi_segment_rtree.remove(result);
    }
    axis_vector.clear();

    /*{
        auto results0 = std::vector<integral_rtree_value_t>();
        auto v0 = integral_point_t{.x = static_cast<integral_coordinate_t>(x1), .y = static_cast<integral_coordinate_t>(y1)};
        input_segment_rtree.query(bgi::nearest(v0, 1), std::back_inserter(results0)); // This is okay due to (pre-)scaling
        for (auto result : results0)
        {
            fprintf(stderr, "%d: (%ld %ld) (%ld %ld) %lf\n", result.second, result.first->v0.x, result.first->v0.y, result.first->v1.x, result.first->v1.y, bg::distance(v0, result.first));
        }

        auto results1 = std::vector<integral_rtree_value_t>();
        auto v1 = integral_point_t{.x = static_cast<integral_coordinate_t>(x2), .y = static_cast<integral_coordinate_t>(y2)};
        input_segment_rtree.query(bgi::nearest(v1, 1), std::back_inserter(results1)); // okay due to scaling
        for (auto result : results1)
        {
            fprintf(stderr, "%d: (%ld %ld) (%ld %ld) %lf\n", result.second, result.first->v0.x, result.first->v0.y, result.first->v1.x, result.first->v1.y, bg::distance(v1, result.first));
        }
                axis_vector.push_back(x1);
                axis_vector.push_back(y1);
                axis_vector.push_back(x2);
                axis_vector.push_back(y2);
        fprintf(stderr, "\n");
    }*/

    // Copy results back
    auto m = sorted_axis_vector.size();
    auto l = m * sizeof(double);
    *return_data = static_cast<double *>(malloc(l));
    memcpy(*return_data, sorted_axis_vector.data(), l);

    return sorted_axis_vector.size();
}
