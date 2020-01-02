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

#include <vector>
#include <set>

#include <boost/polygon/segment_data.hpp>
#include <boost/polygon/voronoi.hpp>

namespace bp = boost::polygon;

typedef bp::point_data<int64_t> point;
typedef bp::segment_data<int64_t> segment;
typedef bp::voronoi_edge<double> voronoi_edge;
typedef bp::voronoi_diagram<double> voronoi_diagram_t;

#define MAGIC_COLOR (33)

extern "C" int get_skeleton(int n, int64_t *data, double **return_data)
{
    std::vector<point> points;
    std::vector<segment> segments;
    std::vector<double> axis_vector;
    voronoi_diagram_t vd;

    // construct voronoi diagram
    for (int i = 0; i < n; i += 4)
    {
        segments.emplace_back(point(data[i + 0], data[i + 1]), point(data[i + 2], data[i + 3]));
    }
    bp::construct_voronoi(points.cbegin(), points.cend(), segments.cbegin(), segments.cend(), &vd);

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
#if defined(DEBUG)
        fprintf(stderr, "VERTEX %lf %lf\n", vit->x(), vit->y());
#endif
        if (!starting_edge->is_primary())
        {
#if defined(DEBUG)
            fprintf(stderr, "\tNO PRIMARY EDGES\n");
#endif
            break;
        }
        auto starting_source_segment = segments[starting_edge->cell()->source_index()];
        auto shared_endpoints = std::set<decltype(starting_source_segment.low())>();

#if defined(DEBUG)
        fprintf(stderr, "*\t%s EDGE (%lf %lf) (%lf %lf)\n", starting_edge->is_primary() ? "PRIMARY" : "SECONDARY", starting_edge->vertex0()->x(), starting_edge->vertex0()->y(), starting_edge->vertex1()->x(), starting_edge->vertex1()->y());
        fprintf(stderr, "*\t\tSEGMENT (%ld %ld) (%ld %ld)\n", starting_source_segment.low().x(), starting_source_segment.low().y(), starting_source_segment.high().x(), starting_source_segment.high().y());
#endif

        // Initialization
        shared_endpoints.insert(starting_source_segment.low());
        shared_endpoints.insert(starting_source_segment.high());

        for (auto edge = starting_edge->rot_next(); edge->cell() != starting_edge->cell(); edge = edge->rot_next())
        {
#if defined(DEBUG)
            if (edge->is_finite())
            {
                fprintf(stderr, "\t%s EDGE (%lf %lf) (%lf %lf)\n", edge->is_primary() ? "PRIMARY" : "SECONDARY", edge->vertex0()->x(), edge->vertex0()->y(), edge->vertex1()->x(), edge->vertex1()->y());
            }
            else
            {
                fprintf(stderr, "\t%s EDGE (%lf %lf) (... ...)\n", edge->is_primary() ? "PRIMARY" : "SECONDARY", edge->vertex0()->x(), edge->vertex0()->y());
            }
#endif
            if (!edge->is_primary())
            {
#if defined(DEBUG)
                fprintf(stderr, "\t\tSKIPPING SECONDARY EDGE\n");
#endif
                continue;
            }
            auto associated_cell = edge->cell();
            auto source_segment = segments[associated_cell->source_index()];
            auto old_shared_endpoints = shared_endpoints;

#if defined(DEBUG)
            fprintf(stderr, "\t\tSEGMENT (%ld %ld) (%ld %ld)\n", source_segment.low().x(), source_segment.low().y(), source_segment.high().x(), source_segment.high().y());
#endif

            // Remove non-shared endpoints
            shared_endpoints.clear();
            if (old_shared_endpoints.count(source_segment.low()) > 0)
            {
#if defined(DEBUG)
                fprintf(stderr, "\t\t\tKEEPING FIRST SEGMENT ENDPOINT\n");
#endif
                shared_endpoints.insert(source_segment.low());
            }
            if (old_shared_endpoints.count(source_segment.high()) > 0)
            {
#if defined(DEBUG)
                fprintf(stderr, "\t\t\tKEEPING SECOND SEGMENT ENDPOINT\n");
#endif
                shared_endpoints.insert(source_segment.high());
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
#if defined(DEBUG)
            fprintf(stderr, "\tBOUNDARY VERTEX: (%lf %lf)\n", vit->x(), vit->y());
#endif
        }
    }

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
#if defined(DEBUG)
                fprintf(stderr, "INTERNAL EDGE: (%lf %lf) (%lf %lf) %ld %ld\n", x1, y1, x2, y2, eit->vertex0()->color(), eit->vertex1()->color());
#endif
                axis_vector.push_back(x1);
                axis_vector.push_back(y1);
                axis_vector.push_back(x2);
                axis_vector.push_back(y2);
            }
        }
    }

    // Copy results back
    *return_data = static_cast<double *>(malloc(axis_vector.size() * sizeof(double)));
    memcpy(*return_data, axis_vector.data(), axis_vector.size() * sizeof(double));

    return axis_vector.size();
}
