/*
 * The MIT License (MIT)
 * =====================
 *
 * Copyright © 2019 Azavea
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

#include <boost/geometry.hpp>
#include <boost/geometry/geometries/polygon.hpp>
#include <boost/geometry/geometries/point_xy.hpp>
#include <boost/polygon/segment_data.hpp>
#include <boost/polygon/voronoi.hpp>

namespace bp = boost::polygon;
namespace bg = boost::geometry;

typedef bg::model::d2::point_xy<int64_t> polygon_integral_point;
typedef boost::geometry::model::referring_segment<polygon_integral_point> polygon_segment;
typedef bg::model::polygon<polygon_integral_point> polygon;

typedef bp::point_data<int64_t> point;
typedef bp::segment_data<int64_t> segment;
typedef bp::voronoi_edge<double> voronoi_edge;
typedef bp::voronoi_diagram<double> voronoi_diagram_t;

#define MAGIC_COLOR (33)

extern "C" int get_skeleton(const char *wkt)
{
    std::vector<point> points;
    std::vector<segment> segments;
    polygon p;
    voronoi_diagram_t vd;

    // construct polygon
    bg::read_wkt(wkt, p);

    // construct voronoi diagram
    bg::for_each_segment(p, [&segments](polygon_segment s1) {
        point low(s1.first.x(), s1.first.y());
        point hi(s1.second.x(), s1.second.y());
        segment s2(low, hi);
        segments.push_back(s2);
    });
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
        fprintf(stderr, "VERTEX %lf %lf\n", vit->x(), vit->y());
        if (!starting_edge->is_primary())
        {
            fprintf(stderr, "\tNO PRIMARY EDGES\n");
            break;
        }
        auto starting_source_segment = segments[starting_edge->cell()->source_index()];
        auto shared_endpoints = std::set<decltype(starting_source_segment.low())>();

        fprintf(stderr, "*\t%s EDGE (%lf %lf) (%lf %lf)\n", starting_edge->is_primary() ? "PRIMARY" : "SECONDARY", starting_edge->vertex0()->x(), starting_edge->vertex0()->y(), starting_edge->vertex1()->x(), starting_edge->vertex1()->y());
        fprintf(stderr, "*\t\tSEGMENT (%ld %ld) (%ld %ld)\n", starting_source_segment.low().x(), starting_source_segment.low().y(), starting_source_segment.high().x(), starting_source_segment.high().y());

        // Sanity check
        /*if (!starting_edge->cell()->contains_segment())
        {
            continue;
        }*/

        // Initialization
        shared_endpoints.insert(starting_source_segment.low());
        shared_endpoints.insert(starting_source_segment.high());

        for (auto edge = starting_edge->rot_next(); edge->cell() != starting_edge->cell(); edge = edge->rot_next())
        {
            if (edge->is_finite())
            {
                fprintf(stderr, "\t%s EDGE (%lf %lf) (%lf %lf)\n", edge->is_primary() ? "PRIMARY" : "SECONDARY", edge->vertex0()->x(), edge->vertex0()->y(), edge->vertex1()->x(), edge->vertex1()->y());
            }
            else
            {
                fprintf(stderr, "\t%s EDGE (%lf %lf) (... ...)\n", edge->is_primary() ? "PRIMARY" : "SECONDARY", edge->vertex0()->x(), edge->vertex0()->y());
            }
            if (!edge->is_primary())
            {
                fprintf(stderr, "\t\tSKIPPING SECONDARY EDGE\n");
                continue;
            }
            auto associated_cell = edge->cell();
            auto source_segment = segments[associated_cell->source_index()];
            auto old_shared_endpoints = shared_endpoints;

            fprintf(stderr, "\t\tSEGMENT (%ld %ld) (%ld %ld)\n", source_segment.low().x(), source_segment.low().y(), source_segment.high().x(), source_segment.high().y());
            // Sanity check
            /*if (!associated_cell->contains_segment())
            {
                shared_endpoints.clear();
                break;
            }*/

            // Remove non-shared endpoints
            shared_endpoints.clear();
            if (old_shared_endpoints.count(source_segment.low()) > 0)
            {
                fprintf(stderr, "\t\t\tKEEPING FIRST SEGMENT ENDPOINT\n");
                shared_endpoints.insert(source_segment.low());
            }
            if (old_shared_endpoints.count(source_segment.high()) > 0)
            {
                fprintf(stderr, "\t\t\tKEEPING SECOND SEGMENT ENDPOINT\n");
                shared_endpoints.insert(source_segment.high());
            }

            // If the number of shared endpoints has dropped to zero, leave
            /*if (shared_endpoints.size() == 0)
            {
                break;
            }*/
        }

        // If there is a shared endpoint between the source segments
        // of all of the cells that meet this voronoi vertex, then it
        // is a boundary vertex (see
        // http://boost.2283326.n4.nabble.com/voronoi-medial-axis-tp4651161p4651225.html )
        if (shared_endpoints.size() > 0)
        {
            vit->color(MAGIC_COLOR);
            fprintf(stderr, "\tBOUNDARY VERTEX: (%lf %lf)\n", vit->x(), vit->y());
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
                fprintf(stderr, "INTERNAL EDGE: (%lf %lf) (%lf %lf) %ld %ld\n", x1, y1, x2, y2, eit->vertex0()->color(), eit->vertex1()->color());
            }
        }
    }

    return 0;
}
