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
typedef bp::voronoi_diagram<double> voronoi_diagram;

extern "C" int get_skeleton(const char *wkt)
{
    std::vector<point> points;
    std::vector<segment> segments;
    polygon p;
    voronoi_diagram vd;

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

    for (auto it = vd.edges().cbegin(); it != vd.edges().cend(); ++it)
    {
        if (it->is_primary() && it->is_finite())
        {
            auto index = it->cell()->source_index();

            double x1, y1, x2, y2;

            x1 = it->vertex0()->x();
            y1 = it->vertex0()->y();
            x2 = it->vertex1()->x();
            y2 = it->vertex1()->y();
            if (x1 <= x2)
            {
                fprintf(stderr, "(%lf %lf) (%lf %lf)\n", x1, y1, x2, y2);
            }
        }
    }

    return 0;
}