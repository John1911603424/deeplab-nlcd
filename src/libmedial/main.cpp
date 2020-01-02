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

#include <vector>

#include <boost/geometry.hpp>
#include <boost/geometry/geometries/polygon.hpp>
#include <boost/geometry/geometries/point_xy.hpp>

#include "medial.h"

namespace bg = boost::geometry;

typedef bg::model::d2::point_xy<int64_t> polygon_integral_point;
typedef bg::model::referring_segment<polygon_integral_point> polygon_segment;
typedef bg::model::polygon<polygon_integral_point> polygon; // XXX

int main()
{
    const char *wkt_polygon = "POLYGON((0 0,0 7,4 2,2 0,0 0))";
    polygon bg_polygon;
    std::vector<int64_t> xy_data;

    int n = -1;
    double *return_data;

    bg::read_wkt(wkt_polygon, bg_polygon);
    bg::for_each_segment(bg_polygon, [&xy_data](polygon_segment s) {
        xy_data.push_back(s.first.x());
        xy_data.push_back(s.first.y());
        xy_data.push_back(s.second.x());
        xy_data.push_back(s.second.y());
    });

    // Compute the internal axis
    n = get_skeleton(xy_data.size(), xy_data.data(), &return_data);

    // Display the internal axis
    for (int i = 0; i < n; i += 4)
    {
        double x1, y1, x2, y2;

        x1 = return_data[i + 0];
        y1 = return_data[i + 1];
        x2 = return_data[i + 2];
        y2 = return_data[i + 3];

        fprintf(stderr, "INTERNAL EDGE: (%lf %lf) (%lf %lf)\n", x1, y1, x2, y2);
    }

    // Cleanup
    free(return_data);
    return_data = nullptr;

    return 0;
}
