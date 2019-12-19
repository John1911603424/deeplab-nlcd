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

#include <boost/polygon/segment_data.hpp>
#include <boost/polygon/voronoi.hpp>

namespace bp = boost::polygon;

typedef bp::point_data<int64_t> point;
typedef bp::segment_data<int64_t> segment;
typedef bp::voronoi_edge<double> voronoi_edge;
typedef bp::voronoi_diagram<double> voronoi_diagram;

int get_edges(const std::vector<segment> &segments, std::vector<voronoi_edge> &edges)
{
    voronoi_diagram vd;
    std::vector<point> points;
    bp::construct_voronoi(points.begin(), points.end(), segments.begin(), segments.end(), &vd);

    return 0;
}