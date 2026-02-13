/**
 *
 * Copyright (C) 2012-2018 by the DOpElib authors
 *
 * This file is part of DOpElib
 *
 * DOpElib is free software: you can redistribute it
 * and/or modify it under the terms of the GNU General Public
 * License as published by the Free Software Foundation, either
 * version 3 of the License, or (at your option) any later
 * version.
 *
 * DOpElib is distributed in the hope that it will be
 * useful, but WITHOUT ANY WARRANTY; without even the implied
 * warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 * PURPOSE.  See the GNU General Public License for more
 * details.
 *
 * Please refer to the file LICENSE.TXT included in this distribution
 * for further information on this license.
 *
 **/

#ifndef FUNCTIONS_
#define FUNCTIONS_

using namespace std;
using namespace dealii;

#include <deal.II/base/function.h>

namespace local
{
	inline double rhs(const Point<2> &p)
	{
        return (1 - p[1] * p[1]) * (6 * p[0] * p[0] + 2) + 2 * (1 - p[0] * p[0]);
    }

    class Obstacle : public Function<2>
	{
		public : Obstacle(double obstacle_value) : Function<2>(2)
    	{
            obstacle_value_ = obstacle_value;
        }

    	double value(const Point<2> &/*p*/, const unsigned int component /*= 0*/) const override
	    {
            if (component == 0)
            {
                return obstacle_value_;
            }
            return 1.;
        }
        double obstacle_value_;
    };

    class
    InitControl : public Function<2>
	{
    	public : InitControl() : Function<2>(3)
    	{
    	}

        double value(const Point<2> &/*p*/, const unsigned int component /* = 0*/ ) const override
        {
            if (component == 0)
            {
                return 0.51;
            }
    		if (component == 1)
    		{
                return 0;
            }
    		if (component == 2)
    		{
                return 0.51;
            }
            return 0;
        }
    };

    class
    DesState : public Function<2>
	{
		public : DesState() : Function<2>(2)
		{
		}

        double value(const Point<2> &p, const unsigned int component /* = 0 */ ) const override
        {
            const double x = p[0];
            const double y = p[1];

            if (component == 0)
            {
                return (1 - x * x) * (1 - y * y);
            }
            return 0;
        }
    };

    class
    DesControl : public Function<2>
	{
		public : DesControl() : Function<2>(3)
		{
		}

        double value(const Point<2> &/*p*/, const unsigned int component /* = 0 */ ) const override
        {
            if (component == 0)
            {
                return 0;
            }
            if (component == 1)
            {
                return 0;
            }
            if (component == 2)
            {
                return 0;
            }

            return 0;
        }
    };
}


#endif
