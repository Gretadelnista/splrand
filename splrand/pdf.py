# -*- coding: utf-8 -*-
#
# Copyright (C) 2019 Luca Baldini (luca.baldini@pi.infn.it)
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Core logic for the pdf definition.
"""

import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt

### Essenzialmente voglio estendere le funzionalità di una spline in modo tale che generi anche numeri random... quindi conviene utilizzare il concetto di eridarietà: in questo modo ho già a disposizione tutti i metodi della classe madre InterpolatedUnivariateSpline.

### La pdf è una funzione che per noi, essenzialmente, è una spline.
### I numeri random li genero col metodo della Inverse Transform => ho bisogno della cdf.
### La ppf è l'inverso della cdf => basta inverstire gli argomenti della spline (= ruotare il grafico di 90°)
### A seconda della pdf, ycdf può assumere valori uguali => devo usare np.unique per farmi restituire un array con valori unici, altrimenti ho problemi nell'usare la spline e calcolare la ppf...


class ProbabilityDensityFunction(InterpolatedUnivariateSpline):
    ''' Class describing a probability density function.
        
    Parameters
    ----------
    x : array-like
        The array of x values to be passed to the pdf.
     
    y : array-like
        The array of y values to be passed to the pdf.
    '''

    def __init__(self, x, y, k=3):
        '''Constructor.
        '''
        InterpolatedUnivariateSpline.__init__(self, x, y, k=k)
        ycdf = np.array([self.integral(x[0], xcdf) for xcdf in x])
        self.cdf = InterpolatedUnivariateSpline(x, ycdf)
        # Need to make sure that the vector I'm passing to the ppf sline as
        # the x values has no duplicates---and need to filter the y
        # accordingly.
        _x, _i  = np.unique(ycdf, return_index=True)
        _y = x[_i]
        self.ppf = InterpolatedUnivariateSpline(_x, _y)

    def prob(self, x1, x2):
        '''Return the probability fro the random variable to be included
            between x1 and x2.
        
        Parameters
        ----------
        x1: flot or array-like
            The left bound for the integration.
        x2: float or array-like.
            The rigth bound for the integration.
        '''
        return self.cdf(x2) - self.cdf(x1)

    def rnd(self, size=1000):
        '''Return an array of random values from the pdf.
            
        Parameters
        ----------
        size: int
            The number of random numbers to extract.
        '''
        return self.ppf(np.random.uniform(size=size))
