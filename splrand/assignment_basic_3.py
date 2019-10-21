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
            between x1 and x2
        '''
        return self.cdf(x2) - self.cdf(x1)

    def rnd(self, size=1000):
        '''Return an array of random values from the pdf.
        '''
        return self.ppf(np.random.uniform(size=size))

def test_triangular():
    '''Unit test with a triangular distribution.
    '''
    x = np.linspace(0. ,1., 101)
    y = 2. * x
    pdf = ProbabilityDensityFunction(x, y)
    a = np.array([0.2, 0.6])
    print(pdf(a))

    plt.figure('pdf')
    plt.plot(x, pdf(x))
    plt.xlabel('x')
    plt.ylabel('pdf (x)')

    plt.figure('cdf')
    plt.plot(x, pdf.cdf(x))
    plt.xlabel('x')
    plt.ylabel('cdf (x)')
    
    plt.figure('ppf')
    q = np.linspace(0., 1., 250)
    plt.plot(q, pdf.ppf(q))
    plt.xlabel('q')
    plt.ylabel('ppf (q)')
    
    plt.figure('Sampling')
    rnd = pdf.rnd(100000)
    plt.hist(rnd, bins=1000)

def test_gaussian(mu=0., sigma=1., support=10., num_points=500):
    '''Unit test with a gaussian distribution
    '''
    from scipy.stats import norm
    x = np.linspace(-support * sigma, support * sigma, num_points)
    x += mu
    y = norm.pdf(x, mu, sigma)
    plt.plot(x,y)
    pdf = ProbabilityDensityFunction(x, y)
    plt.figure('pdf')
    plt.plot(x, pdf(x))
    plt.xlabel('x')
    plt.ylabel('pdf (x)')
    
    plt.figure('cdf')
    plt.plot(x, pdf.cdf(x))
    plt.xlabel('x')
    plt.ylabel('cdf (x)')
    
    plt.figure('ppf')
    q = np.linspace(0., 1., 250)
    plt.plot(q, pdf.ppf(q))
    plt.xlabel('q')
    plt.ylabel('ppf (q)')
    
    plt.figure('Sampling')
    rnd = pdf.rnd(100000)
    ydata, edges, _ = plt.hist(rnd, bins=1000)
    xdata = (edges[:-1] + edges[1:])/2

    def f(x, C, mu, sigma):
        return C * norm.pdf(x, mu, sigma)

    popt, pcov = curve_fit(f, xdata, ydata)
    print(popt)
    print(np.sqrt(pcov.diagonal()))
    _x = np.linspace(-10, 10, 500)
    _y = f(_x, *popt)
    plt.plot(_x, _y)

    mask = ydata > 0
    chi2 = sum(((ydata[mask] - f(xdata[mask], *popt)) / np.sqrt(ydata[mask]))**2)
    nu = mask.sum() - 3
    sigma = np.sqrt(2 * nu)
    print(chi2, nu, sigma)
    



if __name__=='__main__':
    test_gaussian()
    plt.show()
