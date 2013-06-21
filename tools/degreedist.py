#!/usr/bin/python
import sys
import numpy as np
import powerlaw as pl
#import pylab

def read_data(fname):
    lines = open(fname).readlines()
    data = []
    for line in lines:
        data.append(int(line.strip()))
    return data
    
inFile = sys.argv[1]

data=read_data(inFile)
#pl.plot_pdf(data, color='b')
fit = pl.Fit(data, discrete=True, discrete_approximation='round')

print "Power law:", fit.power_law.xmin, fit.power_law.alpha, fit.power_law.sigma
print "Lognormal:", fit.lognormal.mu
R, p = fit.distribution_compare('lognormal', 'power_law', normalized_ratio=True)
print R, p
#pl.plot_pdf(data, color='b')
#pylab.savefig("foo.png")
#print fit.supported_distributions

