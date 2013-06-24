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
fit = pl.Fit(data, discrete=True, discrete_approximation='round', xmin=None)

print "Power law:", "xmin", fit.power_law.xmin, "| alpha", fit.power_law.alpha, "| sigma", fit.power_law.sigma
print "Lognormal: ", fit.lognormal.parameter1_name, fit.lognormal.parameter1, "|", \
                        fit.lognormal.parameter2_name, fit.lognormal.parameter2
print "Exponential: ", fit.exponential.parameter1_name, fit.exponential.parameter1
print "Stretched exponential: ", fit.stretched_exponential.parameter1_name, fit.stretched_exponential.parameter1, "|", \
                                fit.stretched_exponential.parameter2_name, fit.stretched_exponential.parameter2

print
R, p = fit.distribution_compare('power_law', 'exponential', normalized_ratio=True)
print "power_law - exponential:", R, p
R, p = fit.distribution_compare('power_law', 'stretched_exponential', normalized_ratio=True)
print "power_law - stretched_exponential:", R, p
R, p = fit.distribution_compare('power_law', 'lognormal', normalized_ratio=True)
print "power_law - lognormal:", R, p
R, p = fit.distribution_compare('stretched_exponential', 'exponential', normalized_ratio=True)
print "stretched_exponential - exponential:", R, p
R, p = fit.distribution_compare('stretched_exponential', 'lognormal', normalized_ratio=True)
print "stretched_exponential - lognormal:", R, p
R, p = fit.distribution_compare('lognormal', 'exponential', normalized_ratio=True)
print "lognormal - exponential:", R, p

#pl.plot_pdf(data, color='b')
#pylab.savefig("foo.png")
#print fit.supported_distributions

