#!/usr/bin/env python

import numpy
import scipy

# Some helping functions
def increment(key, vmap, val=1):
    vmap[key] += val

def avg_values(vmap):
    if not vmap:
        return None
    return 0 if len(vmap)==0 else round(sum(vmap.values()) / len(vmap), 3)

def check_and_increment(key, vmap, val=1):
    if not key in vmap:
        vmap[key] = 0
    vmap[key] += val

def add_to_list(key, lmap, val):
    if not key in lmap:
        lmap[key] = []
    lmap[key].append(val)

def return_from_map(vmap, key):
    return vmap[key] if key in vmap else None
    
def distribution(vmap, binsz=1, maxv=0):
    d = vmap.values()
    if len(d)==0:
        return "None"
    max_d = max(d)
    if max_d == 0:
        return "0"
    if maxv!=0:
        max_d = maxv
    hist = numpy.histogram(d, [x * binsz for x in range(0, 1+int(max_d/binsz))], (0, max_d), False, None, True)[0]
#    rndhist = map(round, hist, [3]*len(hist))
    return ",".join(map(_str_nozero, hist)) 
#    numpy.histogram(d, [x * 0.1 for x in range(0, 10)], (0, 1), False)[0])

def correlation_list(vmap):
    maxkey = max(vmap.keys())
    values = []
    for i in range(0, maxkey+1):
        if i in vmap:
            values.append(vmap[i])
        else:
            values.append(0)
    return ",".join(map(_str_nozero, values))

def _str_nozero(val):
    return str(val) if val!=0 else ""

def sum_and_count(vmap, key, val, count=1):
    (sumval, sumcount) = (0, 0)
    if key in vmap:
        (sumval, sumcount) = vmap[key]

    vmap[key] = (sumval+val, sumcount+count)

def get_avg_map(vmap, rnd=1):
    for k, v in vmap.iteritems():
        vmap[k] = 1.0*v[0]/v[1]
    return vmap

def update_lifetime(key, vmap, vmapAll, curTs):
    if not key in vmapAll:
        vmapAll[key] = (curTs, 0)
    else:
        lf = vmapAll[key]
        vmapAll[key] = (lf[0], curTs-lf[0])
#        vmap[key] = vmapAll[key]

def avg_lifetime(vmap):
    vals = list(l[1] for l in vmap.values())
#        print len(vals)
    if len(vals)>0:
        return round(sum(vals)/len(vals), 3)
    else:
        return 0

def correlation(vmap1, vmap2):
    x = []
    y = []
    for key in vmap1:
        x.append(vmap1[key])
        y.append(vmap2[key])
    return round(scipy.stats.pearsonr(x, y)[0], 3)

def print_map(vmap):
    for k,v in sorted(vmap.iteritems()):
        print k,v
