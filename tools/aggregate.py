#!/usr/bin/python
import sys
import os

def filetodict(fname, columns):
    lines = open(fname).readlines()
    if len(lines) < 2:
        return {}
    names = lines[0].split("\t")
    vals = lines[1].split("\t") if len(lines)>1 else []
    valsMap = {}
    for i in range(1,len(names)):
        if len(columns)== 0 or names[i].strip() in columns:
            valsMap[names[i].strip()] = vals[i].strip() if len(vals) > i else ""
    return valsMap

def writeheaders(fout, columns, delim="\t"):
    line = "Thresholds"  
    for col in columns:
        line += "%s%s%s%s" % (delim, col, delim, col + "-random")
    line += "\n"
    fout.write(line)
    

def writeresults(fout, threshold, columns, valsMap, rndValsMap={}, delim="\t"):
    line = str(threshold)  
    for col in columns:
        if not col in valsMap:
            line += "%s%s" % (delim, delim)
        else:
            line += "%s%s%s%s" % (delim, str(valsMap[col]), delim, str(rndValsMap[col]) if col in rndValsMap else "")
    line += "\n"
    fout.write(line)

def avgrandom(columns, randomDir, dataset, module, thr, maxIdx):
    rndAvgVals = {}
    for col in columns:
        rndAvgVals[col] = 0
    cnt = 0
    for rnd in range(1, maxIdx+1):
        rnd_thr_fname = randomDir + "/" + str(rnd) + "/output/output_random_" + dataset + "_tfidf_thr_" + str(thr) + "_" + module + ".csv"
        if not os.path.isfile(rnd_thr_fname):
            continue
        rndVals = filetodict(rnd_thr_fname, columns)
        if len(rndVals) == 0:
            continue
        cnt += 1
        for col in columns:
            rndAvgVals[col] += float(rndVals[col])
    
    if cnt == 0:
        return {}
    
    for col in columns:
        rndAvgVals[col] /= cnt

    return rndAvgVals
        
thresholds = [0.05, 0.1, 0.5, 0.7, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6]
modules = { "modularity" : ["Prj modularity users", "Prj modularity objects"],
           "general" : ["Link events", "Nodes count users", "Nodes count objects", "Links count prj users", "Links count prj objects"] }
modules_random = { "modularity" : "modularity", "general" : "general_random" }

if len(sys.argv) < 4 or not sys.argv[1] in modules:
    print "usage: agr.py <category> <dataset> <out file>"
    quit()

module = sys.argv[1]
columns = modules[module]

categ = sys.argv[2]
dataset = sys.argv[3]
dataDir = "data/" + categ
outputDir = dataDir + "/output"
randomDir = dataDir + "/random"
resultsDir = dataDir + "/results"
resultsFile = resultsDir + "/" + module + ".csv"
delimiter = "|"

if not os.path.isdir(resultsDir):
    os.mkdir(resultsDir)

fout = open(resultsFile, "w")

# read main values and write them to file
mainVals = filetodict(outputDir + "/output_" + dataset + "_" + module + ".csv", columns)
writeheaders(fout, columns, delimiter)
if len(columns) == 0:
    columns = sorted(mainVals.iterkeys())
writeresults(fout, 0, columns, mainVals, delim=delimiter)

for thr in thresholds:
    thr_fname = outputDir + "/output_" + dataset + "_tfidf_thr_" + str(thr) + "_" + module + ".csv"
    if not os.path.isfile(thr_fname):
        continue

    thrVals = filetodict(thr_fname, columns)
    if len(thrVals) == 0:
        continue
    
    rndAvgVals = avgrandom(columns, randomDir, dataset, modules_random[module], thr, 100)
    
    writeresults(fout, thr, columns, thrVals, rndAvgVals, delimiter)
    
print resultsFile

