'''
Created on 19 Nov 2012

@author: sorin
'''

import bisect
from tools import check_and_increment,avg_values

from modules import metrics_modules


class Metrics:
    def __init__(self, net, options):
        self.net = net
        self.tsList = []
        self.aggrPeriod = int(options['period'])
        
        self.columns = options['metrics-columns'].split(',')
        self.storeModules = self.__loadModules(options, options['store-modules'].split(','), options['metrics-modules'])
        self.metricsModules = self.__loadModules(options, self.columns, options['metrics-modules'], self.storeModules)

    def __loadModules(self, options, namesList, propertiesAll, dependencies=None):
        modules = {}
        for name in namesList:
            if name in metrics_modules and name in propertiesAll:
                properties = propertiesAll[name]
                # inject default dependencies
                properties['period'] = self.aggrPeriod
                properties['network'] = self.net
                # inject configurable dependencies
                if dependencies and 'depends' in properties:
                    for depends in properties['depends'].split(','):
                        if depends in dependencies:
                            properties[depends] = dependencies[depends]
                            print "Injected '" + depends + "' into '" + name + "'"
                modules[name] = metrics_modules[name](properties)
                print "Loaded module '" + name + "'"
        return modules

    def __getAllModules(self):
        modules = []
        modules.extend(self.storeModules.itervalues())
        modules.extend(self.metricsModules.itervalues())
        return modules
        
    def __getOutputModules(self):
        modules = []
        for name in self.columns:
            if name in self.metricsModules:
                modules.append(self.metricsModules[name])
        return modules
        
    def newEventPre(self, nodeA, nodeB, ts):
        tsAggr = int(ts/ self.aggrPeriod)
        if not tsAggr in self.tsList:
            # get stats fron network for previous timestamp
            self.__updateFromNetwork()
            self.tsList.append(tsAggr)
            # reset all modules
            for module in self.__getAllModules():
                module.reset(tsAggr)
                
        # update modules
        for module in self.__getAllModules():
            module.update(nodeA, nodeB, tsAggr)
                    
        
    def newEventPost(self, nodeA, nodeB, ts):
        pass
            
    def getMetrics(self):
        self.__updateFromNetwork()
        metrics_list = []
        for ts in self.tsList:
            ts_values = []
            for module in self.__getOutputModules():
                ts_values.extend(module.get_values(ts))
            metrics_list.append(ts_values)
            
        return metrics_list

    def getNames(self):
        names = []
        for module in self.__getOutputModules():
            names.extend(module.get_names())
            
        return names

    def __updateFromNetwork(self):
        if len(self.tsList) > 0:
            prevTs = self.tsList[len(self.tsList)-1]
            
            for module in self.__getAllModules():
                module.update_from_network(prevTs)


class MetricsRelative:
    def __init__(self, net, options):
        self.net = net
        self.aggrPeriod = int(options['period'])
        self.tsList = []
        self.nodeDegrees = [dict(), dict()]
        self.nodeStrength = [dict(), dict()]

    def newEventPre(self, nodeA, nodeB, ts):
        self.__updateNodeStats(nodeA, 0, ts)
        self.__updateNodeStats(nodeB, 1, ts)

    def newEventPost(self, nodeA, nodeB, ts):
#        self.__updateNodeStats(nodeA, 0, ts)
#        self.__updateNodeStats(nodeB, 1, ts)
        pass

    def __updateNodeStats(self, node, ntype, ts):
        tsAggr = int(self.__relativeTs(node, ts)/ self.aggrPeriod)
        if not tsAggr in self.tsList:
            self.__init_maps(tsAggr)
            bisect.insort(self.tsList, tsAggr)

        check_and_increment(node, self.nodeStrength[ntype][tsAggr])
        if self.net.hasNode(node):
            self.nodeDegrees[ntype][node][tsAggr] = self.net.getDegree(node)
        else:
            self.nodeDegrees[ntype][node] = dict()
            self.nodeDegrees[ntype][node][tsAggr] = 1
    
    def getMetrics(self):
        metrics_list = []
        degreesTsMap0 = self.__convertToTs(self.nodeDegrees[0])
        degreesTsMap1 = self.__convertToTs(self.nodeDegrees[1])
        for ts in self.tsList:
            metrics_list.append((ts+1,  
                                avg_values(degreesTsMap0[ts]), avg_values(degreesTsMap1[ts]),
                                avg_values(self.nodeStrength[0][ts]), avg_values(self.nodeStrength[1][ts]),
                                ))
        return metrics_list

    def getNames(self):
        return ("Date", "Avg node degree 1", "Avg node degree 2", "Avg node strength 1", "Avg node strength 2" )  

    def __init_maps(self, ts):
        self.nodeStrength[0][ts] = dict()
        self.nodeStrength[1][ts] = dict()

    def __relativeTs(self, node, ts):
        tsRel = 0
        if self.net.hasNode(node):
            tsRel = ts - self.net.getNodeTs(node)
        return tsRel
        
    def __convertToTs(self, vmap):
        tsmapall = dict()
        for node, tsmap in vmap.iteritems():
            for ts, val in tsmap.iteritems():
                if not ts in tsmapall:
                    tsmapall[ts] = dict()
                    tsmapall[ts][node] = val
        return tsmapall


