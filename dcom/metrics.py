'''
Created on 19 Nov 2012

@author: sorin
'''

import bisect
from modules import metrics_modules

"""
Base class for metrics

Includes loading configured modules in memory and returning the output.
"""
class Metrics:
    def __init__(self, net, options):
        self.net = net
        self.aggrPeriod = int(options['period'])
        self.tsList = []
        
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

    def getAllModules(self):
        modules = []
        modules.extend(self.storeModules.itervalues())
        modules.extend(self.metricsModules.itervalues())
        return modules
        
    def getNames(self):
        names = []
        for module in self.getOutputModules():
            names.extend(module.get_names())
            
        return names

    def getMetrics(self):
        self.updateNetworkAll()
        metrics_list = []
        for ts in self.tsList:
            ts_values = []
            for module in self.getOutputModules():
                ts_values.extend(module.get_values(ts))
            metrics_list.append(ts_values)
            
        return metrics_list

    def getOutputModules(self):
        modules = []
        for name in self.columns:
            if name in self.metricsModules:
                modules.append(self.metricsModules[name])
        return modules
        
    def updateNetworkAll(self):
        if len(self.tsList) > 0:
            prevTs = self.tsList[len(self.tsList)-1]
            
            for module in self.getAllModules():
                module.update_from_network(prevTs)
    
    def updateLinkAll(self, nodeA, nodeB, ts):
        for module in self.getAllModules():
            module.update_link(nodeA, nodeB, ts)
        
    def updateNodeAll(self, node, ntype, ts):
        for module in self.getAllModules():
            module.update_node(node, ntype, ts)
        
    def resetAll(self, ts):
        for module in self.getAllModules():
            module.reset(ts)
        

"""
Metrics are update based on absolute (since birth of network) time
"""
class MetricsAbsolute(Metrics):
    def __init__(self, net, options):
        Metrics.__init__(self, net, options)

    def newEventPre(self, nodeA, nodeB, ts):
        tsAggr = int(ts/ self.aggrPeriod)
        if not tsAggr in self.tsList:
            # get stats fron network for previous timestamp
            self.updateNetworkAll()
            self.tsList.append(tsAggr)
            # reset all modules
            self.resetAll(tsAggr)
                
        # update modules
        self.updateLinkAll(nodeA, nodeB, tsAggr)
                    
    def newEventPost(self, nodeA, nodeB, ts):
        pass
            

"""
Metrics are updated based on relative (birth of node) time
"""
class MetricsRelative(Metrics):
    def __init__(self, net, options):
        Metrics.__init__(self, net, options)

    def newEventPre(self, nodeA, nodeB, ts):
        self.__updateNodeStats(nodeA, 0, ts)
        self.__updateNodeStats(nodeB, 1, ts)
                    
    def __updateNodeStats(self, node, ntype, ts):
        tsAggr = int(self.__relativeTs(node, ts)/ self.aggrPeriod)
        if not tsAggr in self.tsList:
            bisect.insort(self.tsList, tsAggr)
        # update modules
        self.updateNodeAll(node, ntype, tsAggr)
        
    def newEventPost(self, nodeA, nodeB, ts):
        pass
            
    def __relativeTs(self, node, ts):
        tsRel = 0
        if self.net.hasNode(node):
            tsRel = ts - self.net.getNodeTs(node)
        return tsRel
