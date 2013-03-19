'''
Created on 19 Nov 2012

@author: sorin
'''

import bisect
from modules import metrics_modules

METRICS_PRE=0
METRICS_POST=1

"""
Base class for metrics

Includes loading configured modules in memory and returning the output.
"""
class Metrics:
    def __init__(self, net, options):
        self.net = net
        self.aggrPeriod = int(options['period'])
        self.tsList = [[], []]
        self.columns = [[], []]
        self.storeModules = [[], []]
        self.metricsModules = [[], []]
        opt_columns = ['metrics-columns', 'post-metrics-columns']
        opt_store = ['store-modules', 'post-store-modules']
        
        for i in [METRICS_PRE, METRICS_POST]:
            self.columns[i] = self.__getListFromConfig(options, opt_columns[i])
            self.storeModules[i] = self.__loadModules(options, 
                                            self.__getListFromConfig(options,opt_store[i]), options['metrics-modules'])
            self.metricsModules[i] = self.__loadModules(options, self.columns[i], options['metrics-modules'], self.storeModules[i])
    
    def __getListFromConfig(self, options, config):
        return options[config].split(',') if config in options and options[config] else []
        
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

    def getAllModules(self, phase):
        modules = []
        modules.extend(self.storeModules[phase].itervalues())
        modules.extend(self.metricsModules[phase].itervalues())
        return modules
        
    def getNames(self, phase):
        names = []
        for module in self.getOutputModules(phase):
            names.extend(module.get_names())
            
        return names

    def getMetrics(self):
        metrics = []
        for i in [METRICS_PRE, METRICS_POST]: 
            metrics_list = [self.getNames(i)]
            self.updateNetworkAll(i)
            for ts in self.tsList[i]:
                ts_values = []
                for module in self.getOutputModules(i):
                    ts_values.extend(module.get_values(ts))
                metrics_list.append(ts_values)
            metrics.append(metrics_list)
            
        return metrics

    def getOutputModules(self, phase):
        modules = []
        for name in self.columns[phase]:
            if name in self.metricsModules[phase]:
                modules.append(self.metricsModules[phase][name])
        return modules
        
    def updateNetworkAll(self, phase):
        if len(self.tsList[phase]) > 0:
            prevTs = self.tsList[phase][len(self.tsList[phase])-1]
            
            for module in self.getAllModules(phase):
                module.update_from_network(prevTs)
    
    def updateLinkAll(self, nodeA, nodeB, ts, phase):
        for module in self.getAllModules(phase):
            module.update_link(nodeA, nodeB, ts)
        
    def updateNodeAll(self, node, ntype, ts, phase):
        for module in self.getAllModules(phase):
            module.update_node(node, ntype, ts)
        
    def resetAll(self, ts, phase):
        for module in self.getAllModules(phase):
            module.reset(ts)
        

"""
Metrics are update based on absolute (since birth of network) time
"""
class MetricsAbsolute(Metrics):
    def __init__(self, net, options):
        Metrics.__init__(self, net, options)

    def newEventPre(self, nodeA, nodeB, ts):
        self.__newEvent(nodeA, nodeB, ts, METRICS_PRE)
                    
    def newEventPost(self, nodeA, nodeB, ts):
        self.__newEvent(nodeA, nodeB, ts, METRICS_POST)
        pass

    def __newEvent(self, nodeA, nodeB, ts, phase):
        tsAggr = int(ts/ self.aggrPeriod)
        if not tsAggr in self.tsList[phase]:
            # get stats fron network for previous timestamp
            self.updateNetworkAll(phase)
            self.tsList[phase].append(tsAggr)
            # reset all modules
            self.resetAll(tsAggr, phase)
                
        # update modules
        self.updateLinkAll(nodeA, nodeB, tsAggr, phase)

"""
Metrics are updated based on relative (birth of node) time
"""
class MetricsRelative(Metrics):
    def __init__(self, net, options):
        Metrics.__init__(self, net, options)

    def newEventPre(self, nodeA, nodeB, ts):
        self.__updateNodeStats(nodeA, 0, ts, METRICS_PRE)
        self.__updateNodeStats(nodeB, 1, ts, METRICS_PRE)
                    
    def __updateNodeStats(self, node, ntype, ts, phase):
        tsAggr = int(self.__relativeTs(node, ts)/ self.aggrPeriod)
        if not tsAggr in self.tsList[phase]:
            bisect.insort(self.tsList[phase], tsAggr)
        # update modules
        self.updateNodeAll(node, ntype, tsAggr, phase)
        
    def newEventPost(self, nodeA, nodeB, ts):
        pass
            
    def __relativeTs(self, node, ts):
        tsRel = 0
        if self.net.hasNode(node):
            tsRel = ts - self.net.getNodeTs(node)
        return tsRel
