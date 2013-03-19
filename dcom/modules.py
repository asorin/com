#!/usr/bin/env python

import abc
from tools import *
from datetime import datetime

class MetricsModule(object):

    __metaclass__ = abc.ABCMeta

    def __init__(self, conf):
        """Initialize module.
        """
        self.names = conf['title'].split(',') if 'title' in conf else [] 
        self.net = conf['network']

    @abc.abstractmethod
    def reset(self, ts):
        """Reset metrics values.
        """

    @abc.abstractmethod
    def update_link(self, nodeA, nodeB, ts):
        """Update module with a new link
        """

    @abc.abstractmethod
    def update_node(self, nodeA, nodeB, ts):
        """Update module with a new link
        """

    @abc.abstractmethod
    def update_from_network(self, ts):
        """Update module from network.
        """

    @abc.abstractmethod
    def get_values(self):
        """Ger values of columns.
        """
        
    def get_names(self):
        """Get title of columns.
        """
        return self.names


"""
Store maps to be used by metrics 
"""

class NodesDegrees(MetricsModule):

    def __init__(self, conf):
        MetricsModule.__init__(self, conf)
        self.nodesDegrees = [dict(), dict()]

    def reset(self, ts):
        pass

    def update_link(self, nodeA, nodeB, ts):
        pass

    def update_node(self, node, ntype, ts):
        if not ts in self.nodesDegrees[ntype]:
            self.nodesDegrees[ntype][ts] = dict()
            
        self.nodesDegrees[ntype][ts][node] = self.net.getDegree(node) if self.net.hasNode(node) else 1 

    def update_from_network(self, ts):
        if not ts in self.nodesDegrees[0]:
            d1, d2 = self.net.getDegrees()
            self.nodesDegrees[0][ts] = d1
            self.nodesDegrees[1][ts] = d2

    def get_values(self, ts):
        return [return_from_map(self.nodesDegrees[0], ts), return_from_map(self.nodesDegrees[1], ts)]


class NodesStrength(MetricsModule):

    def __init__(self, conf):
        MetricsModule.__init__(self, conf)
        self.nodesStrength = [dict(), dict()]

    def reset(self, ts):
        pass

    def update_link(self, nodeA, nodeB, ts):
        self.update_node(nodeA, 0, ts)
        self.update_node(nodeB, 1, ts)

    def update_node(self, node, ntype, ts):
        if not ts in self.nodesStrength[ntype]:
            self.nodesStrength[ntype][ts] = dict()
        check_and_increment(node, self.nodesStrength[ntype][ts])

    def update_from_network(self, ts):
        pass

    def get_values(self, ts):
        return [return_from_map(self.nodesStrength[0], ts), return_from_map(self.nodesStrength[1], ts)]


class NeighbourHubsSingles(MetricsModule):

    def __init__(self, conf):
        MetricsModule.__init__(self, conf)
        self.nodeHubs = [dict(), dict()]
        self.nodeHubsPct = [dict(), dict()]
        self.nodeSingles = [dict(), dict()]
        self.nodeSinglesPct = [dict(), dict()]

    def reset(self, ts):
        pass

    def update_link(self, nodeA, nodeB, ts):
        pass

    def update_node(self, node, ntype, ts):
        pass

    def update_from_network(self, ts):
        self.__update_maps(0, ts)
        self.__update_maps(1, ts)

    def __update_maps(self, ntype, ts):
        hubs, _, singles, _ = self.net.getCountsHubsSingles(ntype, self.net.getNodesCount(ntype) * 0.10)
        self.nodeHubs[ntype][ts] = hubs
        self.nodeSingles[ntype][ts] = singles
        
    def get_values(self, ts):
        return [self.nodeHubs[0][ts], self.nodeHubs[1][ts], self.nodeSingles[0][ts], self.nodeSingles[1][ts]]


class DegreeLinksCount(MetricsModule):

    def __init__(self, conf):
        MetricsModule.__init__(self, conf)
        self.degreeLinksCount = [dict(), dict()]
        self.degreeNewLinksCount = [dict(), dict()]
        self.degreeReinfLinksCount = [dict(), dict()]
        
    def reset(self, ts):
        self.degreeLinksCount[0][ts] = dict()
        self.degreeLinksCount[1][ts] = dict()
        self.degreeNewLinksCount[0][ts] = dict()
        self.degreeNewLinksCount[1][ts] = dict()
        self.degreeReinfLinksCount[0][ts] = dict()
        self.degreeReinfLinksCount[1][ts] = dict()

    def update_link(self, nodeA, nodeB, ts):
        hasNodeA = self.net.hasNode(nodeA)
        hasNodeB = self.net.hasNode(nodeB)
        
        if hasNodeA:
            check_and_increment(self.net.getDegree(nodeA), self.degreeLinksCount[0][ts])
        if hasNodeB:
            check_and_increment(self.net.getDegree(nodeB), self.degreeLinksCount[1][ts])
            
        if self.net.hasEdge(nodeA, nodeB):
            check_and_increment(self.net.getDegree(nodeA), self.degreeReinfLinksCount[0][ts])
            check_and_increment(self.net.getDegree(nodeB), self.degreeReinfLinksCount[1][ts])
        else:
            if hasNodeA:
                check_and_increment(self.net.getDegree(nodeA), self.degreeNewLinksCount[0][ts])
            if hasNodeB:
                check_and_increment(self.net.getDegree(nodeB), self.degreeNewLinksCount[1][ts])

    def update_node(self, node, ntype, ts):
        pass

    def update_from_network(self, ts):
        pass

    def get_values(self, ts):
        return [self.degreeLinksCount[0][ts], self.degreeLinksCount[1][ts],
                self.degreeNewLinksCount[0][ts], self.degreeNewLinksCount[1][ts],
                self.degreeReinfLinksCount[0][ts], self.degreeReinfLinksCount[1][ts]]


class AvgNeighbourDegree(MetricsModule):

    def __init__(self, conf):
        MetricsModule.__init__(self, conf)
        self.nodeAvgDegreeNeigh = [dict(), dict()]
        self.nodeAvgDegreeCorr = [dict(), dict()]

    def reset(self, ts):
        pass

    def update_link(self, nodeA, nodeB, ts):
        pass

    def update_from_network(self, ts):
        avgDegree0, avgDegreeCorr0 = self.net.getAvgNeighbourDegree(0)
        self.nodeAvgDegreeNeigh[0][ts] = avgDegree0
        self.nodeAvgDegreeCorr[0][ts] = avgDegreeCorr0
        avgDegree1, avgDegreeCorr1 = self.net.getAvgNeighbourDegree(1)
        self.nodeAvgDegreeNeigh[1][ts] = avgDegree1
        self.nodeAvgDegreeCorr[1][ts] = avgDegreeCorr1

    def update_node(self, node, ntype, ts):
        pass

    def get_values(self, ts):
        if not ts in self.nodeAvgDegreeNeigh[0]:
            return []
        
        return [self.nodeAvgDegreeNeigh[0][ts], self.nodeAvgDegreeNeigh[1][ts],
                self.nodeAvgDegreeCorr[0][ts], self.nodeAvgDegreeCorr[1][ts]]


class ClustCoef(MetricsModule):

    def __init__(self, conf):
        MetricsModule.__init__(self, conf)
        self.clustCoef = [dict(), dict()]

    def reset(self, ts):
        pass

    def update_link(self, nodeA, nodeB, ts):
        pass

    def update_node(self, node, ntype, ts):
        pass

    def update_from_network(self, ts):
        self.clustCoef[0][ts] = self.net.getClustCoef(0)
        self.clustCoef[1][ts] = dict()#self.net.getClustCoef(1)

    def get_values(self, ts):
        if not ts in self.clustCoef[0]:
            return []

        return [self.clustCoef[0][ts], self.clustCoef[1][ts]]


class ClustCoefOpsahl(MetricsModule):

    def __init__(self, conf):
        MetricsModule.__init__(self, conf)
        self.globalCoefMap = [dict(), dict()]
        self.localCoefMap = [dict(), dict()]
        self.globalCoefMapDegree = [dict(), dict()]
        self.localCoefMapDegree = [dict(), dict()]

    def reset(self, ts):
        pass

    def update_link(self, nodeA, nodeB, ts):
        pass

    def update_node(self, node, ntype, ts):
        pass

    def update_from_network(self, ts):
        globalCoef, localCoef, globalCoefDegree, localCoefDegree = self.net.getClustCoefOpsahlAll(0)
#        globalCoef, localCoef = self.net.getClustCoefOpsahlOriginal(0)
        self.globalCoefMap[0][ts] = globalCoef
        self.localCoefMap[0][ts] = localCoef
        self.globalCoefMapDegree[0][ts] = globalCoefDegree
        self.localCoefMapDegree[0][ts] = localCoefDegree
        
        self.globalCoefMap[1][ts] = 0
        self.localCoefMap[1][ts] = dict()
        self.globalCoefMapDegree[1][ts] = 0
        self.localCoefMapDegree[1][ts] = dict()

    def get_values(self, ts):
        if not ts in self.globalCoefMap[0]:
            return []

        return [self.globalCoefMap[0][ts],self.localCoefMap[0][ts], self.globalCoefMap[1][ts], self.localCoefMap[1][ts],
                self.globalCoefMapDegree[0][ts],self.localCoefMapDegree[0][ts], self.globalCoefMapDegree[1][ts], self.localCoefMapDegree[1][ts]]


class ClustCoefDegree(MetricsModule):

    def __init__(self, conf):
        MetricsModule.__init__(self, conf)
        self.degreeClustCoef = dict()
        self.diffClustCoef = dict()

    def reset(self, ts):
        self.degreeClustCoef[ts] = dict()
        self.diffClustCoef[ts] = dict()

    def update_link(self, nodeA, nodeB, ts):
#        self.degreeClustCoef[ts][nodeA] = self.net.getNewClusterCoef(nodeA)
        pass

    def update_node(self, node, ntype, ts):
        pass

    def update_from_network(self, ts):
        self.degreeClustCoef[ts] = self.net.getGlobalDegreeClusterCoef(0)
#        for n, c in self.net.getClustCoef(0).iteritems():
#            dc = self.degreeClustCoef[ts][n]
#            if dc>0.4 and abs(c-dc)>0.2:
#                print n, round(c,3), round(dc,3)
#            self.diffClustCoef[ts][n] = c - dc
        pass

    def get_values(self, ts):
        return [self.degreeClustCoef[ts], self.diffClustCoef[ts]]


"""
Main metrics modules 
"""

class NewNodes(MetricsModule):
    
    def __init__(self, conf):
        MetricsModule.__init__(self, conf)
        self.newNodes = [dict(), dict()]
    
    def reset(self, ts):
        self.newNodes[0][ts] = 0
        self.newNodes[1][ts] = 0
        
    def update_link(self, nodeA, nodeB, ts):
        if not self.net.hasNode(nodeA):
            increment(ts, self.newNodes[0])
        if not self.net.hasNode(nodeB):
            increment(ts, self.newNodes[1])

    def update_node(self, node, ntype, ts):
        pass

    def update_from_network(self, ts):
        pass
    
    def get_values(self, ts):
        if ts in self.newNodes[0]:
            return [self.newNodes[0][ts], self.newNodes[1][ts]]
        else:
            return []
    
class Date(MetricsModule):
    
    def __init__(self, conf):
        MetricsModule.__init__(self, conf)
        self.aggrPeriod = conf['period']
    
    def reset(self, ts):
        pass
        
    def update_link(self, nodeA, nodeB, ts):
        pass

    def update_node(self, node, ntype, ts):
        pass

    def update_from_network(self, ts):
        pass
    
    def get_values(self, ts):
        return [datetime.utcfromtimestamp(ts*self.aggrPeriod)]


class DateRelative(MetricsModule):
    
    def __init__(self, conf):
        MetricsModule.__init__(self, conf)
        self.aggrPeriod = conf['period']
    
    def reset(self, ts):
        pass
        
    def update_link(self, nodeA, nodeB, ts):
        pass

    def update_node(self, node, ntype, ts):
        pass

    def update_from_network(self, ts):
        pass
    
    def get_values(self, ts):
        return [ts]


class NodesCount(MetricsModule):

    def __init__(self, conf):
        MetricsModule.__init__(self, conf)
        self.nodesCount = [dict(), dict()]

    def reset(self, ts):
        self.nodesCount[0][ts] = 0
        self.nodesCount[1][ts] = 0

    def update_link(self, nodeA, nodeB, ts):
        pass

    def update_node(self, node, ntype, ts):
        pass

    def update_from_network(self, ts):
        self.nodesCount[0][ts] = self.net.getNodesCount(0)
        self.nodesCount[1][ts] = self.net.getNodesCount(1)

    def get_values(self, ts):
        if ts in self.nodesCount[0]:
            return [self.nodesCount[0][ts], self.nodesCount[1][ts]]
        else:
            return []


class AvgNodeDegree(MetricsModule):

    def __init__(self, conf):
        MetricsModule.__init__(self, conf)
        self.degrees = conf['nodes-degrees']

    def reset(self, ts):
        pass

    def update_link(self, nodeA, nodeB, ts):
        pass

    def update_node(self, node, ntype, ts):
        pass

    def update_from_network(self, ts):
        pass

    def get_values(self, ts):
        [d1, d2] = self.degrees.get_values(ts)
        return [avg_values(d1), avg_values(d2)]


class AvgNodesStrength(MetricsModule):

    def __init__(self, conf):
        MetricsModule.__init__(self, conf)
        self.strength = conf['nodes-strength']

    def reset(self, ts):
        pass

    def update_link(self, nodeA, nodeB, ts):
        pass

    def update_node(self, node, ntype, ts):
        pass

    def update_from_network(self, ts):
        pass

    def get_values(self, ts):
        [d1, d2] = self.strength.get_values(ts)
        return [avg_values(d1), avg_values(d2)]


class AvgNodeLifetime(MetricsModule):

    def __init__(self, conf):
        MetricsModule.__init__(self, conf)
        self.aggrPeriod = conf['period']
        self.nodeAvgLifetime = [dict(), dict()]

    def reset(self, ts):
        self.nodeAvgLifetime[0][ts] = 0
        self.nodeAvgLifetime[1][ts] = 0

    def update_link(self, nodeA, nodeB, ts):
        pass

    def update_node(self, node, ntype, ts):
        pass

    def update_from_network(self, ts):
        lf1, lf2 = self.net.getAvgLifetime()
        self.nodeAvgLifetime[0][ts] = lf1
        self.nodeAvgLifetime[1][ts] = lf2

    def get_values(self, ts):
        if ts in self.nodeAvgLifetime[0]:
            return [self.nodeAvgLifetime[0][ts]/self.aggrPeriod, self.nodeAvgLifetime[1][ts]/self.aggrPeriod]
        else:
            return []


class LinksCount(MetricsModule):

    def __init__(self, conf):
        MetricsModule.__init__(self, conf)
        self.linkCount = dict()
        self.reinfLinkCount = dict()

    def reset(self, ts):
        self.linkCount[ts] = 0
        self.reinfLinkCount[ts] = 0

    def update_link(self, nodeA, nodeB, ts):
        increment(ts, self.linkCount)
        if self.net.hasEdge(nodeA, nodeB):
            increment(ts, self.reinfLinkCount)

    def update_node(self, node, ntype, ts):
        pass

    def update_from_network(self, ts):
        pass

    def get_values(self, ts):
        ret = []
        ret.append(self.linkCount[ts] if ts in self.linkCount else 0)
        ret.append(self.reinfLinkCount[ts] if ts in self.reinfLinkCount else 0)
        return ret


class AvgNewLinkShortestPath(MetricsModule):

    def __init__(self, conf):
        MetricsModule.__init__(self, conf)
        self.linkShortestPath = dict()

    def reset(self, ts):
        pass

    def update_link(self, nodeA, nodeB, ts):
        if self.net.hasNode(nodeA) and self.net.hasNode(nodeB) and self.net.hasPath(nodeA, nodeB):
            add_to_list(ts, self.linkShortestPath, self.net.getShortestPath(nodeA, nodeB))

    def update_node(self, node, ntype, ts):
        pass

    def update_from_network(self, ts):
        pass

    def get_values(self, ts):
        if ts in self.linkShortestPath:
            return [round(scipy.mean(self.linkShortestPath[ts]), 3),]
        else:
            return []

class DegreeDistribution(MetricsModule):

    def __init__(self, conf):
        MetricsModule.__init__(self, conf)
        self.degrees = conf['nodes-degrees']

    def reset(self, ts):
        pass

    def update_link(self, nodeA, nodeB, ts):
        pass

    def update_node(self, node, ntype, ts):
        pass

    def update_from_network(self, ts):
        pass

    def get_values(self, ts):
        [d1, d2] = self.degrees.get_values(ts)
        return [distribution(d1), distribution(d2)]


class HubsSinglesDistribution(MetricsModule):

    def __init__(self, conf):
        MetricsModule.__init__(self, conf)
        self.hubssingles = conf['neighbour-hubs-singles']

    def reset(self, ts):
        pass

    def update_link(self, nodeA, nodeB, ts):
        pass

    def update_node(self, node, ntype, ts):
        pass

    def update_from_network(self, ts):
        pass

    def get_values(self, ts):
        [hubs1, hubs2, singles1, singles2] = self.hubssingles.get_values(ts)
        return [distribution(hubs1), distribution(hubs2), distribution(singles1), distribution(singles2)]


class DegreeLinksCountCorrelation(MetricsModule):

    def __init__(self, conf):
        MetricsModule.__init__(self, conf)
        self.degreeLinksCount = conf['degree-links-count']

    def reset(self, ts):
        pass

    def update_link(self, nodeA, nodeB, ts):
        pass

    def update_node(self, node, ntype, ts):
        pass

    def update_from_network(self, ts):
        pass

    def get_values(self, ts):
        [dl1, dl2, dn1, dn2, dr1, dr2] = self.degreeLinksCount.get_values(ts)
        return [correlation_list(dl1), correlation_list(dl2),
                correlation_list(dn1), correlation_list(dn2),
                correlation_list(dr1), correlation_list(dr2)]


class AvgNeighbourDegreeDistribution(MetricsModule):

    def __init__(self, conf):
        MetricsModule.__init__(self, conf)
        self.avgNeighbourDegree = conf['avg-neighbour-degree']

    def reset(self, ts):
        pass

    def update_link(self, nodeA, nodeB, ts):
        pass

    def update_node(self, node, ntype, ts):
        pass

    def update_from_network(self, ts):
        pass

    def get_values(self, ts):
        avgDg1, avgDg2, _, _ = self.avgNeighbourDegree.get_values(ts)
        return [distribution(avgDg1), distribution(avgDg2)]


class DegreeAvgNeighbourDegreeCorrelation(MetricsModule):

    def __init__(self, conf):
        MetricsModule.__init__(self, conf)
        self.avgNeighbourDegree = conf['avg-neighbour-degree']

    def reset(self, ts):
        pass

    def update_link(self, nodeA, nodeB, ts):
        pass

    def update_node(self, node, ntype, ts):
        pass

    def update_from_network(self, ts):
        pass

    def get_values(self, ts):
        _, _, avgDgCorr1, avgDgCorr2 = self.avgNeighbourDegree.get_values(ts)
        return [correlation_list(avgDgCorr1), correlation_list(avgDgCorr2)]


class AvgClustCoef(MetricsModule):

    def __init__(self, conf):
        MetricsModule.__init__(self, conf)
        self.clustCoef = conf['clust-coef']

    def reset(self, ts):
        pass

    def update_link(self, nodeA, nodeB, ts):
        pass

    def update_node(self, node, ntype, ts):
        pass

    def update_from_network(self, ts):
        pass

    def get_values(self, ts):
        coef1, coef2 = self.clustCoef.get_values(ts)
#        print_map(coef1)
        return [avg_values(coef1), avg_values(coef2)]


class ClustCoefDistribution(MetricsModule):

    def __init__(self, conf):
        MetricsModule.__init__(self, conf)
        self.clustCoef = conf['clust-coef']

    def reset(self, ts):
        pass

    def update_link(self, nodeA, nodeB, ts):
        pass

    def update_node(self, node, ntype, ts):
        pass

    def update_from_network(self, ts):
        pass

    def get_values(self, ts):
        coef1, coef2 = self.clustCoef.get_values(ts)
        return [distribution(coef1, 0.1, 1), distribution(coef2, 0.1, 1)]


class AvgClustCoefOpsahl(MetricsModule):

    def __init__(self, conf):
        MetricsModule.__init__(self, conf)
        self.clustCoef = conf['clust-coef-opsahl']

    def reset(self, ts):
        pass

    def update_link(self, nodeA, nodeB, ts):
        pass

    def update_node(self, node, ntype, ts):
        pass

    def update_from_network(self, ts):
        pass

    def get_values(self, ts):
        global1, _, global2, _, globalDg1, _, globalDg2, _ = self.clustCoef.get_values(ts)
        return [global1, global2, globalDg1, globalDg2]


class DegreeClustCoefCorrelation(MetricsModule):

    def __init__(self, conf):
        MetricsModule.__init__(self, conf)
        self.clustCoef = conf['clust-coef']
        self.nodeDegrees = conf['nodes-degrees']

    def reset(self, ts):
        pass

    def update_link(self, nodeA, nodeB, ts):
        pass

    def update_node(self, node, ntype, ts):
        pass

    def update_from_network(self, ts):
        pass

    def get_values(self, ts):
        coef1, coef2 = self.clustCoef.get_values(ts)
        degrees1, degrees2 = self.nodeDegrees.get_values(ts)
        degreeCoefCorr1, degreeCoefCorr2 = dict(), dict()
        
        for node, degree in degrees1.iteritems():
            sum_and_count(degreeCoefCorr1, degree, coef1[node])
        for node, degree in degrees2.iteritems():
            sum_and_count(degreeCoefCorr2, degree, coef2[node])
            
        return [correlation_list(get_avg_map(degreeCoefCorr1)), 
                correlation_list(get_avg_map(degreeCoefCorr2))]


class AvgClustCoefDegree(MetricsModule):

    def __init__(self, conf):
        MetricsModule.__init__(self, conf)
        self.degreeClustCoef = conf['clust-coef-degree']

    def reset(self, ts):
        pass

    def update_link(self, nodeA, nodeB, ts):
        pass

    def update_node(self, node, ntype, ts):
        pass

    def update_from_network(self, ts):
        pass

    def get_values(self, ts):
        [coef1, diff1] = self.degreeClustCoef.get_values(ts)
        return [avg_values(coef1),avg_values(diff1)]
    

class LocalClustCoef(MetricsModule):

    def __init__(self, conf):
        MetricsModule.__init__(self, conf)
        self.nodeClustCoef = dict()
        self.nodeId = int(conf['nodeid'])

    def reset(self, ts):
        self.nodeClustCoef[ts] = 0

    def update_link(self, nodeA, nodeB, ts):
#        self.degreeClustCoef[ts][nodeA] = self.net.getNewClusterCoef(nodeA)
        pass

    def update_node(self, node, ntype, ts):
        pass

    def update_from_network(self, ts):
        self.nodeClustCoef[ts] = self.net.getLocalClustCoef(self.nodeId)
        pass

    def get_values(self, ts):
        return [round(self.nodeClustCoef[ts], 3)]


class LocalClustCoefDegree(MetricsModule):

    def __init__(self, conf):
        MetricsModule.__init__(self, conf)
        self.nodeClustCoef = dict()
        self.nodeId = int(conf['nodeid'])

    def reset(self, ts):
        self.nodeClustCoef[ts] = 0

    def update_link(self, nodeA, nodeB, ts):
#        self.degreeClustCoef[ts][nodeA] = self.net.getNewClusterCoef(nodeA)
        pass

    def update_node(self, node, ntype, ts):
        pass

    def update_from_network(self, ts):
        self.nodeClustCoef[ts] = self.net.getDegreeClusterCoef(self.nodeId, True)
        pass

    def get_values(self, ts):
        return [round(self.nodeClustCoef[ts], 3)]


metrics_modules = {# store modules
                   'nodes-degrees' : NodesDegrees,
                   'nodes-strength' : NodesStrength,
                   'neighbour-hubs-singles' : NeighbourHubsSingles,
                   'degree-links-count' : DegreeLinksCount,
                   'avg-neighbour-degree' : AvgNeighbourDegree,
                   'clust-coef' : ClustCoef,
                   'clust-coef-opsahl' : ClustCoefOpsahl,
                   'clust-coef-degree' : ClustCoefDegree,
                   # metrics modules
                   'date' : Date,
                   'date-relative' : DateRelative,
                   'new-nodes' : NewNodes,
                   'nodes-count' : NodesCount,
                   'avg-node-degree' : AvgNodeDegree,
                   'avg-node-strength' : AvgNodesStrength,
                   'avg-node-lifetime' : AvgNodeLifetime,
                   'links-count' : LinksCount,
                   'avg-new-link-shortest-path' : AvgNewLinkShortestPath,
                   'degree-distribution' : DegreeDistribution,
                   'hubs-singles-distribution' : HubsSinglesDistribution,
                   'degree-links-count-correlation' : DegreeLinksCountCorrelation,
                   'avg-neighbour-degree-distribution' : AvgNeighbourDegreeDistribution,
                   'degree-avg-neighbour-degree-correlation' : DegreeAvgNeighbourDegreeCorrelation,
                   'avg-clust-coef' : AvgClustCoef,
                   'avg-clust-coef-opsahl' : AvgClustCoefOpsahl,
                   'clust-coef-distribution' : ClustCoefDistribution,
                   'degree-clust-coef-correlation' : DegreeClustCoefCorrelation,
                   'avg-clust-coef-degree' : AvgClustCoefDegree,
                   'local-clust-coef' : LocalClustCoef,
                   'local-clust-coef-degree' : LocalClustCoefDegree,
                  }


#class Metrics(MetricsModule):
#
#    def __init__(self, conf):
#        MetricsModule.__init__(self, conf)
#        self. = [dict(), dict()]
#
#    def reset(self, ts):
#        self.[0][ts] = 
#        self.[1][ts] = 
#
#    def update_link(self, nodeA, nodeB, ts):
#
#    def update_node(self, node, ntype, ts):
#
#    def update_from_network(self, ts):
#
#    def get_values(self, ts):
#        if ts in self.[0]:
#            return [self.[0][ts], self.[1][ts]]
#        else:
#            return []

