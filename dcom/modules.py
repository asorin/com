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
        self.clustCoef[1][ts] = self.net.getClustCoef(1)

    def get_values(self, ts):
        if not ts in self.clustCoef[0]:
            return []

        return [self.clustCoef[0][ts], self.clustCoef[1][ts]]


"""
Clustering coefficient maps - Opsahl
"""

class OpsahlClustCoef(MetricsModule):

    def __init__(self, conf, ntype):
        MetricsModule.__init__(self, conf)
        self.globalCoefMap = {}
        self.globalCoefDgMap = {}
        self.localCoefMap = {}
        self.localCoefDgMap = {}
        self.localDegreeCoefMap = {}
        self.localDegreeCoefDgMap = {}
        self.ntype = ntype

    def reset(self, ts):
        pass

    def update_link(self, nodeA, nodeB, ts):
        pass

    def update_node(self, node, ntype, ts):
        pass

    def update_from_network(self, ts):
        globalCoef, globalDgCoef, coefMap, coefDgMap, degreeCoefMap, degreeCoefDgMap = self.net.getClustCoefOpsahlAll(self.ntype)
#        globalCoef, localCoef = self.net.getClustCoefOpsahlOriginal(0)
        self.globalCoefMap[ts] = globalCoef
        self.globalCoefDgMap[ts] = globalDgCoef
        self.localCoefMap[ts] = coefMap
        self.localCoefDgMap[ts] = coefDgMap
        self.localDegreeCoefMap[ts] = degreeCoefMap
        self.localDegreeCoefDgMap[ts] = degreeCoefDgMap

    def get_values(self, ts):
        if not ts in self.globalCoefMap:
            return []

        return [self.globalCoefMap[ts], self.globalCoefDgMap[ts],
                self.localCoefMap[ts], self.localCoefDgMap[ts],
                self.localDegreeCoefMap[ts], self.localDegreeCoefDgMap[ts]]

class OpsahlClustCoefUsers(OpsahlClustCoef):
    def __init__(self, conf):
        OpsahlClustCoef.__init__(self, conf, 0)

class OpsahlClustCoefObjects(OpsahlClustCoef):
    def __init__(self, conf):
        OpsahlClustCoef.__init__(self, conf, 1)


"""
Clustering coefficient maps - Latapy
"""

class LatapyClustCoef(MetricsModule):

    def __init__(self, conf, ntype):
        MetricsModule.__init__(self, conf)
        self.coefMap, self.coefDgMap = {}, {}
        self.degreeCoefMap, self.degreeCoefDgMap = {}, {}
        self.ntype = ntype

    def reset(self, ts):
        self.coefMap[ts] = {}
        self.coefDgMap[ts] = {}
        self.degreeCoefMap[ts] = {}
        self.degreeCoefDgMap[ts] = {}

    def update_link(self, nodeA, nodeB, ts):
        pass

    def update_node(self, node, ntype, ts):
        pass

    def update_from_network(self, ts):
        coefs, coefsDg, degreeCoefs, degreeCoefsDg = self.net.getClusterCoefLatapy(self.ntype)
        self.coefMap[ts] = coefs
        self.coefDgMap[ts] = coefsDg
        self.degreeCoefMap[ts] = degreeCoefs
        self.degreeCoefDgMap[ts] = degreeCoefsDg

    def get_values(self, ts):
        return [self.coefMap[ts], self.coefDgMap[ts], self.degreeCoefMap[ts], self.degreeCoefDgMap[ts]]


class LatapyClustCoefUsers(LatapyClustCoef):
    def __init__(self, conf):
        LatapyClustCoef.__init__(self, conf, 0)

class LatapyClustCoefObjects(LatapyClustCoef):
    def __init__(self, conf):
        LatapyClustCoef.__init__(self, conf, 1)


"""
Collaborative similarity - Shang et al 2011
Users and objects
"""

class CollabSimilarity(MetricsModule):

    def __init__(self, conf, ntype):
        MetricsModule.__init__(self, conf)
        self.simMap, self.degreeSimMap = {}, {}
        self.ntype = ntype

    def reset(self, ts):
        self.simMap[ts] = {}
        self.degreeSimMap[ts] = {}

    def update_link(self, nodeA, nodeB, ts):
        pass

    def update_node(self, node, ntype, ts):
        pass

    def update_from_network(self, ts):
        sims, degreeSims = self.net.getCollabSimilarityAll(self.ntype)
        self.simMap[ts] = sims
        self.degreeSimMap[ts] = degreeSims

    def get_values(self, ts):
        return [self.simMap[ts], self.degreeSimMap[ts]]


class CollabSimilarityUsers(CollabSimilarity):
    def __init__(self, conf):
        CollabSimilarity.__init__(self, conf, 0)

class CollabSimilarityObjects(CollabSimilarity):
    def __init__(self, conf):
        CollabSimilarity.__init__(self, conf, 1)


"""
Projected network modularity - Louvain method
Users and objects
"""

class PrjModularity(MetricsModule):

    def __init__(self, conf, ntype):
        MetricsModule.__init__(self, conf)
        self.prjMod = {}
        self.ntype = ntype

    def reset(self, ts):
        pass

    def update_link(self, nodeA, nodeB, ts):
        pass

    def update_node(self, node, ntype, ts):
        pass

    def update_from_network(self, ts):
        self.prjMod[ts] = self.net.getPrjModularity(self.ntype)

    def get_values(self, ts):
        return [self.prjMod[ts]]


class PrjModularityUsers(PrjModularity):
    def __init__(self, conf):
        PrjModularity.__init__(self, conf, 0)

class PrjModularityObjects(PrjModularity):
    def __init__(self, conf):
        PrjModularity.__init__(self, conf, 1)


"""
Links count for projected network
Users and objects
"""

class PrjLinksCount(MetricsModule):

    def __init__(self, conf, ntype):
        MetricsModule.__init__(self, conf)
        self.prjLinksCount = {}
        self.ntype = ntype

    def reset(self, ts):
        pass

    def update_link(self, nodeA, nodeB, ts):
        pass

    def update_node(self, node, ntype, ts):
        pass

    def update_from_network(self, ts):
        self.prjLinksCount[ts] = self.net.getPrjLinksCount(self.ntype)

    def get_values(self, ts):
        return [self.prjLinksCount[ts]]


class PrjLinksCountUsers(PrjLinksCount):
    def __init__(self, conf):
        PrjLinksCount.__init__(self, conf, 0)

class PrjLinksCountObjects(PrjLinksCount):
    def __init__(self, conf):
        PrjLinksCount.__init__(self, conf, 1)


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

class LinkWeightsDistribution(MetricsModule):

    def __init__(self, conf):
        MetricsModule.__init__(self, conf)
        self.weightsDist = dict()

    def reset(self, ts):
        pass

    def update_link(self, nodeA, nodeB, ts):
        pass
    
    def update_node(self, node, ntype, ts):
        pass

    def update_from_network(self, ts):
        self.weightsDist[ts] = self.net.getWeightsDist()
        pass

    def get_values(self, ts):
        return [return_from_map(self.weightsDist, ts)]


class LinkDegreeWeightCorrelation(MetricsModule):

    def __init__(self, conf):
        MetricsModule.__init__(self, conf)
        self.weightsCorr = [dict(), dict()]

    def reset(self, ts):
        pass

    def update_link(self, nodeA, nodeB, ts):
        pass
    
    def update_node(self, node, ntype, ts):
        pass

    def update_from_network(self, ts):
        dwMap1, dwMap2 = self.net.getDegreeWeightMap()
        self.weightsCorr[0][ts] = correlation_list(dwMap1)
        self.weightsCorr[1][ts] = correlation_list(dwMap2)

    def get_values(self, ts):
        return [return_from_map(self.weightsCorr[0], ts), return_from_map(self.weightsCorr[1], ts)]


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

"""
Average and distribution of clustering coefficiet
NetworkX implementation (Latapy)
"""

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


"""
Average clustering coefficient - Opsahl
Users and objects
"""

class OpsahlAvgClustCoef(MetricsModule):

    def __init__(self, conf, ntype):
        MetricsModule.__init__(self, conf)
        self.clustCoef = conf['opsahl-clust-coef-'+ntype]

    def reset(self, ts):
        pass

    def update_link(self, nodeA, nodeB, ts):
        pass

    def update_node(self, node, ntype, ts):
        pass

    def update_from_network(self, ts):
        pass

    def get_values(self, ts):
        globalc, globalcDg, localCoefMap, localCoefDgMap, _, _ = self.clustCoef.get_values(ts)
#        for node, coef in sorted(localCoefMap.iteritems()):
#            print node, coef, localCoefDgMap[node]
        return [globalc, globalcDg]

class OpsahlAvgClustCoefUsers(OpsahlAvgClustCoef):

    def __init__(self, conf):
        OpsahlAvgClustCoef.__init__(self, conf, "users")

class OpsahlAvgClustCoefObjects(OpsahlAvgClustCoef):
    def __init__(self, conf):
        OpsahlAvgClustCoef.__init__(self, conf, "objects")


"""
Clustering coefficient distribution - Opsahl
Users and objects
"""

class OpsahlClustCoefDist(MetricsModule):

    def __init__(self, conf, ntype):
        MetricsModule.__init__(self, conf)
        self.clustCoef = conf['opsahl-clust-coef-'+ntype]

    def reset(self, ts):
        pass

    def update_link(self, nodeA, nodeB, ts):
        pass

    def update_node(self, node, ntype, ts):
        pass

    def update_from_network(self, ts):
        pass

    def get_values(self, ts):
        _, _, nodeCoefs, nodeCoefsDg, _, _ = self.clustCoef.get_values(ts)
        return [distribution(nodeCoefs, 0.1, 1), distribution(nodeCoefsDg, 0.1, 1)]

class OpsahlClustCoefDistUsers(OpsahlClustCoefDist):

    def __init__(self, conf):
        OpsahlClustCoefDist.__init__(self, conf, "users")

class OpsahlClustCoefDistObjects(OpsahlClustCoefDist):
    def __init__(self, conf):
        OpsahlClustCoefDist.__init__(self, conf, "objects")


"""
Clustering coefficient correlation with degree - Opsahl 
Users and objects
"""

class OpsahlClustCoefDegreeCorr(MetricsModule):

    def __init__(self, conf, ntype):
        MetricsModule.__init__(self, conf)
        self.clustCoefOpsahl = conf['opsahl-clust-coef-'+ntype]

    def reset(self, ts):
        pass

    def update_link(self, nodeA, nodeB, ts):
        pass

    def update_node(self, node, ntype, ts):
        pass

    def update_from_network(self, ts):
        pass

    def get_values(self, ts):
        [_, _, _, _, degreeCoefs, degreeCoefsDg] = self.clustCoefOpsahl.get_values(ts)

        return [correlation_list(degreeCoefs), correlation_list(degreeCoefsDg)]


class OpsahlClustCoefDegreeCorrUsers(OpsahlClustCoefDegreeCorr):
    def __init__(self, conf):
        OpsahlClustCoefDegreeCorr.__init__(self, conf, "users")

class OpsahlClustCoefDegreeCorrObjects(OpsahlClustCoefDegreeCorr):
    def __init__(self, conf):
        OpsahlClustCoefDegreeCorr.__init__(self, conf, "objects")


"""
Average clustering coefficient - Latapy
Users and objects
"""

class LatapyAvgClustCoef(MetricsModule):

    def __init__(self, conf, ntype):
        MetricsModule.__init__(self, conf)
        self.clustCoefLatapy = conf['latapy-clust-coef-'+ntype]

    def reset(self, ts):
        pass

    def update_link(self, nodeA, nodeB, ts):
        pass

    def update_node(self, node, ntype, ts):
        pass

    def update_from_network(self, ts):
        pass

    def get_values(self, ts):
        [coefs, coefsDg, _, _] = self.clustCoefLatapy.get_values(ts)
        return [avg_values(coefs),avg_values(coefsDg)]

class LatapyAvgClustCoefUsers(LatapyAvgClustCoef):
    def __init__(self, conf):
        LatapyAvgClustCoef.__init__(self, conf, "users")

class LatapyAvgClustCoefObjects(LatapyAvgClustCoef):
    def __init__(self, conf):
        LatapyAvgClustCoef.__init__(self, conf, "objects")


"""
Clustering coefficient distribution - Latapy
Users and objects
"""

class LatapyClustCoefDist(MetricsModule):

    def __init__(self, conf, ntype):
        MetricsModule.__init__(self, conf)
        self.clustCoef = conf['latapy-clust-coef-'+ntype]

    def reset(self, ts):
        pass

    def update_link(self, nodeA, nodeB, ts):
        pass

    def update_node(self, node, ntype, ts):
        pass

    def update_from_network(self, ts):
        pass

    def get_values(self, ts):
        [coefs, coefsDg, _, _] = self.clustCoef.get_values(ts)
        return [distribution(coefs, 0.1, 1), distribution(coefsDg, 0.1, 1)]

class LatapyClustCoefDistUsers(LatapyClustCoefDist):

    def __init__(self, conf):
        LatapyClustCoefDist.__init__(self, conf, "users")

class LatapyClustCoefDistObjects(LatapyClustCoefDist):
    def __init__(self, conf):
        LatapyClustCoefDist.__init__(self, conf, "objects")


"""
Clustering coefficient correlation with degree - Latapy 
Users and objects
"""

class LatapyClustCoefDegreeCorr(MetricsModule):

    def __init__(self, conf, ntype):
        MetricsModule.__init__(self, conf)
        self.clustCoefLatapy = conf['latapy-clust-coef-'+ntype]

    def reset(self, ts):
        pass

    def update_link(self, nodeA, nodeB, ts):
        pass

    def update_node(self, node, ntype, ts):
        pass

    def update_from_network(self, ts):
        pass

    def get_values(self, ts):
        [_, _, degreeCoefs, degreeCoefsDg] = self.clustCoefLatapy.get_values(ts)

        return [correlation_list(degreeCoefs), correlation_list(degreeCoefsDg)]


class LatapyClustCoefDegreeCorrUsers(LatapyClustCoefDegreeCorr):
    def __init__(self, conf):
        LatapyClustCoefDegreeCorr.__init__(self, conf, "users")

class LatapyClustCoefDegreeCorrObjects(LatapyClustCoefDegreeCorr):
    def __init__(self, conf):
        LatapyClustCoefDegreeCorr.__init__(self, conf, "objects")


"""
Local clustering coefficient - Latapy
"""

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


class LatapyLocalClustCoef(MetricsModule):

    def __init__(self, conf):
        MetricsModule.__init__(self, conf)
        self.nodeCoef = {}
        self.nodeCoefDg = {}
        self.nodeId = int(conf['nodeid'])

    def reset(self, ts):
        self.nodeCoef[ts] = 0
        self.nodeCoefDg[ts] = 0

    def update_link(self, nodeA, nodeB, ts):
        pass

    def update_node(self, node, ntype, ts):
        pass

    def update_from_network(self, ts):
        coef, coefDg = self.net.getNodeClusterCoefLatapy(self.nodeId, True)
        self.nodeCoef[ts] = coef
        self.nodeCoefDg[ts] = coefDg

    def get_values(self, ts):
        return [round(self.nodeCoef[ts], 3), round(self.nodeCoefDg[ts], 3)]


"""
Average collaborative similarity - Shang et al 2011
Users and objects
"""

class AvgCollabSimilarity(MetricsModule):

    def __init__(self, conf, ntype):
        MetricsModule.__init__(self, conf)
        self.collSim = conf['collab-similarity-'+ntype]

    def reset(self, ts):
        pass

    def update_link(self, nodeA, nodeB, ts):
        pass

    def update_node(self, node, ntype, ts):
        pass

    def update_from_network(self, ts):
        pass

    def get_values(self, ts):
        [simMap, _] = self.collSim.get_values(ts)
        return [avg_values(simMap)]

class AvgCollabSimilarityUsers(AvgCollabSimilarity):
    def __init__(self, conf):
        AvgCollabSimilarity.__init__(self, conf, "users")

class AvgCollabSimilarityObjects(AvgCollabSimilarity):
    def __init__(self, conf):
        AvgCollabSimilarity.__init__(self, conf, "objects")


"""
Collaborative similarity distribution
Users and objects
"""

class CollabSimilarityDist(MetricsModule):

    def __init__(self, conf, ntype):
        MetricsModule.__init__(self, conf)
        self.collSim = conf['collab-similarity-'+ntype]

    def reset(self, ts):
        pass

    def update_link(self, nodeA, nodeB, ts):
        pass

    def update_node(self, node, ntype, ts):
        pass

    def update_from_network(self, ts):
        pass

    def get_values(self, ts):
        [simMap, _] = self.collSim.get_values(ts)
        return [distribution(simMap, 0.1, 1)]

class CollabSimilarityDistUsers(CollabSimilarityDist):

    def __init__(self, conf):
        CollabSimilarityDist.__init__(self, conf, "users")

class CollabSimilarityDistObjects(CollabSimilarityDist):
    def __init__(self, conf):
        CollabSimilarityDist.__init__(self, conf, "objects")


"""
Collaborative similarity correlation with degree 
Users and objects
"""

class CollabSimilarityDegreeCorr(MetricsModule):

    def __init__(self, conf, ntype):
        MetricsModule.__init__(self, conf)
        self.collSim = conf['collab-similarity-'+ntype]

    def reset(self, ts):
        pass

    def update_link(self, nodeA, nodeB, ts):
        pass

    def update_node(self, node, ntype, ts):
        pass

    def update_from_network(self, ts):
        pass

    def get_values(self, ts):
        [_, degreeSimMap] = self.collSim.get_values(ts)

        return [correlation_list(degreeSimMap)]


class CollabSimilarityDegreeCorrUsers(CollabSimilarityDegreeCorr):
    def __init__(self, conf):
        CollabSimilarityDegreeCorr.__init__(self, conf, "users")

class CollabSimilarityDegreeCorrObjects(CollabSimilarityDegreeCorr):
    def __init__(self, conf):
        CollabSimilarityDegreeCorr.__init__(self, conf, "objects")



metrics_modules = {# store modules
                   'nodes-degrees' : NodesDegrees,
                   'nodes-strength' : NodesStrength,
                   'neighbour-hubs-singles' : NeighbourHubsSingles,
                   'degree-links-count' : DegreeLinksCount,
                   'avg-neighbour-degree' : AvgNeighbourDegree,
                   'clust-coef' : ClustCoef,
                   'latapy-clust-coef-users' : LatapyClustCoefUsers,
                   'latapy-clust-coef-objects' : LatapyClustCoefObjects,
                   'opsahl-clust-coef-users' : OpsahlClustCoefUsers,
                   'opsahl-clust-coef-objects' : OpsahlClustCoefObjects,
                   'collab-similarity-users' : CollabSimilarityUsers,
                   'collab-similarity-objects' : CollabSimilarityObjects,
                   # metrics modules
                   'date' : Date,
                   'date-relative' : DateRelative,
                   'new-nodes' : NewNodes,
                   'nodes-count' : NodesCount,
                   'avg-node-degree' : AvgNodeDegree,
                   'avg-node-strength' : AvgNodesStrength,
                   'avg-node-lifetime' : AvgNodeLifetime,
                   'links-count' : LinksCount,
                   'link-weights-distribution' : LinkWeightsDistribution,
                   'link-degree-weight-correlation' : LinkDegreeWeightCorrelation,
                   'avg-new-link-shortest-path' : AvgNewLinkShortestPath,
                   'degree-distribution' : DegreeDistribution,
                   'hubs-singles-distribution' : HubsSinglesDistribution,
                   'degree-links-count-correlation' : DegreeLinksCountCorrelation,
                   'avg-neighbour-degree-distribution' : AvgNeighbourDegreeDistribution,
                   'degree-avg-neighbour-degree-correlation' : DegreeAvgNeighbourDegreeCorrelation,
                   
                   'avg-clust-coef' : AvgClustCoef,
                   'clust-coef-distribution' : ClustCoefDistribution,
                   'local-clust-coef' : LocalClustCoef,
                   
                   'opsahl-avg-clust-coef-users' : OpsahlAvgClustCoefUsers,
                   'opsahl-avg-clust-coef-objects' : OpsahlAvgClustCoefObjects,
                   'opsahl-clust-coef-dist-users' : OpsahlClustCoefDistUsers,
                   'opsahl-clust-coef-dist-objects' : OpsahlClustCoefDistObjects,
                   'opsahl-clust-coef-degree-corr-users' : OpsahlClustCoefDegreeCorrUsers,
                   'opsahl-clust-coef-degree-corr-objects' : OpsahlClustCoefDegreeCorrObjects,
                   
                   'latapy-avg-clust-coef-users' : LatapyAvgClustCoefUsers,
                   'latapy-avg-clust-coef-objects' : LatapyAvgClustCoefObjects,
                   'latapy-clust-coef-dist-users' : LatapyClustCoefDistUsers,
                   'latapy-clust-coef-dist-objects' : LatapyClustCoefDistObjects,
                   'latapy-clust-coef-degree-corr-users' : LatapyClustCoefDegreeCorrUsers,
                   'latapy-clust-coef-degree-corr-objects' : LatapyClustCoefDegreeCorrObjects,
                   'latapy-local-clust-coef' : LatapyLocalClustCoef,
                   
                   'avg-collab-similarity-users' : AvgCollabSimilarityUsers,
                   'avg-collab-similarity-objects' : AvgCollabSimilarityObjects,
                   'collab-similarity-dist-users' : CollabSimilarityDistUsers,
                   'collab-similarity-dist-objects' : CollabSimilarityDistObjects,
                   'collab-similarity-degree-corr-users' : CollabSimilarityDegreeCorrUsers,
                   'collab-similarity-degree-corr-objects' : CollabSimilarityDegreeCorrObjects,
                   
                   'prj-modularity-users' : PrjModularityUsers,
                   'prj-modularity-objects' : PrjModularityObjects,
                   'prj-links-count-users' : PrjLinksCountUsers,
                   'prj-links-count-objects' : PrjLinksCountObjects,
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

