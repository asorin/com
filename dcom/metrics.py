'''
Created on 19 Nov 2012

@author: sorin
'''

from datetime import datetime
import bisect
import scipy
import scipy.stats
import numpy


class MetricsRelative:
    def __init__(self, net, aggrPeriod = 86400):
        self.net = net
        self.aggrPeriod = aggrPeriod
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

        _check_and_increment(node, self.nodeStrength[ntype][tsAggr])
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
                                _avg_values(degreesTsMap0[ts]), _avg_values(degreesTsMap1[ts]),
                                _avg_values(self.nodeStrength[0][ts]), _avg_values(self.nodeStrength[1][ts]),
                                ))
        return metrics_list
#
    def getNames(self):
        return ("Date", "Avg node degree 1", "Avg node degree 2", "Avg node strength 1", "Avg node strength 2" )  

    def __init_maps(self, ts):
        self.nodeStrength[0][ts] = dict()
        self.nodeStrength[1][ts] = dict()

    def __relativeTs(self, node, ts):
        tsRel = 0
        if self.net.hasNode(node):
            tsRel = ts - self.net.getNodeTs(node)
#   	    print ts, self.net.getNodeTs(node), tsRel/2419200
        return tsRel
        
    def __convertToTs(self, vmap):
        tsmapall = dict()
        for node, tsmap in vmap.iteritems():
             for ts, val in tsmap.iteritems():
                 if not ts in tsmapall:
                     tsmapall[ts] = dict()
                 tsmapall[ts][node] = val
        return tsmapall

class Metrics:
    def __init__(self, net, aggrPeriod = 86400):
        self.net = net
        self.aggrPeriod = aggrPeriod
        self.tsList = []
        # metrics on nodes
        self.newNodesCount = [dict(), dict()]
        self.nodeDegrees = [dict(), dict()]
        self.nodeStrength = [dict(), dict()]
        self.nodeAvgLifetime = [dict(), dict()]
        self.nodeHubs = [dict(), dict()]
        self.nodeHubsPct = [dict(), dict()]
        self.nodeSingles = [dict(), dict()]
        self.nodeSinglesPct = [dict(), dict()]
        self.nodeClustCoef = [dict(), dict()]
        self.nodeAvgDegreeNeigh = [dict(), dict()]
        self.nodeAvgDegreeCorr = [dict(), dict()]
        self.nodeHubsDegreeCorr = [dict(), dict()]
        self.nodeSinglesDegreeCorr = [dict(), dict()]
        self.nodeLinksDegreeCorr = [dict(), dict()]
        self.nodeNewLinksDegreeCorr = [dict(), dict()]
        self.nodeReinfLinksDegreeCorr = [dict(), dict()]
        # metrics on links
        self.linkCount = dict()
        self.nodesCount = [dict(), dict()]
        self.reinfLinkCount = dict()
        self.neighSameGroupLinkCount = [dict(), dict()]
        self.linkShortestPath = dict()
        self.linkHubsRatio = dict()
        self.linkSinglesRatio = dict()

    def newEventPre(self, nodeA, nodeB, ts):
        hasNodeA = self.net.hasNode(nodeA)
        hasNodeB = self.net.hasNode(nodeB)
        hasLink = self.net.hasEdge(nodeA, nodeB)

        tsAggr = int(ts/ self.aggrPeriod)
        if not tsAggr in self.tsList:
            self.__init_maps(tsAggr)
            # get stats fron network for previous timestamp
            self.__updateFromNetwork()
            self.tsList.append(tsAggr)
            
        # nodes metrics
        if not hasNodeA:
            _increment(tsAggr, self.newNodesCount[0])
        if not hasNodeB:
            _increment(tsAggr, self.newNodesCount[1])
            
        _check_and_increment(nodeA, self.nodeStrength[0][tsAggr])
        _check_and_increment(nodeB, self.nodeStrength[1][tsAggr])
        
        # links metrics
        _increment(tsAggr, self.linkCount)
        if hasNodeA and hasNodeB:
            if self.net.hasPartition():
                _increment(tsAggr, self.neighSameGroupLinkCount[0], self.__sameGrpNeighbors(nodeA, nodeB))
                _increment(tsAggr, self.neighSameGroupLinkCount[1], self.__sameGrpNeighbors(nodeB, nodeA))
#            if self.net.hasPath(nodeA, nodeB):
#                self._add_to_list(tsAggr, self.linkShortestPath, self.net.getShortestPath(nodeA, nodeB))
        if hasNodeA:
            _check_and_increment(self.net.getDegree(nodeA), self.nodeLinksDegreeCorr[0][tsAggr])
        if hasNodeB:
            _check_and_increment(self.net.getDegree(nodeB), self.nodeLinksDegreeCorr[1][tsAggr])
        
        if hasLink:
            _increment(tsAggr, self.reinfLinkCount)
            _check_and_increment(self.net.getDegree(nodeA), self.nodeReinfLinksDegreeCorr[0][tsAggr])
            _check_and_increment(self.net.getDegree(nodeB), self.nodeReinfLinksDegreeCorr[1][tsAggr])
        else:
            if hasNodeA and hasNodeB and self.net.hasPath(nodeA, nodeB):
                _add_to_list(tsAggr, self.linkShortestPath, self.net.getShortestPath(nodeA, nodeB))
            if hasNodeA:
                _check_and_increment(self.net.getDegree(nodeA), self.nodeNewLinksDegreeCorr[0][tsAggr])
            if hasNodeB:
                _check_and_increment(self.net.getDegree(nodeB), self.nodeNewLinksDegreeCorr[1][tsAggr])
                    
        
    def newEventPost(self, nodeA, nodeB, ts):
        pass
            
    def getMetrics(self):
        self.__updateFromNetwork()
        metrics_list = []
        for ts in self.tsList:
            metrics_list.append((datetime.utcfromtimestamp(ts*self.aggrPeriod), self.nodesCount[0][ts],
                                 self.nodesCount[1][ts], self.newNodesCount[0][ts], 
                                self.newNodesCount[1][ts], self.linkCount[ts], self.reinfLinkCount[ts],
                                _avg_values(self.nodeDegrees[0][ts]), _avg_values(self.nodeDegrees[1][ts]),
                                _avg_values(self.nodeStrength[0][ts]), _avg_values(self.nodeStrength[1][ts]),
                                self.nodeAvgLifetime[0][ts]/self.aggrPeriod, self.nodeAvgLifetime[1][ts]/self.aggrPeriod,
                                round(scipy.mean(self.linkShortestPath[ts]), 3),
#                                _avg_values(self.nodeClustCoef[0][ts]), _avg_values(self.nodeClustCoef[1][ts]),
#                                _correlation(self.nodeDegrees[0][ts], self.nodeClustCoef[0][ts]), 
#                                _correlation(self.nodeDegrees[1][ts], self.nodeClustCoef[1][ts]),
#                                _correlation(self.nodeHubs[0][ts], self.nodeSingles[0][ts]), 
#                                _correlation(self.nodeDegrees[0][ts], self.nodeHubs[0][ts]), 
#                                _correlation(self.nodeDegrees[0][ts], self.nodeSingles[0][ts]), 
#                                round(self.linkHubsRatio[ts], 3), round(self.linkSinglesRatio[ts], 3),
                                _distribution(self.nodeDegrees[0][ts]), _distribution(self.nodeDegrees[1][ts]),
#                                _distribution(self.nodeHubs[0][ts]), _distribution(self.nodeSingles[0][ts]),
#                                _correlation_list(self.nodeHubsDegreeCorr[0][ts]), _correlation_list(self.nodeHubsDegreeCorr[1][ts]), 
#                                _correlation_list(self.nodeSinglesDegreeCorr[0][ts]), _correlation_list(self.nodeSinglesDegreeCorr[1][ts]),
#                                _distribution(self.nodeHubsPct[0][ts], 0.1, 1), _distribution(self.nodeSinglesPct[0][ts], 0.1, 1),
#                                _distribution(self.nodeClustCoef[0][ts]), _distribution(self.nodeClustCoef[1][ts]),
#                                _distribution(self.nodeAvgDegreeNeigh[0][ts]), _distribution(self.nodeAvgDegreeNeigh[1][ts]),
#                                 _correlation_list(self.nodeAvgDegreeCorr[0][ts]), _correlation_list(self.nodeAvgDegreeCorr[1][ts]),
#                                _correlation_list(self.nodeLinksDegreeCorr[0][ts]), _correlation_list(self.nodeLinksDegreeCorr[1][ts]), 
#                                _correlation_list(self.nodeNewLinksDegreeCorr[0][ts]), _correlation_list(self.nodeNewLinksDegreeCorr[1][ts]), 
#                                _correlation_list(self.nodeReinfLinksDegreeCorr[0][ts]), _correlation_list(self.nodeReinfLinksDegreeCorr[1][ts]),
                                 ))
#                                    self.neighSameGroupLinkCount[0][ts], self.neighSameGroupLinkCount[1][ts]))
        return metrics_list

    def getNames(self):
        return ("Date", "Nodes count q1", "Nodes count 2", "New nodes 1", "New nodes 2", "Link events", "Reinforced links", 
                "Avg node degree 1", "Avg node degree 2", "Avg node strength 1", "Avg node strength 2",
                "Avg node lifetime 1", "Avg node lifetime 2", "Avg new link shortest path",
#                "Avg clust coef 1", "Avg clust coef 2", 
#                "Correlation degree/clustering 1", 
#                "Correlation degree/clustering 2", 
#                "Correlation hubs-singles 1",
#                "Correlation degree-hubs 1",
#                "Correlation degree-singles 1",
#                "Hub links ratio", "Single links ratio",
                "Degree dist 1", "Degree dist 2",
#                "Node hubs dist 1", "Node singles dist 1"
#                "Degree-hubs corr 1", "Degree-hubs corr 2",
#                "Degree-singles corr 1", "Degree-singles corr 2",
#                "Node hubs pct dist 1", #"Node singles pct dist 1",
#                 "Degree-avg degree corr 1", "Degree-avg degree corr 2",
#                 "Degree-links events corr 1", "Degree-links events corr 2",
#                 "Degree-new links events corr 1", "Degree-new links events corr 2",
#                 "Degree-reinf links events corr 1", "Degree-reinf links events corr 2",

                )  
#                "Neighbours in same group 1", "Neighbours in same group 2")

    def __init_maps(self, ts):
        self.newNodesCount[0][ts] = 0
        self.newNodesCount[1][ts] = 0
        self.nodeStrength[0][ts] = dict()
        self.nodeStrength[1][ts] = dict()
        self.nodeAvgLifetime[0][ts] = 0
        self.nodeAvgLifetime[1][ts] = 0
        self.nodeClustCoef[0][ts] = dict()
        self.nodeClustCoef[0][ts] = dict()
        self.linkCount[ts] = 0
        self.reinfLinkCount[ts] = 0
        self.neighSameGroupLinkCount[0][ts] = 0
        self.neighSameGroupLinkCount[1][ts] = 0
        self.nodeLinksDegreeCorr[0][ts] = dict()
        self.nodeLinksDegreeCorr[1][ts] = dict()
        self.nodeNewLinksDegreeCorr[0][ts] = dict()
        self.nodeNewLinksDegreeCorr[1][ts] = dict()
        self.nodeReinfLinksDegreeCorr[0][ts] = dict()
        self.nodeReinfLinksDegreeCorr[1][ts] = dict()
        
    def __sameGrpNeighbors(self, nodeSrc, nodeDst):
        sameGrp = 0
        partition = self.net.getPartition()
        
        if (not nodeSrc in partition) or (not nodeDst in partition): 
            return 0
        
        group = partition[nodeSrc]
        neighbors_list = self.net.getNeighbors(nodeDst)
        for neighbor in neighbors_list:
            if neighbor in partition and group == partition[neighbor]:
                sameGrp+=1
                
        if sameGrp >= int(len(neighbors_list)):
            return 1
        else:
            return 0
    
    def __updateFromNetwork(self):
        if len(self.tsList) > 0:
            prevTs = self.tsList[len(self.tsList)-1]
            # update degrees
            d1, d2 = self.net.getDegrees()
            self.nodeDegrees[0][prevTs] = d1
            self.nodeDegrees[1][prevTs] = d2
            # update lifetime
            lf1, lf2 = self.net.getAvgLifetime()
            self.nodeAvgLifetime[0][prevTs] = lf1
            self.nodeAvgLifetime[1][prevTs] = lf2
            # Update nodes count
            self.nodesCount[0][prevTs] = self.net.getNodesCount(0)
            self.nodesCount[1][prevTs] = self.net.getNodesCount(1)

            # Get counts of hubs and single neighbors for each node
#            hubs, hubsPct, singles, singlesPct = self.net.getCountsHubsSingles(0, self.nodesCount[0][prevTs] * 0.10)
#            self.nodeHubs[0][prevTs] = hubs
#            self.nodeHubsPct[0][prevTs] = hubsPct
#            self.nodeSingles[0][prevTs] = singles
#            self.nodeSinglesPct[0][prevTs] = singlesPct
        
            # Get correlation of hubs and singles count with node degree
            hubsCorr, singlesCorr = self.net.getDegreeCorrHubsSingles(0, self.nodesCount[0][prevTs] * 0.10)
            self.nodeHubsDegreeCorr[0][prevTs] = hubsCorr
            self.nodeSinglesDegreeCorr[0][prevTs] = singlesCorr
            hubsCorr, singlesCorr = self.net.getDegreeCorrHubsSingles(1, self.nodesCount[1][prevTs] * 0.10)
            self.nodeHubsDegreeCorr[1][prevTs] = hubsCorr
            self.nodeSinglesDegreeCorr[1][prevTs] = singlesCorr
                
            # Get hub edges ratio
#            hubsRatio, singleRatio = self.net.getEdgesRatio(self.nodesCount[0][prevTs] * 0.05)
#            self.linkHubsRatio[prevTs] = hubsRatio
#            self.linkSinglesRatio[prevTs] = singleRatio

            # update average degree of neighbours
            avgDegree0, avgDegreeCorr0 = self.net.getAvgNeighborDegree(0)
            self.nodeAvgDegreeNeigh[0][prevTs] = avgDegree0
            self.nodeAvgDegreeCorr[0][prevTs] = avgDegreeCorr0
            avgDegree1, avgDegreeCorr1 = self.net.getAvgNeighborDegree(1)
            self.nodeAvgDegreeNeigh[1][prevTs] = avgDegree1
            self.nodeAvgDegreeCorr[1][prevTs] = avgDegreeCorr1

            # update clustering coef
#            self.nodeClustCoef[0][prevTs] = self.net.getClustCoef(0)
#            self.nodeClustCoef[1][prevTs] = self.net.getClustCoef(1)
#            print datetime.utcfromtimestamp(prevTs*self.aggrPeriod)
            


# Some helping functions
def _increment(key, vmap, val=1):
    vmap[key] += val

def _check_and_increment(key, vmap, val=1):
    if not key in vmap:
        vmap[key] = 0
    vmap[key] += val

def _update_lifetime(key, vmap, vmapAll, curTs):
    if not key in vmapAll:
        vmapAll[key] = (curTs, 0)
    else:
        lf = vmapAll[key]
        vmapAll[key] = (lf[0], curTs-lf[0])
#        vmap[key] = vmapAll[key]

def _add_to_list(key, lmap, val):
    if not key in lmap:
        lmap[key] = []
    lmap[key].append(val)
    
def _avg_values(vmap):
    return 0 if len(vmap)==0 else round(sum(vmap.values()) / len(vmap), 3)

def _avg_lifetime(vmap):
    vals = list(l[1] for l in vmap.values())
#        print len(vals)
    if len(vals)>0:
        return round(sum(vals)/len(vals), 3)
    else:
        return 0

def _distribution(vmap, binsz=1, maxv=0):
    d = vmap.values()
    max_d = max(d)
    if max_d == 0:
        return "0"
    if maxv!=0:
        max_d = maxv
    hist = numpy.histogram(d, [x * binsz for x in range(0, 1+int(max_d/binsz))], (0, max_d), False, None, True)[0]
#    rndhist = map(round, hist, [3]*len(hist))
    return ",".join(map(_str_nozero, hist)) 
#    numpy.histogram(d, [x * 0.1 for x in range(0, 10)], (0, 1), False)[0])
     
def _correlation(vmap1, vmap2):
    x = []
    y = []
    for key in vmap1:
        x.append(vmap1[key])
        y.append(vmap2[key])
    return round(scipy.stats.pearsonr(x, y)[0], 3)

def _correlation_list(vmap):
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
        
