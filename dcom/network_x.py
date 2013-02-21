import networkx as nx
from metrics import Metrics, MetricsRelative
from networkx.algorithms import bipartite
import louvain
import scipy

from tools import sum_and_count
from tools import get_avg_map

class NetworkX():

    def __init__(self, options):
        self.G = nx.Graph()
        if options['action']=="metrics":
            self.metrics = MetricsRelative(self, options) if options['relative'] else Metrics(self, options)
        else:
            self.metrics = None
        self.partition = None

    def addLink(self, nodeA, nodeB, ts=None):
        
        if not self.metrics is None:
            self.metrics.newEventPre(nodeA, nodeB, ts)
        
        hadNodeA = nodeA in self.G
        hadNodeB = nodeB in self.G
        
        self.G.add_edge(nodeA, nodeB, timestamp=ts)
        
        self.__updateNode(nodeA, 0, ts, hadNodeA)
        self.__updateNode(nodeB, 1, ts, hadNodeB)

        if not self.metrics is None:
            self.metrics.newEventPost(nodeA, nodeB, ts)

    def flush(self):
        pass
    
    def setPartition(self, partition):
        self.partition = partition
                
    def findPartition(self, ntype, threshold):
        prjG = self.__get_projection(ntype, threshold)
        
        self.partition = louvain.best_partition(prjG)
        
        return self.partition

    def hasPartition(self):
        return not self.partition is None
    
    def __get_projection(self, ntype, threshold):
        nodes = set(n for n,d in self.G.nodes(data=True) if d["type"]==ntype)
        prjG = bipartite.weighted_projected_graph(self.G, nodes)
        print "Number of projected edges without threshold: " + str(len(prjG.edges()))
        listEdges = [ (u,v) for u,v,edata in prjG.edges(data=True) if edata['weight'] >= threshold ]
        print "Number of projected edges with threshold: " + str(len(listEdges))
        
        return nx.Graph(listEdges);

    def getPartition(self):
        return self.partition
    
    def hasNode(self, node):
        return node in self.G
    
    def hasEdge(self, nodeA, nodeB):
        return self.G.has_edge(nodeA, nodeB)
        
    def getNodeTs(self, node):
        return self.G.node[node]["start"]
    
    def getNeighbors(self, node):
        return self.G.neighbors(node)
    
    def getDegree(self, node):
        return self.G.degree(node)
    
    def getDegrees(self):
        nodes = set(n for n,d in self.G.nodes(data=True) if d["type"]==1)
        return bipartite.degrees(self.G, nodes)
    
    def getNodesCount(self, ntype):
        nodes = set(n for n,d in self.G.nodes(data=True) if d["type"]==ntype)
        return len(nodes)
        
    def getAvgLifetime(self):
        nodeLf0 = list(d["lifetime"] for _,d in self.G.nodes(data=True) if d["type"]==0)
        nodeLf1 = list(d["lifetime"] for _,d in self.G.nodes(data=True) if d["type"]==1)
        return sum(nodeLf0)/len(nodeLf0), sum(nodeLf1)/len(nodeLf1)
    
    def hasPath(self, nodeA, nodeB):
        return nx.has_path(self.G, nodeA, nodeB)
    
    def getShortestPath(self, nodeA, nodeB):
        return nx.shortest_path_length(self.G, nodeA, nodeB)
    
    def getClustCoef(self, ntype):
        nodes = set(n for n,d in self.G.nodes(data=True) if d["type"]==ntype)
        return bipartite.clustering(self.G, nodes, mode='dot')
    
    def __updateNode(self, node, ntype, ts, existed):
        if not existed:
            self.G.node[node]["type"] = ntype
            self.G.node[node]["start"] = ts
            self.G.node[node]["lifetime"] = 0
        else:
            self.G.node[node]["lifetime"] = ts - self.G.node[node]["start"]
    
    """
    Returns two maps: 
        map of nodes with average degree of their neighbors
        map of correlation between degree of nodes and average degree of their neighbors
    """
    def getAvgNeighbourDegree(self, ntype):
        nodes = set(n for n,d in self.G.nodes(data=True) if d["type"]==ntype)
        avgDegreeMap = dict()
        avgDegreeCorrMap = dict()
        for node in nodes:
            d = self.G.degree(node)
            degrees = list(self.G.degree(n) for n in self.G.neighbors(node))
            avgDegreeMap[node] = scipy.mean(degrees)
            sum_and_count(avgDegreeCorrMap, d, sum(degrees), len(degrees))
                
        return avgDegreeMap, get_avg_map(avgDegreeCorrMap)
    
    """
    Counts the number of neighbour hubs and singles for each node
    Returns map of nodes-hubs, nodes-hubs ratio, nodes-singles, nodes-singles ratio
    """
    def getCountsHubsSingles(self, ntype, hubThreshold):
        nodes = set(n for n,d in self.G.nodes(data=True) if d["type"]==ntype)
        hubsMap = dict()
        hubsPctMap = dict()
        singlesMap = dict()
        singlesPctMap = dict()
        for node in nodes:
            hubs = 0
            singles = 0
            neighbors = float(len(self.G.neighbors(node)))
            for neigh in self.G.neighbors(node):
                degree = self.G.degree(neigh)
                if degree >= hubThreshold:
                    hubs += 1
                elif degree == 1:
                    singles += 1
            hubsMap[node] = hubs
            hubsPctMap[node] = hubs/neighbors
            singlesMap[node] = singles
            singlesPctMap[node] = singles/neighbors
            
        return hubsMap, hubsPctMap, singlesMap, singlesPctMap

    """ 
    Get correlation of hubs and singles count with node degree
    """
    def getDegreeCorrHubsSingles(self, ntype, hubThreshold):
        nodes = set(n for n,d in self.G.nodes(data=True) if d["type"]==ntype)
        hubsDegreeCorrMap = dict()
        singlesDegreeCorrMap = dict()
        for node in nodes:
            hubs = 0
            singles = 0
            neighborsNode = self.G.neighbors(node)
            degreeNode = len(neighborsNode)
            for neighbor in neighborsNode:
                degreeNeighbor = self.G.degree(neighbor)
                if degreeNeighbor >= hubThreshold:
                    hubs += 1
                elif degreeNeighbor == 1:
                    singles += 1
            sum_and_count(hubsDegreeCorrMap, degreeNode, hubs, degreeNode)
            sum_and_count(singlesDegreeCorrMap, degreeNode, singles, degreeNode)
        
        return get_avg_map(hubsDegreeCorrMap), get_avg_map(singlesDegreeCorrMap) 
                

    """
    Get hub and single links ratio
    """
    def getEdgesRatio(self, hubThreshold):
        hublinks = 0.0
        singlelinks = 0.0
        for e in self.G.edges_iter():
            d2 = self.G.degree(e[1])
            if d2 >= hubThreshold:
                hublinks += 1
            elif d2 == 1:
                singlelinks +=1
        return hublinks/self.G.size(), singlelinks/self.G.size()

    def getNeighbourhood(self, node):
        nodes = [node]
        nodes.extend(self.G.neighbors(node))
        for n in self.G.neighbors(node):
            nodes.extend(self.G.neighbors(n))
        return self.G.subgraph(nodes)
    
    def save(self, fname, node=None):
        if node is None:
            nx.write_gml(self.G, fname)
        else:
            nx.write_gml(self.getNeighbourhood(int(node)), fname)
        
