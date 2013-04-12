import networkx as nx
from metrics import MetricsAbsolute, MetricsRelative
from networkx.algorithms import bipartite
import louvain
import scipy
import math
import itertools
import time

from tools import sum_and_count
from tools import get_avg_map
from dcom.tools import check_and_increment

def print_timing(func):
    def wrapper(*arg):
        t1 = time.time()
        res = func(*arg)
        t2 = time.time()
        print '%s took %0.3f ms' % (func.func_name, (t2-t1)*1000.0)
        return res
    return wrapper


class NetworkX():

    def __init__(self, options):
        self.G = nx.Graph()
        if options['action']=="metrics":
            self.metrics = MetricsRelative(self, options) if options['relative'] else MetricsAbsolute(self, options)
        else:
            self.metrics = None
        self.partition = None

    def addLink(self, nodeA, nodeB, ts=None, weight=1):
        
        if not self.metrics is None:
            self.metrics.newEventPre(nodeA, nodeB, ts)
        
        hadNodeA = nodeA in self.G
        hadNodeB = nodeB in self.G

        self.G.add_edge(nodeA, nodeB, timestamp=ts, weight=weight)
        
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
            for nn in self.G.neighbors(n):
                nodes.extend(self.G.neighbors(nn))
        return self.G.subgraph(nodes)
    
    def save(self, fname, node=None, outFormat="gml"):
        towrite = self.G if node is None else self.getNeighbourhood(int(node))
        if outFormat=="gml":
            nx.write_gml(towrite, fname)
        elif outFormat=="edgelist":
            nx.write_edgelist(towrite, fname, delimiter=' ')
        
    def getClustCoef(self, ntype):
        nodes = set(n for n,d in self.G.nodes(data=True) if d["type"]==ntype)
        clustMap = bipartite.clustering(self.G, nodes, mode='dot')
        return clustMap
    
    def getLocalClustCoef(self, node):
        return bipartite.clustering(self.G, [node], mode='dot')[node] if node in self.G else 0.0
    
    def getClustCoefOpsahlOriginal(self, ntype):
        nodes = set(n for n,d in self.G.nodes(data=True) if d["type"]==ntype)
        num4Paths = {}
        num4Cycles = {}
        clustMap = {}
        total4Paths, total4Cycles = 0, 0
        for p1 in nodes:
            for s1 in self.G.neighbors(p1): # 1-path
                for p2 in self.G.neighbors(s1): # 2-path
                    if p2 == p1:
                        continue
                    for s2 in self.G.neighbors(p2): # 3-path
                        if s2 == s1:
                            continue
                        for p3 in self.G.neighbors(s2): # 4-path
                            if p3 == p2 or p3 == p1:
                                continue
                            focalnode = p2
                            total4Paths += 1
                            check_and_increment(focalnode, num4Paths);
                            # check for cycle
                            isClosed = False
                            for s3 in self.G.neighbors(p3): # 5-path
                                if s3 == s2 or s3 == s1:
                                    continue
                                for p4 in self.G.neighbors(s3):
                                    if p4 == p1:
                                        isClosed = True
                                        break
                                if isClosed: # only needs to be one cycle for this 4-path
                                    break
                            if isClosed:
                                total4Cycles += 1
                                check_and_increment(focalnode, num4Cycles);
                            
        for n, paths in sorted(num4Paths.iteritems()):
            cycles = num4Cycles[n] if n in num4Cycles else 0
            clustMap[n] = round(float(cycles) / paths, 6)
#            print "node", n, ":", paths, cycles, clustMap[n]
        globalCoef = round(float(total4Cycles) / total4Paths, 6)
        
        return globalCoef, clustMap

    @print_timing
    def getClustCoefOpsahlAll(self, ntype):
        nodes = set(n for n,d in self.G.nodes(data=True) if d["type"]==ntype)
#        nodes = [4]
        nbrsMap, coefMap, coefDgMap = {}, {}, {}
        degreeCoefMap, degreeCoefDgMap = {}, {}
        total4Paths, totalVal4Paths, total4Cycles, totalVal4Cycles = 0, 0.0, 0, 0.0
        for node in sorted(nodes):
            num4Paths, val4Paths, num4Cycles, val4Cycles = self.__getPathsAndCycles(node, nbrsMap)
            total4Paths += num4Paths
            totalVal4Paths += val4Paths
            total4Cycles += num4Cycles
            totalVal4Cycles += val4Cycles

            if num4Paths>0:
                degree = self.G.degree(node)
                coef = round(float(num4Cycles) / num4Paths, 3)
                coef_dg = round(float(val4Cycles) / val4Paths, 3)
                coefMap[node] = coef
                coefDgMap[node] = coef_dg
                sum_and_count(degreeCoefMap, degree, coef)
                sum_and_count(degreeCoefDgMap, degree, coef_dg)
#                print node, num4Cycles, num4Paths, int(val4Cycles), int(val4Paths)
#                if (coefDgMap[node] >= coefMap[node]):
#                print "node", node, ":", num4Paths, round(val4Paths,0), num4Cycles, round(val4Cycles, 0), coefMap[node], coefDgMap[node]
        
        globalCoef = round(float(total4Cycles) / total4Paths, 3)
        globalDgCoef = round(float(totalVal4Cycles) / totalVal4Paths, 3)
        return globalCoef, globalDgCoef, coefMap, coefDgMap, get_avg_map(degreeCoefMap), get_avg_map(degreeCoefDgMap)
    
    def getClustCoefOpsahlLocal(self, node):
        num4Paths, num4Cycles, _, _ = self.__getPathsAndCycles(node)
        return round(float(num4Cycles) / num4Paths, 3) if num4Paths > 0 else None
    
    def __getPathsAndCycles(self, node, nbrsMap={}):
        num4Paths = 0
        val4Paths = 0.0
        num4Cycles = 0
        val4Cycles = 0.0
        
        nbrs = [n for n in self.G[node]]
        for (n1_1, n1_2) in list(itertools.product(nbrs, repeat=2)):
            if n1_1 >= n1_2:
                continue

            w_n1_1 = self.__degree_weight(self.G.degree(n1_1))
            w_n1_2 = self.__degree_weight(self.G.degree(n1_2))

            nbrs2_1 = set(self.G[n1_1])-set([node])
            nbrs2_2 = set(self.G[n1_2])-set([node])
            
            nbrs2_common = nbrs2_1 & nbrs2_2
            n4Paths = len(nbrs2_1) * len(nbrs2_2) - len(nbrs2_common)

            num4Paths += n4Paths
            val4Paths += (n4Paths * (float(w_n1_1 + w_n1_2)/2))
            
            n4Cycles, v4Cycles = self.__getCycles(nbrs2_1, nbrs2_2, n1_1, n1_2, nbrsMap)
            num4Cycles += n4Cycles
            val4Cycles += v4Cycles

        return num4Paths, val4Paths, num4Cycles, val4Cycles

    def __getCycles(self, nbrs2_1, nbrs2_2, n1_1, n1_2, nbrsMap):
        num4Cycles = 0
        val4Cycles = 0
        w_n1_1 = self.__degree_weight(self.G.degree(n1_1))
        w_n1_2 = self.__degree_weight(self.G.degree(n1_2))
        
        for (n2_1, n2_2) in list(itertools.product(nbrs2_1, nbrs2_2)):
            if n2_1 == n2_2:
                continue
            
            if n2_1<n2_2:
                nmin, nmax = n2_1, n2_2
            else:
                nmin, nmax = n2_2, n2_1

            nbrid = (nmax<<32) + nmin
            if not nbrid in nbrsMap:
                nbrs3_common = set(self.G[n2_1]) & set(self.G[n2_2])
                common_weight = self.__degree_weight_nodes(nbrs3_common) if len(nbrs3_common)>0 else 0
                nbrsMap[nbrid] = (common_weight, nbrs3_common)
            else:
                (common_weight, nbrs3_common) = nbrsMap[nbrid]
            
            common_len = len(nbrs3_common)
            if n1_1 in nbrs3_common:
                common_len -= 1
                common_weight -= w_n1_1
            if n1_2 in nbrs3_common:
                common_len -= 1
                common_weight -= w_n1_2
            if common_len>0:
                num4Cycles += 1
                val4Cycles += (float(common_weight + w_n1_1 + w_n1_2) / 3)

        return num4Cycles, val4Cycles


    def getD2Neighbors(self, node):
        nodes = []
        for n in self.G.neighbors(node):
            nodes.extend(self.G.neighbors(n))
        return set(nodes) - set([node])

    def getNodeClusterCoefLatapy(self, node, doprint=False):
        if not self.hasNode(node):
            return 0.0
#        doprint = True if node==18 else False
        
        cc_degree = 0.0
        cc_orig = 0.0
        nbrs2=set([u for nbr in self.G[node] for u in self.G[nbr]])-set([node])

        if doprint:
            print "Node:", node
            print "Node degree:", self.G.degree(node)
            nbrdetails = sorted([[self.G.degree(n), n, round(self.__degree_weight(self.G.degree(n)),3)] for n in self.G[node]], reverse=True)
            print "D1 neighbours:", "|".join([str(n) for [_,n,_] in nbrdetails])
            print "Neigh degrees:", [d for [d,_,_] in nbrdetails]
            print "Neigh contrib:", [c for [_,_,c] in nbrdetails]
#            print "D2 neighbours:", sorted(list(nbrs2))
            neigh_details=[]
        for u in nbrs2:
            ccuv_degree = self.__cc_degree(self.G, u, node, doprint)
            ccuv_orig = self.__cc_orig(self.G, u, node)
            if doprint:
                common = len(set(self.G[u]) & set(self.G[node]))
                if common>1:
                    neigh_details.append([u, common, round(ccuv_orig,2), round(ccuv_degree,2)])
                print str(node)+"-"+str(u)+": orig="+str(round(ccuv_orig,2)) + ", new=" + str(round(ccuv_degree,2))
            cc_degree += ccuv_degree
            cc_orig += ccuv_orig

        if cc_degree > 0.0: 
            cc_degree /= len(nbrs2)
        if cc_orig > 0.0: 
            cc_orig /= len(nbrs2)
            
        if doprint:
#            print "D2 neighbors", neigh_details
            print "cc new:", str(cc_degree)
            print "cc orig:", str(cc_orig)

        return cc_orig, cc_degree

    def getClusterCoefLatapy(self, ntype):
        nodes = set(n for n,d in self.G.nodes(data=True) if d["type"]==ntype)
        coefMap, coefDgMap = {}, {}
        degreeCoefMap, degreeCoefDgMap = {}, {}
        for node in nodes:
            degree = self.G.degree(node)
            cc_orig, cc_degree = self.getNodeClusterCoefLatapy(node)
            coefMap[node] = cc_orig
            sum_and_count(degreeCoefMap, degree, cc_orig)
            coefDgMap[node] = cc_degree
            sum_and_count(degreeCoefDgMap, degree, cc_degree)
            
        return coefMap, coefDgMap, get_avg_map(degreeCoefMap), get_avg_map(degreeCoefDgMap)
    
    def __cc_orig(self, G, u, v):
        nu, nv = set(G[u]), set(G[v])
        return float(len(nu & nv))/len(nu | nv)

    def __cc_degree(self, G, u, v, doprint=False):
        nu, nv = set(G[u]), set(G[v])
        dsum = 0.0
        lsum = 0.0
        nu_and_nv = nu & nv
        for z in nu | nv:
            score = self.__degree_weight(G.degree(z))
            if z in nu_and_nv:
#                if doprint:
#                    print "D" + str(z) + "=" + str(G.degree(z)) + ", " + str(self.__degree_weight(G.degree(z)))
                dsum += score
#            if G.degree(z)>1: 
            lsum += score
#            lsum += 1
            
        return dsum/lsum

    def __degree_weight_nodes(self, nodes):
        w_max = 0
        for node in nodes:
            w_node = self.__degree_weight(self.G.degree(node))
            if w_node > w_max:
                w_max = w_node
        return w_max
        
    def __degree_weight(self, d):
        discrete_d = int(2+d/10)
        return 1/math.log(discrete_d,2) if d>1 else 0.0


    @print_timing
    def getCollabSimilarityAll(self, ntype):
        nodes = set(n for n,d in self.G.nodes(data=True) if d["type"]==ntype)
#        nodes = [4]
        simMap = {}
        degreeSimMap = {}
        pairMap = {}
        for node in sorted(nodes):
            sim = self.getCollabSimilarity(node, pairMap)
            if sim == None:
                continue
            simMap[node] = sim
            sum_and_count(degreeSimMap, self.G.degree(node)/10, sim)
        
        return simMap, get_avg_map(degreeSimMap)
    
    def getCollabSimilarity(self, node, pairMap):
        nbrs = [n for n in self.G[node]]
        sum_sim = 0.0
        count = 0
#        print node
        for (n1_1, n1_2) in list(itertools.product(nbrs, repeat=2)):
            if n1_1 >= n1_2:
                continue
            nid = str(n1_1)+"_"+str(n1_2)
            if nid in pairMap:
                sim = pairMap[nid]
            else:
                sim = self.__similarity(n1_1, n1_2)
                pairMap[nid] = sim
                
            sum_sim += sim
            count += 1

        return sum_sim/count if count > 0 else None
    
    def __similarity(self, n1, n2):
        nbrs_n1 = set(self.G[n1])
        nbrs_n2 = set(self.G[n2])
        
        sim = float(len(nbrs_n1 & nbrs_n2)) / len(nbrs_n1 | nbrs_n2)
#        print n1, n2, round(sim, 3)
        return sim

    def transformTfIdf(self, outf):
        N_users = float(len(set(n for n,d in self.G.nodes(data=True) if d["type"]==0)))
        for e in self.G.edges_iter(data=True):
            n = float(int(e[2]['weight']))
#            d_user = self.G.degree(e[0])
            d_user = self.G.degree(e[0], weight='weight')
            d_object = self.G.degree(e[1])
            w = (n/d_user) * math.log(N_users/d_object, 2)
            outf.write( "%s\t%s\t0\t%.3f\n" % (e[0], e[1], w))

