from metrics import MetricsAbsolute
import igraph

class NetworkIG():

    def __init__(self, aggrPeriod = 86400):
        self.G = igraph.Graph()
        self.metrics = MetricsAbsolute(self, aggrPeriod)
        self.partition = None
        self.vindex = dict()
        self.elist = list()

    def addLink(self, nodeA, nodeB, ts=None):
        
        if (not ts is None):
            self.metrics.newEvent(nodeA, nodeB, ts)
        
        self.__addNodes((nodeA, nodeB))
#        self.G.add_edge(self.vindex[nodeA], self.vindex[nodeB], timestamp=ts)
        self.elist.append((self.vindex[nodeA], self.vindex[nodeB]))
        self.G.vs[self.vindex[nodeA]]["type"] = 0
        self.G.vs[self.vindex[nodeB]]["type"] = 1
        self.G.vs[self.vindex[nodeA]]["id"] = nodeA
        self.G.vs[self.vindex[nodeB]]["id"] = nodeB

    def flush(self):
        self.G.add_edges(self.elist)
        
    def __addNodes(self, nodes):
        for node in nodes:
            if not node in self.vindex:
                idx = self.G.vcount()
                self.vindex[node] = idx
                self.G.add_vertex(node, type=0)
            
    def setPartition(self, partition):
        self.partition = partition
                
    def findPartition(self, ntype, threshold):
        prjG = self.__get_projection(ntype, threshold)
        print "Number of projected edges: " + str(prjG.ecount())
        vertices = [v for v in prjG.vs if prjG.degree(v.index) == 0]
        if len(vertices) > 0:
            print "%d vertices have no neighbors!" % (len(vertices))
        prjG.delete_vertices(vertices)

        # apply clustering
        clustering = prjG.community_walktrap(weights="weight")
        return self.__createPartition(clustering.as_clustering())


    def getPartition(self):
        return self.partition

    def hasPartition(self):
        return not self.partition is None
    
    def __get_projection(self, ntype, threshold):
        return self.G.bipartite_projection()[ntype]

    def __createPartition(self, clustering):
        partition = dict()
        cidx = 0
        for cluster in clustering.subgraphs():
            if cluster.vcount() > 1:
                for v in cluster.vs:
                    node = v["id"]
                    partition[node] = cidx
                cidx += 1
        return partition 
        
    def hasNode(self, node):
        return node in self.vindex
    
    def hasEdge(self, nodeA, nodeB):
        return (nodeA, nodeB) in self.G.es
        
    def getNeighbors(self, node):
        neighbors = []
        idxNeigh = self.G.neighborhood(self.vindex[node])
        for idx in idxNeigh:
            neighbors.append(self.G.vs[idx]["id"])
        return neighbors
    
    