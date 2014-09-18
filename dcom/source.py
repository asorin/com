#!/usr/bin/env python

class CommunitySourceError(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


class NetSource():
    def __init__(self, delimiter, linksFile, nodes1File=None, nodes2File=None, partitionFile=None):
        self.delimiter = delimiter
        self.links = []
        self.nodes = [dict(), dict()]
        self.partition = None
        self.__loadNetwork(linksFile, nodes1File, nodes2File, partitionFile)


    def __loadNetwork(self, linksFile, nodes1File=None, nodes2File=None, partitionFile=None):
        try:
            if nodes1File:
                self.__loadNodes(0, nodes1File);
            if nodes2File:
                self.__loadNodes(1, nodes2File);
            if partitionFile:
                self.__loadPartition(partitionFile)
                
            self.__loadLinks(linksFile);
        
        except IOError as e:
            raise CommunitySourceError("Cannot read from file %s: %s" % (self.srcFile, str(e)))


    def __loadLinks(self, srcFile):
        print "Loading links from %s" % (srcFile)
        f = open(srcFile)

        # read content
        for line in f:
            if len(line) == 0 or line[0] == '#':
                continue
            row = line.strip("\r\n").split(self.delimiter)
            if len(row)<2:
                raise CommunitySourceError("Invalid number of fields: %d" % len(row))
            nodeA = int(row[0])
            nodeB = int(row[1])
            ts = 0
            weight = 1
            if len(row)>2:
                ts = int(row[2])
#                ts = datetime.utcfromtimestamp(float(row[2])/1000)
#            else:
#                ts = datetime.now()
                if len(row)>3:
                    weight = float(row[3])

            self.links.append((nodeA, nodeB, ts, weight))

            if not nodeA in self.nodes[0]:
                self.nodes[0][nodeA] = { "name": str(nodeA), "ts": ts }
            elif ts>0 and self.nodes[0][nodeA]["ts"] == 0:
                self.nodes[0][nodeA]["ts"] = ts

            if not nodeB in self.nodes[1]:
                self.nodes[1][nodeB] = { "name": str(nodeB), "ts": ts }
            elif ts>0 and self.nodes[1][nodeB]["ts"] == 0:
                self.nodes[1][nodeB]["ts"] = ts
        
        f.close()


    def __loadNodes(self, nodeType, nodesFile):
        print "Loading nodes from %s" % (nodesFile)
        f = open(nodesFile)
        
        for line in f:
            row = line.strip("\r\n").split(self.delimiter)
            if len(row) != 2:
                raise CommunitySourceError("Invalid number of fields: %d" % len(row))
            self.nodes[nodeType][int(row[0])] = { "name": row[1].replace(' ','_'), "ts": 0 }
        
        f.close()
    
    def __loadPartition(self, partitionFile):
        print "Loading partition from %s" % (partitionFile)
        f = open(partitionFile)
        self.partition = dict()
        
        for line in f:
            row = line.strip("\r\n").split(self.delimiter)
            if len(row) != 2:
                raise CommunitySourceError("Invalid number of fields: %d" % len(row))
            self.partition[int(row[0])] = int(row[1])

        f.close()
        

    def getNames(self, nodes):
        names = set()
        for n in nodes:
            if n in self.nodes[0]:
                names.add(self.nodes[0][n])
            elif n in self.nodes[1]:
                names.add(self.nodes[1][n])
        return names
