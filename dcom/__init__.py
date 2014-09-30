#!/usr/bin/env python
# - wrapper for network
# - source reader
# - metrics wrapper
#

from argparse import RawTextHelpFormatter
from collections import defaultdict
import argparse
import logging
import sys
import yaml

from dcom.source import *
#from dcom.network_ig import NetworkIG
from dcom.network_x import NetworkX


from _version import __version__
__version__ = __version__  # Keep pyflakes happy with unused import


logger = logging.getLogger(__name__)

def parse_args(args):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=RawTextHelpFormatter)
    parser.add_argument('-v', '--version', action='version',
            version='version %s' % __version__)
    parser.add_argument('-c', '--config', action='store', default="conf/dcom.yml",
            help=('configuration file'))
    parser.add_argument('-l', '--links', action='store', required=True,
            help=('the links file containing the list of link'))
    parser.add_argument('-n1', '--node1', action='store', default=None,
            help=('the file containing details of first nodes'))
    parser.add_argument('-n2', '--node2', action='store', default=None, 
            help=('the file containing details of second nodes'))
    parser.add_argument('-d', '--delimiter', action='store', default='\t',
            help=('delimiter of fields (default is tab)'))
    parser.add_argument('-o', '--output', action='store', default=None,
            help=('output file'))
    parser.add_argument('-a', '--action', action='store', default="metrics",
            help=('action to be performed'))
    parser.add_argument('-pe', '--period', action='store', default='86400',
            help=('the aggregation period for metrics (in seconds)'))
    parser.add_argument('-p', '--partition', action='store', default=None,
            help=('file containing partition of nodes'))
    parser.add_argument('-lib', '--library', action='store', default="x",
            help=('library to use (x or ig)'))
    parser.add_argument('-rel', '--relative', action='store', default=False,
            help=('use relative time'))
    parser.add_argument('-nd', '--node', action='store', default=None,
            help=('node id'))
    parser.add_argument('-nt', '--ntype', action='store', default=0,
            help=('node type'))
    parser.add_argument('-nc', '--nclusters', action='store', default=0,
            help=('number of clusters'))
    parser.add_argument('-oi', '--onlineinit', action='store', default=0,
            help=('number of nodes to start the online clustering'))
    parser.add_argument('-os', '--onlinestep', action='store', default=1,
            help=('number of nodes in each step for online clustering'))
    parser.add_argument('-rt', '--real-time', action='store', default=0,
            help=('specify whether real-time or not'))


    return parser.parse_args(args)


def read_config(options):
    """ Read the YAML configuration file and return a dict """
    config_file = options['config']
    
    with open(config_file) as f:
        config = yaml.load(f)
    options.update(config)


def do_metrics(options):
    net = options['network']
    outf = options['output_file']
    delimiter = options['delimiter']
    
#    outf.write(delimiter.join(net.metrics.getNames()) + "\n")
    metrics = net.metrics.getMetrics()
    for metric_list in metrics:
        for metric in metric_list:
            if len(metric)>0:
                outf.write(delimiter.join(map(str,metric)) + "\n")

def do_partition_louvain_ntype(options, ntype, startGroup):
    net = options['network']
    outf = options['output_file']
    threshold = options['partition_threshold']
#    src = options['source']
    delimiter = options['delimiter']
    
    partition = net.findPartitionLouvain(ntype, threshold)
    for node, group in partition.iteritems():
        outf.write("%d%s%d\n" % (node, delimiter, startGroup+group))
    
    community_set = set(partition.values())
#    c_named_list = []
    for i in community_set:
        community_nodes = [node for node in partition.keys() if partition[node] == i]
#        c_named_list.append(src.getNames(community_nodes))
#        print i, src.getNames(community_nodes)
        print startGroup+i, len(community_nodes)

    return len(community_set)

def do_partition_louvain(options):
    # partition first set of nodes
    cno = do_partition_ntype(options, 0, 0)
    # partition the second set of nodes
    do_partition_ntype(options, 1, cno)

def write_partition(outf, partition):
    communities = {}
    for node, group in partition.iteritems():
        if not group in communities:
            communities[group] = []
        communities[group].append(node)
    for c in communities.itervalues():
        outf.write("%s\n" % (" ".join(map(str, sorted(c)))))

def do_partition_svd(options):
    net = options['network']
    outf = options['output_file']
    ntype = int(options['ntype'])
    nclusters = int(options['nclusters'])

    partition = net.findPartitionSVD(ntype, nclusters)
    if partition!=None:
        write_partition(outf, partition)

def do_partition_lsi(options):
    net = options['network']
    outf = options['output_file']
    ntype = int(options['ntype'])
    nclusters = int(options['nclusters'])

    partition = net.findPartitionLSI(ntype, nclusters)
    if partition!=None:
        write_partition(outf, partition)

def do_partition_coclust(options):
    net = options['network']
    outf = options['output_file']
    ntype = int(options['ntype'])
    nclusters = int(options['nclusters'])

    partition = net.findPartitionCoClust(ntype, nclusters)
    write_partition(outf, partition)

def do_partition_online(options):
    net = options['network']
    outf = options['output_file']
    ntype = int(options['ntype'])
    nclusters = int(options['nclusters'])
    online_init_nodes = int(options['onlineinit'])
    online_step_nodes = int(options['onlinestep'])

    partition = net.findPartitionOnline(nclusters,online_init_nodes,online_step_nodes)
    if partition!=None:
        write_partition(outf, partition)

def do_partition_incremental(options):
    net = options['network']
    outf = options['output_file']
    ntype = int(options['ntype'])
    nclusters = int(options['nclusters'])
    online_init_nodes = int(options['onlineinit'])

    partition = net.findPartitionIncremental(nclusters,online_init_nodes)
    if partition!=None:
        write_partition(outf, partition)

def do_save(options):
    net = options['network']
    outf = options['output_file']
    node = options['node']
    net.save(outf, node)

def do_save_prj(options):
    net = options['network']
    outf = options['output_file']
    ntype = int(options['ntype'])
    net.savePrjWeighted(ntype, outf)

def do_save_prj_colisted(options):
    net = options['network']
    outf = options['output_file']
    ntype = int(options['ntype'])
    net.savePrjCoCit(ntype, outf)

def do_transform(options):
    net = options['network']
    outf = options['output_file']
    net.transformTfIdf(outf)
        
def main(args):
    actions = { "metrics" : do_metrics, "partition-louvain" : do_partition_louvain, "partition-svd" : do_partition_svd, "partition-lsi" : do_partition_lsi, "partition-coclust" : do_partition_coclust, "partition-online": do_partition_online, "partition-incremental": do_partition_incremental, "save" : do_save, "save_prj" : do_save_prj, "save_prj_colisted" : do_save_prj_colisted, "transform" : do_transform }

    options = vars(parse_args(args or sys.argv[1:]))
    
    if not options['action'] in actions:
        print "Invalid action: ", options['action']
        sys.exit(1)
    
    outf = sys.stdout
    if options['output']:
        outf = open(options['output'],"w")
    options['output_file'] = outf

    options['partition_ntype'] = 1
    options['partition_threshold'] = 1
    
    read_config(options)
    
    try:
        src = NetSource(options['delimiter'], options['links'],
                                options['node1'], options['node2'],
                                options['partition'])
        print "Feed %d links into network" % (len(src.links))
        if options['library']=='x':
            net = NetworkX(options, src)
# igraph not supported
#        elif options['library']=='ig':
#            net = NetworkIG(int(options['period']))
        else:
            raise CommunitySourceError("Invalid library: %s" % (options['library']))
        
        for link in src.links:
            net.addLink(link[0], link[1], link[2], link[3])
        net.flush()
        options['network'] = net
        options['source'] = src
        
        actions[options['action']](options)
        
        outf.close()
        
    except CommunitySourceError as e:
        print e.value
        sys.exit(1)
