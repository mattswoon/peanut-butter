import json


def read_json(path):
    with open(path, 'r') as fp:
        conf = json.load(fp)

    regions = map(Region, conf.get('regions'))

    indexer = Indexer(regions)

    S0 = indexer.vector(conf['initial']['susceptible'])
    I0 = indexer.vector(conf['initial']['infected'])
    R0 = indexer.vector(conf['initial']['recovered'])

    gamma = conf['parameters']['gamma']
