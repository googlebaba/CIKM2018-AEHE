# coding: utf-8
#读取图形文件
from Node import Node
class ReadWholeGraph:
    def __init__(self, nodesPath, edgesPath):
        self.nodesPath = nodesPath
        self.edgesPath = edgesPath
    
    def ReadGraph(self):
        data = {}
        author2paperId = {}
        authorId2paperId = {}
        paperId2authorId = {}
        paperId2venueId = {}
        venueId2paperId = {}
        author2id = {}
        with open(self.nodesPath) as f:
            line1 = f.readline()
            for line in f:
                try:
                    line = line.strip().split('\t')
                    node = Node()
                    node.setId(int(line[0]))
                    node.setType(line[1])
                    node.setValue(line[2])
                    data[int(line[0])] = node
                    author2id[line[2]] = int(line[0])
                except Exception as e:
                    print(e)
        print('data', len(data))
        with open(self.edgesPath) as f:
            for line in f:
                line = line.strip().split('\t')
                try:
                    start = int(line[0])
                    end = int(line[1])
                    startNode = data.get(start)
                    endNode = data.get(end)
                    startNode.out_ids.append(end)
                    startNode.out_nodes.append(endNode)
                    endNode.in_ids.append(start)
                    endNode.in_nodes.append(startNode)
                    if startNode.getType() == 'author' and \
                        endNode.getType() == 'paper':
                        if startNode.getValue() not in author2paperId:
                            author2paperId[startNode.getValue()] = [end]
                        else:
                            author2paperId[startNode.getValue()].append(end)
                        if start not in authorId2paperId:
                            authorId2paperId[start] = [end]
                        else:
                            authorId2paperId[start].append(end)
                        if end not in paperId2authorId:
                            paperId2authorId[endNode.getId()] = [start]
                        else:
                            paperId2authorId[endNode.getId()].append(start)
                    if startNode.getType() == 'paper' and \
                        endNode.getType() == 'conf':
                        paperId2venueId[start] = end
                        if end not in venueId2paperId:
                            venueId2paperId[end] = []
                        venueId2paperId[end].append(start)
                        
                except Exception as e:
                    print(e, line)
        return data, author2paperId, paperId2venueId, paperId2authorId, authorId2paperId, venueId2paperId, author2id

if __name__ == "__main__":
    rwg = ReadWholeGraph('../data-prepared/graph.node', '../data-prepared/graph.edge')
    data, author2paper, paper2venue = rwg.ReadGraph()


                
