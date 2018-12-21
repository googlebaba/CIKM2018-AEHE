# coding: utf-8

class Node:
    def __init__(self):
        self.in_nodes = []
        self.out_nodes = []
        self.in_ids = []
        self.out_ids = []
        self.neighbours = ()
        self.id = None
        self.type = None
        self.value = None

    def getId(self):
        return self.id
    
    def setId(self, id):
        self.id = id

    def getType(self):
        return self.type

    def setType(self, type):
        self.type = type

    def setTypeId(self, typeId):
        self.typeId = typeId

    def hashCode(self):
        return self.id

    def equals(self, obj):
        if isinstance(obj, Node):
            if Node.getId() == this.id:
                return True
        return False
    def getValue(self):
        return self.value

    def setValue(self, value):
        self.value = value


