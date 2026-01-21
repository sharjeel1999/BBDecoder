

class FlowArchive:
    def __init__(self):
        self.max_index = 0
        self.flow_archive = {}
    
    def add_flow(self, flow):
        """
        Adds a flow to the archive.
        :param flow: The flow to be added.
        """
        self.flow_archive[self.max_index] = flow
        self.max_index += 1
    
    def get_flow(self, index):
        """
        Retrieves a flow from the archive by its index.
        :param index: The index of the flow to retrieve.
        :return: The flow corresponding to the given index.
        """
        return self.flow_archive.get(index, None)
    
    def get_all_flows(self):
        """
        Retrieves all flows in the archive.
        :return: A list of all flows.
        """
        return list(self.flow_archive.values())