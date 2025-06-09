class Set:
    
    def __init__(self, name):
        self.name = name


    def constrain(self, var):
        raise NotImplementedError("Subclasses must implement this method")
