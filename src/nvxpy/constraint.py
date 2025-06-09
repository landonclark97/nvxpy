class Constraint:
    def __init__(self, left, op, right):
        assert op in [">=", "<=", "==", ">>", "<<"]
        self.left = left
        self.op = op
        self.right = right
