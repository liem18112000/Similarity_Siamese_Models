from utility.proxy.object import Object as Base

class RealObject(Base):

    def __init__(self, adaptee):
        self._adaptee = adaptee

    def request(self, params = None):
        pass