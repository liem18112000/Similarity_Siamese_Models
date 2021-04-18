from utility.proxy.realObject import RealObject
from utility.proxy.object import Object as Base

class Proxy(Base):

    def _isRealObject(self, obj):
        return isinstance(obj, RealObject)

    def _isRealObjectAvailable(self):
        return self._realObject is not None

    def __init__(self, realObject):
        self._realObject = None
        self.setRealObject(realObject)
        
    def setRealObject(self, realObject):
        if self._isRealObject(realObject):
            self._realObject = realObject
        else:
            raise Exception("realObject is not a real object")

    def request(self, params = None):
        if self._isRealObjectAvailable():
            return self._realObject.request(params)
        raise Exception("realObject is not initialized")
