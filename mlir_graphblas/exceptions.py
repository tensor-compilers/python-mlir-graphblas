class GrbError(Exception):
    pass


# API Errors
class GrbNullPointer(GrbError):
    pass


class GrbInvalidValue(GrbError):
    pass


class GrbInvalidIndex(GrbError):
    pass


class GrbDomainMismatch(GrbError):
    pass


class GrbDimensionMismatch(GrbError):
    pass


class GrbOutputNotEmpty(GrbError):
    pass


# Execution Errors
class GrbIndexOutOfBounds(GrbError):
    pass


class GrbEmptyObject(GrbError):
    pass
