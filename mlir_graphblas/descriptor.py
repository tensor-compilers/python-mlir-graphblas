class Descriptor:
    def __init__(self, replace=False, mask_complement=False, mask_structure=False,
                 transpose0=False, transpose1=False):
        self.replace = replace
        self.mask_complement = mask_complement
        self.mask_structure = mask_structure
        self.transpose0 = transpose0
        self.transpose1 = transpose1

    def __repr__(self):
        ret = [
            'R' if self.replace else '.',
            '~' if self.mask_complement else '.',
            'S' if self.mask_structure else 'V',
            'T' if self.transpose0 else '.',
            'T' if self.transpose1 else '.',
        ]
        return ''.join(ret)


# Populate with all builtin descriptors
NULL = Descriptor()
T1 = Descriptor(transpose1=True)
T0 = Descriptor(transpose0=True)
T0T1 = Descriptor(transpose0=True, transpose1=True)
C = Descriptor(mask_complement=True)
S = Descriptor(mask_structure=True)
CT1 = Descriptor(mask_complement=True, transpose1=True)
ST1 = Descriptor(mask_structure=True, transpose1=True)
CT0 = Descriptor(mask_complement=True, transpose0=True)
ST0 = Descriptor(mask_structure=True, transpose0=True)
CT0T1 = Descriptor(mask_complement=True, transpose0=True, transpose1=True)
ST0T1 = Descriptor(mask_structure=True, transpose0=True, transpose1=True)
SC = Descriptor(mask_complement=True, mask_structure=True)
SCT1 = Descriptor(mask_complement=True, mask_structure=True, transpose1=True)
SCT0 = Descriptor(mask_complement=True, mask_structure=True, transpose0=True)
SCT0T1 = Descriptor(mask_complement=True, mask_structure=True, transpose0=True, transpose1=True)
R = Descriptor(replace=True)
RT1 = Descriptor(replace=True, transpose1=True)
RT0 = Descriptor(replace=True, transpose0=True)
RT0T1 = Descriptor(replace=True, transpose0=True, transpose1=True)
RC = Descriptor(replace=True, mask_complement=True)
RS = Descriptor(replace=True, mask_structure=True)
RCT1 = Descriptor(replace=True, mask_complement=True, transpose1=True)
RST1 = Descriptor(replace=True, mask_structure=True, transpose1=True)
RCT0 = Descriptor(replace=True, mask_complement=True, transpose0=True)
RST0 = Descriptor(replace=True, mask_structure=True, transpose0=True)
RCT0T1 = Descriptor(replace=True, mask_complement=True, transpose0=True, transpose1=True)
RST0T1 = Descriptor(replace=True, mask_structure=True, transpose0=True, transpose1=True)
RSC = Descriptor(replace=True, mask_complement=True, mask_structure=True)
RSCT1 = Descriptor(replace=True, mask_complement=True, mask_structure=True, transpose1=True)
RSCT0 = Descriptor(replace=True, mask_complement=True, mask_structure=True, transpose0=True)
RSCT0T1 = Descriptor(replace=True, mask_complement=True, mask_structure=True, transpose0=True, transpose1=True)
