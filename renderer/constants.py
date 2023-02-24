"""
    Some constants and enumerants
    @author: Qianyue He
    @date: 2023-2-24
"""

# FIXME: TRANSPORT_UNI is used in UDPT, TRANSPORT_RAD seems to be erroneous with refraction
TRANSPORT_UNI = -1      # Unidirectional Path tracing
TRANSPORT_RAD = 0       # radiance transport
TRANSPORT_IMP = 1       # importance transport

VERTEX_SURFACE = 0 
VERTEX_MEDIUM  = 1 
VERTEX_EMITTER = 2 
VERTEX_CAMERA  = 3 