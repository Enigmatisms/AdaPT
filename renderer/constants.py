"""
    Some constants and enumerants
    @author: Qianyue He
    @date: 2023-2-24
"""
from taichi.math import pi, vec3

# ================ Enumerant tags ==================
# FIXME: TRANSPORT_UNI is used in UDPT, TRANSPORT_RAD seems to be erroneous with refraction
TRANSPORT_UNI = -1      # Unidirectional Path tracing
TRANSPORT_RAD = 0       # radiance transport
TRANSPORT_IMP = 1       # importance transport

VERTEX_SURFACE = 0 
VERTEX_MEDIUM  = 1 
VERTEX_EMITTER = 2 
VERTEX_CAMERA  = 3 
VERTEX_NULL    = -1     # dummpy uninitialized vertex 

ON_SURFACE = 1          # whether on surface

# =============== Math constants ================
INV_PI = 1. / pi
INV_2PI = INV_PI * 0.5
PI2 = 2. * pi

ZERO_V3 = vec3([0, 0, 0])
ONES_V3 = vec3([1, 1, 1])
AXIS_X  = vec3([1, 0, 0])
AXIS_Y  = vec3([0, 1, 0])
AXIS_Z  = vec3([0, 0, 1])