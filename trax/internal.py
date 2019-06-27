# -*- coding: utf-8 -*-
#
# WORD_SIZE is: 8
# POINTER_SIZE is: 8
# LONGDOUBLE_SIZE is: 16
#
import ctypes, sys

# Generated using https://github.com/trolldbois/ctypeslib
# clang2py -l trax --debug ../include/trax.h

# if local wordsize is same as target, keep ctypes pointer function.
if ctypes.sizeof(ctypes.c_void_p) == 8:
    POINTER_T = ctypes.POINTER
else:
    # required to access _ctypes
    import _ctypes
    # Emulate a pointer class using the approriate c_int32/c_int64 type
    # The new class should have :
    # ['__module__', 'from_param', '_type_', '__dict__', '__weakref__', '__doc__']
    # but the class should be submitted to a unique instance for each base type
    # to that if A == B, POINTER_T(A) == POINTER_T(B)
    ctypes._pointer_t_type_cache = {}
    def POINTER_T(pointee):
        # a pointer should have the same length as LONG
        fake_ptr_base_type = ctypes.c_uint64 
        # specific case for c_void_p
        if pointee is None: # VOID pointer type. c_void_p.
            pointee = type(None) # ctypes.c_void_p # ctypes.c_ulong
            clsname = 'c_void'
        else:
            clsname = pointee.__name__
        if clsname in ctypes._pointer_t_type_cache:
            return ctypes._pointer_t_type_cache[clsname]
        # make template
        class _T(_ctypes._SimpleCData,):
            _type_ = 'L'
            _subtype_ = pointee
            def _sub_addr_(self):
                return self.value
            def __repr__(self):
                return '%s(%d)'%(clsname, self.value)
            def contents(self):
                raise TypeError('This is not a ctypes pointer.')
            def __init__(self, **args):
                raise TypeError('This is not a ctypes pointer. It is not instanciable.')
        _class = type('LP_%d_%s'%(8, clsname), (_T,),{}) 
        ctypes._pointer_t_type_cache[clsname] = _class
        return _class

c_int128 = ctypes.c_ubyte*16
c_uint128 = c_int128
void = None
if ctypes.sizeof(ctypes.c_longdouble) == 16:
    c_long_double_t = ctypes.c_longdouble
else:
    c_long_double_t = ctypes.c_ubyte*16

_libraries = {}
if sys.platform.startswith('linux'):
    _libraries['trax'] = ctypes.CDLL('libtrax.so')
elif sys.platform in ['darwin']:
    _libraries['trax'] = ctypes.CDLL('libtrax.dynlib')
elif sys.platform in ['win32']:
    _libraries['trax'] = ctypes.CDLL('trax.dll')
else:
    raise RuntimeError('Unsupported platform')

class struct_trax_image(ctypes.Structure):
    _pack_ = True # source:False
    _fields_ = [
    ('type', ctypes.c_int16),
    ('PADDING_0', ctypes.c_ubyte * 2),
    ('width', ctypes.c_int32),
    ('height', ctypes.c_int32),
    ('format', ctypes.c_int32),
    ('data', POINTER_T(ctypes.c_char)),
     ]

trax_image = struct_trax_image
trax_region = None
class struct_trax_properties(ctypes.Structure):
    pass

trax_properties = struct_trax_properties
class struct_trax_bounds(ctypes.Structure):
    _pack_ = True # source:False
    _fields_ = [
    ('top', ctypes.c_float),
    ('bottom', ctypes.c_float),
    ('left', ctypes.c_float),
    ('right', ctypes.c_float),
     ]

trax_bounds = struct_trax_bounds
trax_logger = ctypes.CFUNCTYPE(None, POINTER_T(ctypes.c_char), ctypes.c_int32, POINTER_T(None))
trax_enumerator = POINTER_T(ctypes.CFUNCTYPE(None, POINTER_T(ctypes.c_char), POINTER_T(ctypes.c_char), POINTER_T(None)))
class struct_trax_logging(ctypes.Structure):
    pass

struct_trax_logging._pack_ = True # source:False
struct_trax_logging._fields_ = [
    ('flags', ctypes.c_int32),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('callback', POINTER_T(trax_logger)),
    ('data', POINTER_T(None)),
]

trax_logging = struct_trax_logging
class struct_trax_metadata(ctypes.Structure):
    _pack_ = True # source:False
    _fields_ = [
    ('format_region', ctypes.c_int32),
    ('format_image', ctypes.c_int32),
    ('channels', ctypes.c_int32),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('tracker_name', POINTER_T(ctypes.c_char)),
    ('tracker_description', POINTER_T(ctypes.c_char)),
    ('tracker_family', POINTER_T(ctypes.c_char)),
     ]

trax_metadata = struct_trax_metadata
trax_configuration = struct_trax_metadata
class struct_trax_handle(ctypes.Structure):
    _pack_ = True # source:False
    _fields_ = [
    ('flags', ctypes.c_int32),
    ('version', ctypes.c_int32),
    ('stream', POINTER_T(None)),
    ('logging', trax_logging),
    ('metadata', POINTER_T(struct_trax_metadata)),
     ]

trax_handle = struct_trax_handle
class struct_trax_image_list(ctypes.Structure):
    _pack_ = True # source:False
    _fields_ = [
    ('images', POINTER_T(struct_trax_image) * 3),
     ]

trax_image_p = POINTER_T(struct_trax_image)
trax_image_list_p = POINTER_T(struct_trax_image_list)
trax_region_p = POINTER_T(None)
trax_properties_p = POINTER_T(struct_trax_properties)

trax_image_list = struct_trax_image_list
trax_no_log = (struct_trax_logging).in_dll(_libraries['trax'], 'trax_no_log')
trax_no_bounds = (struct_trax_bounds).in_dll(_libraries['trax'], 'trax_no_bounds')

trax_version = _libraries['trax'].trax_version
trax_version.restype = ctypes.c_char_p
trax_version.argtypes = []

trax_metadata_create = _libraries['trax'].trax_metadata_create
trax_metadata_create.restype = POINTER_T(struct_trax_metadata)
trax_metadata_create.argtypes = [ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p]

trax_metadata_release = _libraries['trax'].trax_metadata_release
trax_metadata_release.restype = None
trax_metadata_release.argtypes = [POINTER_T(POINTER_T(struct_trax_metadata))]

trax_logger_setup = _libraries['trax'].trax_logger_setup
trax_logger_setup.restype = trax_logging
trax_logger_setup.argtypes = [POINTER_T(trax_logger), POINTER_T(None), ctypes.c_int32]

trax_client_setup_file = _libraries['trax'].trax_client_setup_file
trax_client_setup_file.restype = POINTER_T(struct_trax_handle)
trax_client_setup_file.argtypes = [ctypes.c_int32, ctypes.c_int32, trax_logging]

trax_client_setup_socket = _libraries['trax'].trax_client_setup_socket
trax_client_setup_socket.restype = POINTER_T(struct_trax_handle)
trax_client_setup_socket.argtypes = [ctypes.c_int32, ctypes.c_int32, trax_logging]

trax_client_wait = _libraries['trax'].trax_client_wait
trax_client_wait.restype = ctypes.c_int32
trax_client_wait.argtypes = [POINTER_T(struct_trax_handle), POINTER_T(POINTER_T(None)), POINTER_T(struct_trax_properties)]

trax_client_initialize = _libraries['trax'].trax_client_initialize
trax_client_initialize.restype = ctypes.c_int32
trax_client_initialize.argtypes = [POINTER_T(struct_trax_handle), POINTER_T(struct_trax_image_list), POINTER_T(None), POINTER_T(struct_trax_properties)]

trax_client_frame = _libraries['trax'].trax_client_frame
trax_client_frame.restype = ctypes.c_int32
trax_client_frame.argtypes = [POINTER_T(struct_trax_handle), POINTER_T(struct_trax_image_list), POINTER_T(struct_trax_properties)]

trax_server_setup = _libraries['trax'].trax_server_setup
trax_server_setup.restype = POINTER_T(struct_trax_handle)
trax_server_setup.argtypes = [POINTER_T(struct_trax_metadata), trax_logging]

trax_server_setup_file = _libraries['trax'].trax_server_setup_file
trax_server_setup_file.restype = POINTER_T(struct_trax_handle)
trax_server_setup_file.argtypes = [POINTER_T(struct_trax_metadata), ctypes.c_int32, ctypes.c_int32, trax_logging]

trax_server_wait = _libraries['trax'].trax_server_wait
trax_server_wait.restype = ctypes.c_int32
trax_server_wait.argtypes = [POINTER_T(struct_trax_handle), POINTER_T(POINTER_T(struct_trax_image_list)), POINTER_T(POINTER_T(None)), POINTER_T(struct_trax_properties)]

trax_server_reply = _libraries['trax'].trax_server_reply
trax_server_reply.restype = ctypes.c_int32
trax_server_reply.argtypes = [POINTER_T(struct_trax_handle), POINTER_T(None), POINTER_T(struct_trax_properties)]

trax_terminate = _libraries['trax'].trax_terminate
trax_terminate.restype = ctypes.c_int32
trax_terminate.argtypes = [POINTER_T(struct_trax_handle)]

trax_cleanup = _libraries['trax'].trax_cleanup
trax_cleanup.restype = ctypes.c_int32
trax_cleanup.argtypes = [POINTER_T(POINTER_T(struct_trax_handle))]

trax_set_parameter = _libraries['trax'].trax_set_parameter
trax_set_parameter.restype = ctypes.c_int32
trax_set_parameter.argtypes = [POINTER_T(struct_trax_handle), ctypes.c_int32, ctypes.c_int32]

trax_get_parameter = _libraries['trax'].trax_get_parameter
trax_get_parameter.restype = ctypes.c_int32
trax_get_parameter.argtypes = [POINTER_T(struct_trax_handle), ctypes.c_int32, POINTER_T(ctypes.c_int32)]

trax_image_release = _libraries['trax'].trax_image_release
trax_image_release.restype = None
trax_image_release.argtypes = [POINTER_T(POINTER_T(struct_trax_image))]

trax_image_create_path = _libraries['trax'].trax_image_create_path
trax_image_create_path.restype = POINTER_T(struct_trax_image)
trax_image_create_path.argtypes = [POINTER_T(ctypes.c_char)]

trax_image_create_url = _libraries['trax'].trax_image_create_url
trax_image_create_url.restype = POINTER_T(struct_trax_image)
trax_image_create_url.argtypes = [POINTER_T(ctypes.c_char)]

trax_image_create_memory = _libraries['trax'].trax_image_create_memory
trax_image_create_memory.restype = POINTER_T(struct_trax_image)
trax_image_create_memory.argtypes = [ctypes.c_int32, ctypes.c_int32, ctypes.c_int32]

trax_image_create_buffer = _libraries['trax'].trax_image_create_buffer
trax_image_create_buffer.restype = POINTER_T(struct_trax_image)
trax_image_create_buffer.argtypes = [ctypes.c_int32, ctypes.c_char_p]

trax_image_get_type = _libraries['trax'].trax_image_get_type
trax_image_get_type.restype = ctypes.c_int32
trax_image_get_type.argtypes = [POINTER_T(struct_trax_image)]

trax_image_get_path = _libraries['trax'].trax_image_get_path
trax_image_get_path.restype = ctypes.c_char_p
trax_image_get_path.argtypes = [POINTER_T(struct_trax_image)]

trax_image_get_url = _libraries['trax'].trax_image_get_url
trax_image_get_url.restype = ctypes.c_char_p
trax_image_get_url.argtypes = [POINTER_T(struct_trax_image)]

trax_image_get_memory_header = _libraries['trax'].trax_image_get_memory_header
trax_image_get_memory_header.restype = None
trax_image_get_memory_header.argtypes = [POINTER_T(struct_trax_image), POINTER_T(ctypes.c_int32), POINTER_T(ctypes.c_int32), POINTER_T(ctypes.c_int32)]

trax_image_write_memory_row = _libraries['trax'].trax_image_write_memory_row
trax_image_write_memory_row.restype = POINTER_T(ctypes.c_char)
trax_image_write_memory_row.argtypes = [POINTER_T(struct_trax_image), ctypes.c_int32]

trax_image_get_memory_row = _libraries['trax'].trax_image_get_memory_row
trax_image_get_memory_row.restype = POINTER_T(ctypes.c_char)
trax_image_get_memory_row.argtypes = [POINTER_T(struct_trax_image), ctypes.c_int32]

trax_image_get_buffer = _libraries['trax'].trax_image_get_buffer
trax_image_get_buffer.restype = POINTER_T(ctypes.c_char)
trax_image_get_buffer.argtypes = [POINTER_T(struct_trax_image), POINTER_T(ctypes.c_int32), POINTER_T(ctypes.c_int32)]

trax_region_release = _libraries['trax'].trax_region_release
trax_region_release.restype = None
trax_region_release.argtypes = [POINTER_T(POINTER_T(None))]

trax_region_get_type = _libraries['trax'].trax_region_get_type
trax_region_get_type.restype = ctypes.c_int32
trax_region_get_type.argtypes = [POINTER_T(None)]

trax_region_create_special = _libraries['trax'].trax_region_create_special
trax_region_create_special.restype = ctypes.c_void_p
trax_region_create_special.argtypes = [ctypes.c_int32]

trax_region_set_special = _libraries['trax'].trax_region_set_special
trax_region_set_special.restype = None
trax_region_set_special.argtypes = [POINTER_T(None), ctypes.c_int32]

trax_region_get_special = _libraries['trax'].trax_region_get_special
trax_region_get_special.restype = ctypes.c_int32
trax_region_get_special.argtypes = [POINTER_T(None)]

trax_region_create_rectangle = _libraries['trax'].trax_region_create_rectangle
trax_region_create_rectangle.restype = POINTER_T(None)
trax_region_create_rectangle.argtypes = [ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float]

trax_region_set_rectangle = _libraries['trax'].trax_region_set_rectangle
trax_region_set_rectangle.restype = None
trax_region_set_rectangle.argtypes = [POINTER_T(None), ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float]

trax_region_get_rectangle = _libraries['trax'].trax_region_get_rectangle
trax_region_get_rectangle.restype = None
trax_region_get_rectangle.argtypes = [POINTER_T(None), POINTER_T(ctypes.c_float), POINTER_T(ctypes.c_float), POINTER_T(ctypes.c_float), POINTER_T(ctypes.c_float)]

trax_region_create_polygon = _libraries['trax'].trax_region_create_polygon
trax_region_create_polygon.restype = POINTER_T(None)
trax_region_create_polygon.argtypes = [ctypes.c_int32]

trax_region_set_polygon_point = _libraries['trax'].trax_region_set_polygon_point
trax_region_set_polygon_point.restype = None
trax_region_set_polygon_point.argtypes = [POINTER_T(None), ctypes.c_int32, ctypes.c_float, ctypes.c_float]

trax_region_get_polygon_point = _libraries['trax'].trax_region_get_polygon_point
trax_region_get_polygon_point.restype = None
trax_region_get_polygon_point.argtypes = [POINTER_T(None), ctypes.c_int32, POINTER_T(ctypes.c_float), POINTER_T(ctypes.c_float)]

trax_region_get_polygon_count = _libraries['trax'].trax_region_get_polygon_count
trax_region_get_polygon_count.restype = ctypes.c_int32
trax_region_get_polygon_count.argtypes = [POINTER_T(None)]

trax_region_bounds = _libraries['trax'].trax_region_bounds
trax_region_bounds.restype = trax_bounds
trax_region_bounds.argtypes = [POINTER_T(None)]

trax_region_contains = _libraries['trax'].trax_region_contains
trax_region_contains.restype = ctypes.c_int32
trax_region_contains.argtypes = [POINTER_T(None), ctypes.c_float, ctypes.c_float]

trax_region_clone = _libraries['trax'].trax_region_clone
trax_region_clone.restype = POINTER_T(None)
trax_region_clone.argtypes = [POINTER_T(None)]

trax_region_convert = _libraries['trax'].trax_region_convert
trax_region_convert.restype = POINTER_T(None)
trax_region_convert.argtypes = [POINTER_T(None), ctypes.c_int32]

trax_region_overlap = _libraries['trax'].trax_region_overlap
trax_region_overlap.restype = ctypes.c_float
trax_region_overlap.argtypes = [POINTER_T(None), POINTER_T(None), trax_bounds]

trax_region_encode = _libraries['trax'].trax_region_encode
trax_region_encode.restype = POINTER_T(ctypes.c_char)
trax_region_encode.argtypes = [POINTER_T(None)]

trax_region_decode = _libraries['trax'].trax_region_decode
trax_region_decode.restype = POINTER_T(None)
trax_region_decode.argtypes = [POINTER_T(ctypes.c_char)]

trax_properties_release = _libraries['trax'].trax_properties_release
trax_properties_release.restype = None
trax_properties_release.argtypes = [POINTER_T(POINTER_T(struct_trax_properties))]

trax_properties_clear = _libraries['trax'].trax_properties_clear
trax_properties_clear.restype = None
trax_properties_clear.argtypes = [POINTER_T(struct_trax_properties)]

trax_properties_create = _libraries['trax'].trax_properties_create
trax_properties_create.restype = POINTER_T(struct_trax_properties)
trax_properties_create.argtypes = []

trax_properties_set = _libraries['trax'].trax_properties_set
trax_properties_set.restype = None
trax_properties_set.argtypes = [POINTER_T(struct_trax_properties), ctypes.c_char_p, ctypes.c_char_p]

trax_properties_set_int = _libraries['trax'].trax_properties_set_int
trax_properties_set_int.restype = None
trax_properties_set_int.argtypes = [POINTER_T(struct_trax_properties), ctypes.c_char_p, ctypes.c_int32]

trax_properties_set_float = _libraries['trax'].trax_properties_set_float
trax_properties_set_float.restype = None
trax_properties_set_float.argtypes = [POINTER_T(struct_trax_properties), ctypes.c_char_p, ctypes.c_float]

trax_properties_get = _libraries['trax'].trax_properties_get
trax_properties_get.restype = ctypes.c_char_p
trax_properties_get.argtypes = [POINTER_T(struct_trax_properties), ctypes.c_char_p]

trax_properties_get_int = _libraries['trax'].trax_properties_get_int
trax_properties_get_int.restype = ctypes.c_int32
trax_properties_get_int.argtypes = [POINTER_T(struct_trax_properties), ctypes.c_char_p, ctypes.c_int32]

trax_properties_get_float = _libraries['trax'].trax_properties_get_float
trax_properties_get_float.restype = ctypes.c_float
trax_properties_get_float.argtypes = [POINTER_T(struct_trax_properties), ctypes.c_char_p, ctypes.c_float]

trax_properties_count = _libraries['trax'].trax_properties_count
trax_properties_count.restype = ctypes.c_int32
trax_properties_count.argtypes = [POINTER_T(struct_trax_properties)]

trax_properties_enumerate = _libraries['trax'].trax_properties_enumerate
trax_properties_enumerate.restype = None
trax_properties_enumerate.argtypes = [POINTER_T(struct_trax_properties), trax_enumerator, POINTER_T(None)]

trax_image_list_create = _libraries['trax'].trax_image_list_create
trax_image_list_create.restype = trax_image_list_p
trax_image_list_create.argtypes = []

trax_image_list_release = _libraries['trax'].trax_image_list_release
trax_image_list_release.restype = None
trax_image_list_release.argtypes = [POINTER_T(trax_image_list_p)]

trax_image_list_clear = _libraries['trax'].trax_image_list_clear
trax_image_list_clear.restype = None
trax_image_list_clear.argtypes = [trax_image_list_p]

trax_image_list_get = _libraries['trax'].trax_image_list_get
trax_image_list_get.restype = trax_image_p
trax_image_list_get.argtypes = [trax_image_list_p, ctypes.c_int32]

trax_image_list_set = _libraries['trax'].trax_image_list_set
trax_image_list_set.restype = None
trax_image_list_set.argtypes = [trax_image_list_p, trax_image_p, ctypes.c_int32]

__all__ = \
    ['struct_trax_bounds', 'struct_trax_handle',
    'struct_trax_image', 'struct_trax_image_list',
    'struct_trax_logging', 'struct_trax_metadata',
    'struct_trax_properties', 'trax_bounds', 'trax_cleanup',
    'trax_client_frame', 'trax_client_initialize',
    'trax_client_setup_file', 'trax_client_setup_socket',
    'trax_client_wait', 'trax_configuration', 'trax_enumerator',
    'trax_get_parameter', 'trax_handle', 'trax_image',
    'trax_image_create_buffer', 'trax_image_create_memory',
    'trax_image_create_path', 'trax_image_create_url',
    'trax_image_get_buffer', 'trax_image_get_memory_header',
    'trax_image_get_memory_row', 'trax_image_get_path',
    'trax_image_get_type', 'trax_image_get_url', 'trax_image_list',
    'trax_image_release', 'trax_image_write_memory_row',
    'trax_logger', 'trax_logger_setup', 
    'trax_logging', 'trax_metadata', 'trax_metadata_create',
    'trax_metadata_release', 'trax_no_bounds', 'trax_no_log',
    'trax_properties', 'trax_properties_clear',
    'trax_properties_count', 'trax_properties_create',
    'trax_properties_enumerate', 'trax_properties_get',
    'trax_properties_get_float', 'trax_properties_get_int',
    'trax_properties_release', 'trax_properties_set',
    'trax_properties_set_float', 'trax_properties_set_int',
    'trax_region', 'trax_region_bounds', 'trax_region_clone',
    'trax_region_contains', 'trax_region_convert',
    'trax_region_create_polygon', 'trax_region_create_rectangle',
    'trax_region_create_special', 'trax_region_decode',
    'trax_region_encode', 'trax_region_get_polygon_count',
    'trax_region_get_polygon_point', 'trax_region_get_rectangle',
    'trax_region_get_special', 'trax_region_get_type',
    'trax_region_overlap', 'trax_region_release',
    'trax_region_set_polygon_point', 'trax_region_set_rectangle',
    'trax_region_set_special', 'trax_server_reply',
    'trax_server_setup', 'trax_server_setup_file', 'trax_server_wait',
    'trax_set_parameter', 'trax_terminate', 'trax_version',
    'trax_image_p', 'trax_region_p', 'trax_properties_p',
    'trax_image_list_p', 'trax_image_list_get', 'trax_image_list_create',
    'trax_image_list_set', 'trax_image_list_release', 'trax_image_list_clear']