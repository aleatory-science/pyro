import sys
import numpy as np
import ctypes as ct
# Stub code for OpenCL setup.

import pyopencl as cl
import numpy as np
import sys

if cl.version.VERSION < (2015,2):
    raise Exception('Futhark requires at least PyOpenCL version 2015.2.  Installed version is %s.' %
                    cl.version.VERSION_TEXT)

def parse_preferred_device(s):
    pref_num = 0
    if len(s) > 1 and s[0] == '#':
        i = 1
        while i < len(s):
            if not s[i].isdigit():
                break
            else:
                pref_num = pref_num * 10 + int(s[i])
            i += 1
        while i < len(s) and s[i].isspace():
            i += 1
        return (s[i:], pref_num)
    else:
        return (s, 0)

def get_prefered_context(interactive=False, platform_pref=None, device_pref=None):
    if device_pref != None:
        (device_pref, device_num) = parse_preferred_device(device_pref)
    else:
        device_num = 0

    if interactive:
        return cl.create_some_context(interactive=True)

    def blacklisted(p, d):
        return platform_pref == None and device_pref == None and \
            p.name == "Apple" and d.name.find("Intel(R) Core(TM)") >= 0
    def platform_ok(p):
        return not platform_pref or p.name.find(platform_pref) >= 0
    def device_ok(d):
        return not device_pref or d.name.find(device_pref) >= 0

    device_matches = 0

    for p in cl.get_platforms():
        if not platform_ok(p):
            continue
        for d in p.get_devices():
            if blacklisted(p,d) or not device_ok(d):
                continue
            if device_matches == device_num:
                return cl.Context(devices=[d])
            else:
                device_matches += 1
    raise Exception('No OpenCL platform and device matching constraints found.')

def size_assignment(s):
    name, value = s.split('=')
    return (name, int(value))

def check_types(self, required_types):
    if 'f64' in required_types:
        if self.device.get_info(cl.device_info.PREFERRED_VECTOR_WIDTH_DOUBLE) == 0:
            raise Exception('Program uses double-precision floats, but this is not supported on chosen device: %s' % self.device.name)

def apply_size_heuristics(self, size_heuristics, sizes):
    for (platform_name, device_type, size, valuef) in size_heuristics:
        if sizes[size] == None \
           and self.platform.name.find(platform_name) >= 0 \
           and (self.device.type & device_type) == device_type:
               sizes[size] = valuef(self.device)
    return sizes

def initialise_opencl_object(self,
                             program_src='',
                             command_queue=None,
                             interactive=False,
                             platform_pref=None,
                             device_pref=None,
                             default_group_size=None,
                             default_num_groups=None,
                             default_tile_size=None,
                             default_reg_tile_size=None,
                             default_threshold=None,
                             size_heuristics=[],
                             required_types=[],
                             all_sizes={},
                             user_sizes={}):
    if command_queue is None:
        self.ctx = get_prefered_context(interactive, platform_pref, device_pref)
        self.queue = cl.CommandQueue(self.ctx)
    else:
        self.ctx = command_queue.context
        self.queue = command_queue
    self.device = self.queue.device
    self.platform = self.device.platform
    self.pool = cl.tools.MemoryPool(cl.tools.ImmediateAllocator(self.queue))
    device_type = self.device.type

    check_types(self, required_types)

    max_group_size = int(self.device.max_work_group_size)
    max_tile_size = int(np.sqrt(self.device.max_work_group_size))

    self.max_group_size = max_group_size
    self.max_tile_size = max_tile_size
    self.max_threshold = 0
    self.max_num_groups = 0

    self.max_local_memory = int(self.device.local_mem_size)

    # Futhark reserves 4 bytes of local memory for its own purposes.
    self.max_local_memory -= 4

    # See comment in rts/c/opencl.h.
    if self.platform.name.find('NVIDIA CUDA') >= 0:
        self.max_local_memory -= 12
    elif self.platform.name.find('AMD') >= 0:
        self.max_local_memory -= 16

    self.free_list = {}

    self.global_failure = self.pool.allocate(np.int32().itemsize)
    cl.enqueue_fill_buffer(self.queue, self.global_failure, np.int32(-1), 0, np.int32().itemsize)
    self.global_failure_args = self.pool.allocate(np.int64().itemsize *
                                                  (self.global_failure_args_max+1))
    self.failure_is_an_option = np.int32(0)

    if 'default_group_size' in sizes:
        default_group_size = sizes['default_group_size']
        del sizes['default_group_size']

    if 'default_num_groups' in sizes:
        default_num_groups = sizes['default_num_groups']
        del sizes['default_num_groups']

    if 'default_tile_size' in sizes:
        default_tile_size = sizes['default_tile_size']
        del sizes['default_tile_size']

    if 'default_reg_tile_size' in sizes:
        default_reg_tile_size = sizes['default_reg_tile_size']
        del sizes['default_reg_tile_size']

    if 'default_threshold' in sizes:
        default_threshold = sizes['default_threshold']
        del sizes['default_threshold']

    default_group_size_set = default_group_size != None
    default_tile_size_set = default_tile_size != None
    default_sizes = apply_size_heuristics(self, size_heuristics,
                                          {'group_size': default_group_size,
                                           'tile_size': default_tile_size,
                                           'reg_tile_size': default_reg_tile_size,
                                           'num_groups': default_num_groups,
                                           'lockstep_width': None,
                                           'threshold': default_threshold})
    default_group_size = default_sizes['group_size']
    default_num_groups = default_sizes['num_groups']
    default_threshold = default_sizes['threshold']
    default_tile_size = default_sizes['tile_size']
    default_reg_tile_size = default_sizes['reg_tile_size']
    lockstep_width = default_sizes['lockstep_width']

    if default_group_size > max_group_size:
        if default_group_size_set:
            sys.stderr.write('Note: Device limits group size to {} (down from {})\n'.
                             format(max_tile_size, default_group_size))
        default_group_size = max_group_size

    if default_tile_size > max_tile_size:
        if default_tile_size_set:
            sys.stderr.write('Note: Device limits tile size to {} (down from {})\n'.
                             format(max_tile_size, default_tile_size))
        default_tile_size = max_tile_size

    for (k,v) in user_sizes.items():
        if k in all_sizes:
            all_sizes[k]['value'] = v
        else:
            raise Exception('Unknown size: {}\nKnown sizes: {}'.format(k, ' '.join(all_sizes.keys())))

    self.sizes = {}
    for (k,v) in all_sizes.items():
        if v['class'] == 'group_size':
            max_value = max_group_size
            default_value = default_group_size
        elif v['class'] == 'num_groups':
            max_value = max_group_size # Intentional!
            default_value = default_num_groups
        elif v['class'] == 'tile_size':
            max_value = max_tile_size
            default_value = default_tile_size
        elif v['class'] == 'reg_tile_size':
            max_value = None
            default_value = default_reg_tile_size
        elif v['class'].startswith('threshold'):
            max_value = None
            default_value = default_threshold
        else:
            # Bespoke sizes have no limit or default.
            max_value = None
        if v['value'] == None:
            self.sizes[k] = default_value
        elif max_value != None and v['value'] > max_value:
            sys.stderr.write('Note: Device limits {} to {} (down from {}\n'.
                             format(k, max_value, v['value']))
            self.sizes[k] = max_value
        else:
            self.sizes[k] = v['value']

    # XXX: we perform only a subset of z-encoding here.  Really, the
    # compiler should provide us with the variables to which
    # parameters are mapped.
    if (len(program_src) >= 0):
        return cl.Program(self.ctx, program_src).build(
            ["-DLOCKSTEP_WIDTH={}".format(lockstep_width)]
            + ["-D{}={}".format(s.replace('z', 'zz').replace('.', 'zi').replace('#', 'zh'),v) for (s,v) in self.sizes.items()])

def opencl_alloc(self, min_size, tag):
    min_size = 1 if min_size == 0 else min_size
    assert min_size > 0
    return self.pool.allocate(min_size)

def opencl_free_all(self):
    self.pool.free_held()

def sync(self):
    failure = np.empty(1, dtype=np.int32)
    cl.enqueue_copy(self.queue, failure, self.global_failure, is_blocking=True)
    self.failure_is_an_option = np.int32(0)
    if failure[0] >= 0:
        # Reset failure information.
        cl.enqueue_fill_buffer(self.queue, self.global_failure, np.int32(-1), 0, np.int32().itemsize)

        # Read failure args.
        failure_args = np.empty(self.global_failure_args_max+1, dtype=np.int64)
        cl.enqueue_copy(self.queue, failure_args, self.global_failure_args, is_blocking=True)

        raise Exception(self.failure_msgs[failure[0]].format(*failure_args))
import pyopencl.array
import time
import argparse
sizes = {}
synchronous = False
preferred_platform = None
preferred_device = None
default_threshold = None
default_group_size = None
default_num_groups = None
default_tile_size = None
default_reg_tile_size = None
fut_opencl_src = """#ifdef cl_clang_storage_class_specifiers
#pragma OPENCL EXTENSION cl_clang_storage_class_specifiers : enable
#endif
#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable
__kernel void dummy_kernel(__global unsigned char *dummy, int n)
{
    const int thread_gid = get_global_id(0);
    
    if (thread_gid >= n)
        return;
}
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_int64_extended_atomics : enable
typedef char int8_t;
typedef short int16_t;
typedef int int32_t;
typedef long int64_t;
typedef uchar uint8_t;
typedef ushort uint16_t;
typedef uint uint32_t;
typedef ulong uint64_t;
#ifdef cl_nv_pragma_unroll
static inline void mem_fence_global()
{
    asm("membar.gl;");
}
#else
static inline void mem_fence_global()
{
    mem_fence(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
}
#endif
static inline void mem_fence_local()
{
    mem_fence(CLK_LOCAL_MEM_FENCE);
}
static inline uint8_t add8(uint8_t x, uint8_t y)
{
    return x + y;
}
static inline uint16_t add16(uint16_t x, uint16_t y)
{
    return x + y;
}
static inline uint32_t add32(uint32_t x, uint32_t y)
{
    return x + y;
}
static inline uint64_t add64(uint64_t x, uint64_t y)
{
    return x + y;
}
static inline uint8_t sub8(uint8_t x, uint8_t y)
{
    return x - y;
}
static inline uint16_t sub16(uint16_t x, uint16_t y)
{
    return x - y;
}
static inline uint32_t sub32(uint32_t x, uint32_t y)
{
    return x - y;
}
static inline uint64_t sub64(uint64_t x, uint64_t y)
{
    return x - y;
}
static inline uint8_t mul8(uint8_t x, uint8_t y)
{
    return x * y;
}
static inline uint16_t mul16(uint16_t x, uint16_t y)
{
    return x * y;
}
static inline uint32_t mul32(uint32_t x, uint32_t y)
{
    return x * y;
}
static inline uint64_t mul64(uint64_t x, uint64_t y)
{
    return x * y;
}
static inline uint8_t udiv8(uint8_t x, uint8_t y)
{
    return x / y;
}
static inline uint16_t udiv16(uint16_t x, uint16_t y)
{
    return x / y;
}
static inline uint32_t udiv32(uint32_t x, uint32_t y)
{
    return x / y;
}
static inline uint64_t udiv64(uint64_t x, uint64_t y)
{
    return x / y;
}
static inline uint8_t udiv_up8(uint8_t x, uint8_t y)
{
    return (x + y - 1) / y;
}
static inline uint16_t udiv_up16(uint16_t x, uint16_t y)
{
    return (x + y - 1) / y;
}
static inline uint32_t udiv_up32(uint32_t x, uint32_t y)
{
    return (x + y - 1) / y;
}
static inline uint64_t udiv_up64(uint64_t x, uint64_t y)
{
    return (x + y - 1) / y;
}
static inline uint8_t umod8(uint8_t x, uint8_t y)
{
    return x % y;
}
static inline uint16_t umod16(uint16_t x, uint16_t y)
{
    return x % y;
}
static inline uint32_t umod32(uint32_t x, uint32_t y)
{
    return x % y;
}
static inline uint64_t umod64(uint64_t x, uint64_t y)
{
    return x % y;
}
static inline uint8_t udiv_safe8(uint8_t x, uint8_t y)
{
    return y == 0 ? 0 : x / y;
}
static inline uint16_t udiv_safe16(uint16_t x, uint16_t y)
{
    return y == 0 ? 0 : x / y;
}
static inline uint32_t udiv_safe32(uint32_t x, uint32_t y)
{
    return y == 0 ? 0 : x / y;
}
static inline uint64_t udiv_safe64(uint64_t x, uint64_t y)
{
    return y == 0 ? 0 : x / y;
}
static inline uint8_t udiv_up_safe8(uint8_t x, uint8_t y)
{
    return y == 0 ? 0 : (x + y - 1) / y;
}
static inline uint16_t udiv_up_safe16(uint16_t x, uint16_t y)
{
    return y == 0 ? 0 : (x + y - 1) / y;
}
static inline uint32_t udiv_up_safe32(uint32_t x, uint32_t y)
{
    return y == 0 ? 0 : (x + y - 1) / y;
}
static inline uint64_t udiv_up_safe64(uint64_t x, uint64_t y)
{
    return y == 0 ? 0 : (x + y - 1) / y;
}
static inline uint8_t umod_safe8(uint8_t x, uint8_t y)
{
    return y == 0 ? 0 : x % y;
}
static inline uint16_t umod_safe16(uint16_t x, uint16_t y)
{
    return y == 0 ? 0 : x % y;
}
static inline uint32_t umod_safe32(uint32_t x, uint32_t y)
{
    return y == 0 ? 0 : x % y;
}
static inline uint64_t umod_safe64(uint64_t x, uint64_t y)
{
    return y == 0 ? 0 : x % y;
}
static inline int8_t sdiv8(int8_t x, int8_t y)
{
    int8_t q = x / y;
    int8_t r = x % y;
    
    return q - ((r != 0 && r < 0 != y < 0) ? 1 : 0);
}
static inline int16_t sdiv16(int16_t x, int16_t y)
{
    int16_t q = x / y;
    int16_t r = x % y;
    
    return q - ((r != 0 && r < 0 != y < 0) ? 1 : 0);
}
static inline int32_t sdiv32(int32_t x, int32_t y)
{
    int32_t q = x / y;
    int32_t r = x % y;
    
    return q - ((r != 0 && r < 0 != y < 0) ? 1 : 0);
}
static inline int64_t sdiv64(int64_t x, int64_t y)
{
    int64_t q = x / y;
    int64_t r = x % y;
    
    return q - ((r != 0 && r < 0 != y < 0) ? 1 : 0);
}
static inline int8_t sdiv_up8(int8_t x, int8_t y)
{
    return sdiv8(x + y - 1, y);
}
static inline int16_t sdiv_up16(int16_t x, int16_t y)
{
    return sdiv16(x + y - 1, y);
}
static inline int32_t sdiv_up32(int32_t x, int32_t y)
{
    return sdiv32(x + y - 1, y);
}
static inline int64_t sdiv_up64(int64_t x, int64_t y)
{
    return sdiv64(x + y - 1, y);
}
static inline int8_t smod8(int8_t x, int8_t y)
{
    int8_t r = x % y;
    
    return r + (r == 0 || (x > 0 && y > 0) || (x < 0 && y < 0) ? 0 : y);
}
static inline int16_t smod16(int16_t x, int16_t y)
{
    int16_t r = x % y;
    
    return r + (r == 0 || (x > 0 && y > 0) || (x < 0 && y < 0) ? 0 : y);
}
static inline int32_t smod32(int32_t x, int32_t y)
{
    int32_t r = x % y;
    
    return r + (r == 0 || (x > 0 && y > 0) || (x < 0 && y < 0) ? 0 : y);
}
static inline int64_t smod64(int64_t x, int64_t y)
{
    int64_t r = x % y;
    
    return r + (r == 0 || (x > 0 && y > 0) || (x < 0 && y < 0) ? 0 : y);
}
static inline int8_t sdiv_safe8(int8_t x, int8_t y)
{
    return y == 0 ? 0 : sdiv8(x, y);
}
static inline int16_t sdiv_safe16(int16_t x, int16_t y)
{
    return y == 0 ? 0 : sdiv16(x, y);
}
static inline int32_t sdiv_safe32(int32_t x, int32_t y)
{
    return y == 0 ? 0 : sdiv32(x, y);
}
static inline int64_t sdiv_safe64(int64_t x, int64_t y)
{
    return y == 0 ? 0 : sdiv64(x, y);
}
static inline int8_t sdiv_up_safe8(int8_t x, int8_t y)
{
    return sdiv_safe8(x + y - 1, y);
}
static inline int16_t sdiv_up_safe16(int16_t x, int16_t y)
{
    return sdiv_safe16(x + y - 1, y);
}
static inline int32_t sdiv_up_safe32(int32_t x, int32_t y)
{
    return sdiv_safe32(x + y - 1, y);
}
static inline int64_t sdiv_up_safe64(int64_t x, int64_t y)
{
    return sdiv_safe64(x + y - 1, y);
}
static inline int8_t smod_safe8(int8_t x, int8_t y)
{
    return y == 0 ? 0 : smod8(x, y);
}
static inline int16_t smod_safe16(int16_t x, int16_t y)
{
    return y == 0 ? 0 : smod16(x, y);
}
static inline int32_t smod_safe32(int32_t x, int32_t y)
{
    return y == 0 ? 0 : smod32(x, y);
}
static inline int64_t smod_safe64(int64_t x, int64_t y)
{
    return y == 0 ? 0 : smod64(x, y);
}
static inline int8_t squot8(int8_t x, int8_t y)
{
    return x / y;
}
static inline int16_t squot16(int16_t x, int16_t y)
{
    return x / y;
}
static inline int32_t squot32(int32_t x, int32_t y)
{
    return x / y;
}
static inline int64_t squot64(int64_t x, int64_t y)
{
    return x / y;
}
static inline int8_t srem8(int8_t x, int8_t y)
{
    return x % y;
}
static inline int16_t srem16(int16_t x, int16_t y)
{
    return x % y;
}
static inline int32_t srem32(int32_t x, int32_t y)
{
    return x % y;
}
static inline int64_t srem64(int64_t x, int64_t y)
{
    return x % y;
}
static inline int8_t squot_safe8(int8_t x, int8_t y)
{
    return y == 0 ? 0 : x / y;
}
static inline int16_t squot_safe16(int16_t x, int16_t y)
{
    return y == 0 ? 0 : x / y;
}
static inline int32_t squot_safe32(int32_t x, int32_t y)
{
    return y == 0 ? 0 : x / y;
}
static inline int64_t squot_safe64(int64_t x, int64_t y)
{
    return y == 0 ? 0 : x / y;
}
static inline int8_t srem_safe8(int8_t x, int8_t y)
{
    return y == 0 ? 0 : x % y;
}
static inline int16_t srem_safe16(int16_t x, int16_t y)
{
    return y == 0 ? 0 : x % y;
}
static inline int32_t srem_safe32(int32_t x, int32_t y)
{
    return y == 0 ? 0 : x % y;
}
static inline int64_t srem_safe64(int64_t x, int64_t y)
{
    return y == 0 ? 0 : x % y;
}
static inline int8_t smin8(int8_t x, int8_t y)
{
    return x < y ? x : y;
}
static inline int16_t smin16(int16_t x, int16_t y)
{
    return x < y ? x : y;
}
static inline int32_t smin32(int32_t x, int32_t y)
{
    return x < y ? x : y;
}
static inline int64_t smin64(int64_t x, int64_t y)
{
    return x < y ? x : y;
}
static inline uint8_t umin8(uint8_t x, uint8_t y)
{
    return x < y ? x : y;
}
static inline uint16_t umin16(uint16_t x, uint16_t y)
{
    return x < y ? x : y;
}
static inline uint32_t umin32(uint32_t x, uint32_t y)
{
    return x < y ? x : y;
}
static inline uint64_t umin64(uint64_t x, uint64_t y)
{
    return x < y ? x : y;
}
static inline int8_t smax8(int8_t x, int8_t y)
{
    return x < y ? y : x;
}
static inline int16_t smax16(int16_t x, int16_t y)
{
    return x < y ? y : x;
}
static inline int32_t smax32(int32_t x, int32_t y)
{
    return x < y ? y : x;
}
static inline int64_t smax64(int64_t x, int64_t y)
{
    return x < y ? y : x;
}
static inline uint8_t umax8(uint8_t x, uint8_t y)
{
    return x < y ? y : x;
}
static inline uint16_t umax16(uint16_t x, uint16_t y)
{
    return x < y ? y : x;
}
static inline uint32_t umax32(uint32_t x, uint32_t y)
{
    return x < y ? y : x;
}
static inline uint64_t umax64(uint64_t x, uint64_t y)
{
    return x < y ? y : x;
}
static inline uint8_t shl8(uint8_t x, uint8_t y)
{
    return x << y;
}
static inline uint16_t shl16(uint16_t x, uint16_t y)
{
    return x << y;
}
static inline uint32_t shl32(uint32_t x, uint32_t y)
{
    return x << y;
}
static inline uint64_t shl64(uint64_t x, uint64_t y)
{
    return x << y;
}
static inline uint8_t lshr8(uint8_t x, uint8_t y)
{
    return x >> y;
}
static inline uint16_t lshr16(uint16_t x, uint16_t y)
{
    return x >> y;
}
static inline uint32_t lshr32(uint32_t x, uint32_t y)
{
    return x >> y;
}
static inline uint64_t lshr64(uint64_t x, uint64_t y)
{
    return x >> y;
}
static inline int8_t ashr8(int8_t x, int8_t y)
{
    return x >> y;
}
static inline int16_t ashr16(int16_t x, int16_t y)
{
    return x >> y;
}
static inline int32_t ashr32(int32_t x, int32_t y)
{
    return x >> y;
}
static inline int64_t ashr64(int64_t x, int64_t y)
{
    return x >> y;
}
static inline uint8_t and8(uint8_t x, uint8_t y)
{
    return x & y;
}
static inline uint16_t and16(uint16_t x, uint16_t y)
{
    return x & y;
}
static inline uint32_t and32(uint32_t x, uint32_t y)
{
    return x & y;
}
static inline uint64_t and64(uint64_t x, uint64_t y)
{
    return x & y;
}
static inline uint8_t or8(uint8_t x, uint8_t y)
{
    return x | y;
}
static inline uint16_t or16(uint16_t x, uint16_t y)
{
    return x | y;
}
static inline uint32_t or32(uint32_t x, uint32_t y)
{
    return x | y;
}
static inline uint64_t or64(uint64_t x, uint64_t y)
{
    return x | y;
}
static inline uint8_t xor8(uint8_t x, uint8_t y)
{
    return x ^ y;
}
static inline uint16_t xor16(uint16_t x, uint16_t y)
{
    return x ^ y;
}
static inline uint32_t xor32(uint32_t x, uint32_t y)
{
    return x ^ y;
}
static inline uint64_t xor64(uint64_t x, uint64_t y)
{
    return x ^ y;
}
static inline bool ult8(uint8_t x, uint8_t y)
{
    return x < y;
}
static inline bool ult16(uint16_t x, uint16_t y)
{
    return x < y;
}
static inline bool ult32(uint32_t x, uint32_t y)
{
    return x < y;
}
static inline bool ult64(uint64_t x, uint64_t y)
{
    return x < y;
}
static inline bool ule8(uint8_t x, uint8_t y)
{
    return x <= y;
}
static inline bool ule16(uint16_t x, uint16_t y)
{
    return x <= y;
}
static inline bool ule32(uint32_t x, uint32_t y)
{
    return x <= y;
}
static inline bool ule64(uint64_t x, uint64_t y)
{
    return x <= y;
}
static inline bool slt8(int8_t x, int8_t y)
{
    return x < y;
}
static inline bool slt16(int16_t x, int16_t y)
{
    return x < y;
}
static inline bool slt32(int32_t x, int32_t y)
{
    return x < y;
}
static inline bool slt64(int64_t x, int64_t y)
{
    return x < y;
}
static inline bool sle8(int8_t x, int8_t y)
{
    return x <= y;
}
static inline bool sle16(int16_t x, int16_t y)
{
    return x <= y;
}
static inline bool sle32(int32_t x, int32_t y)
{
    return x <= y;
}
static inline bool sle64(int64_t x, int64_t y)
{
    return x <= y;
}
static inline int8_t pow8(int8_t x, int8_t y)
{
    int8_t res = 1, rem = y;
    
    while (rem != 0) {
        if (rem & 1)
            res *= x;
        rem >>= 1;
        x *= x;
    }
    return res;
}
static inline int16_t pow16(int16_t x, int16_t y)
{
    int16_t res = 1, rem = y;
    
    while (rem != 0) {
        if (rem & 1)
            res *= x;
        rem >>= 1;
        x *= x;
    }
    return res;
}
static inline int32_t pow32(int32_t x, int32_t y)
{
    int32_t res = 1, rem = y;
    
    while (rem != 0) {
        if (rem & 1)
            res *= x;
        rem >>= 1;
        x *= x;
    }
    return res;
}
static inline int64_t pow64(int64_t x, int64_t y)
{
    int64_t res = 1, rem = y;
    
    while (rem != 0) {
        if (rem & 1)
            res *= x;
        rem >>= 1;
        x *= x;
    }
    return res;
}
static inline bool itob_i8_bool(int8_t x)
{
    return x;
}
static inline bool itob_i16_bool(int16_t x)
{
    return x;
}
static inline bool itob_i32_bool(int32_t x)
{
    return x;
}
static inline bool itob_i64_bool(int64_t x)
{
    return x;
}
static inline int8_t btoi_bool_i8(bool x)
{
    return x;
}
static inline int16_t btoi_bool_i16(bool x)
{
    return x;
}
static inline int32_t btoi_bool_i32(bool x)
{
    return x;
}
static inline int64_t btoi_bool_i64(bool x)
{
    return x;
}
#define sext_i8_i8(x) ((int8_t) (int8_t) x)
#define sext_i8_i16(x) ((int16_t) (int8_t) x)
#define sext_i8_i32(x) ((int32_t) (int8_t) x)
#define sext_i8_i64(x) ((int64_t) (int8_t) x)
#define sext_i16_i8(x) ((int8_t) (int16_t) x)
#define sext_i16_i16(x) ((int16_t) (int16_t) x)
#define sext_i16_i32(x) ((int32_t) (int16_t) x)
#define sext_i16_i64(x) ((int64_t) (int16_t) x)
#define sext_i32_i8(x) ((int8_t) (int32_t) x)
#define sext_i32_i16(x) ((int16_t) (int32_t) x)
#define sext_i32_i32(x) ((int32_t) (int32_t) x)
#define sext_i32_i64(x) ((int64_t) (int32_t) x)
#define sext_i64_i8(x) ((int8_t) (int64_t) x)
#define sext_i64_i16(x) ((int16_t) (int64_t) x)
#define sext_i64_i32(x) ((int32_t) (int64_t) x)
#define sext_i64_i64(x) ((int64_t) (int64_t) x)
#define zext_i8_i8(x) ((int8_t) (uint8_t) x)
#define zext_i8_i16(x) ((int16_t) (uint8_t) x)
#define zext_i8_i32(x) ((int32_t) (uint8_t) x)
#define zext_i8_i64(x) ((int64_t) (uint8_t) x)
#define zext_i16_i8(x) ((int8_t) (uint16_t) x)
#define zext_i16_i16(x) ((int16_t) (uint16_t) x)
#define zext_i16_i32(x) ((int32_t) (uint16_t) x)
#define zext_i16_i64(x) ((int64_t) (uint16_t) x)
#define zext_i32_i8(x) ((int8_t) (uint32_t) x)
#define zext_i32_i16(x) ((int16_t) (uint32_t) x)
#define zext_i32_i32(x) ((int32_t) (uint32_t) x)
#define zext_i32_i64(x) ((int64_t) (uint32_t) x)
#define zext_i64_i8(x) ((int8_t) (uint64_t) x)
#define zext_i64_i16(x) ((int16_t) (uint64_t) x)
#define zext_i64_i32(x) ((int32_t) (uint64_t) x)
#define zext_i64_i64(x) ((int64_t) (uint64_t) x)
#if defined(__OPENCL_VERSION__)
static int32_t futrts_popc8(int8_t x)
{
    return popcount(x);
}
static int32_t futrts_popc16(int16_t x)
{
    return popcount(x);
}
static int32_t futrts_popc32(int32_t x)
{
    return popcount(x);
}
static int32_t futrts_popc64(int64_t x)
{
    return popcount(x);
}
#elif defined(__CUDA_ARCH__)
static int32_t futrts_popc8(int8_t x)
{
    return __popc(zext_i8_i32(x));
}
static int32_t futrts_popc16(int16_t x)
{
    return __popc(zext_i16_i32(x));
}
static int32_t futrts_popc32(int32_t x)
{
    return __popc(x);
}
static int32_t futrts_popc64(int64_t x)
{
    return __popcll(x);
}
#else
static int32_t futrts_popc8(int8_t x)
{
    int c = 0;
    
    for (; x; ++c)
        x &= x - 1;
    return c;
}
static int32_t futrts_popc16(int16_t x)
{
    int c = 0;
    
    for (; x; ++c)
        x &= x - 1;
    return c;
}
static int32_t futrts_popc32(int32_t x)
{
    int c = 0;
    
    for (; x; ++c)
        x &= x - 1;
    return c;
}
static int32_t futrts_popc64(int64_t x)
{
    int c = 0;
    
    for (; x; ++c)
        x &= x - 1;
    return c;
}
#endif
#if defined(__OPENCL_VERSION__)
static uint8_t futrts_mul_hi8(uint8_t a, uint8_t b)
{
    return mul_hi(a, b);
}
static uint16_t futrts_mul_hi16(uint16_t a, uint16_t b)
{
    return mul_hi(a, b);
}
static uint32_t futrts_mul_hi32(uint32_t a, uint32_t b)
{
    return mul_hi(a, b);
}
static uint64_t futrts_mul_hi64(uint64_t a, uint64_t b)
{
    return mul_hi(a, b);
}
#elif defined(__CUDA_ARCH__)
static uint8_t futrts_mul_hi8(uint8_t a, uint8_t b)
{
    uint16_t aa = a;
    uint16_t bb = b;
    
    return aa * bb >> 8;
}
static uint16_t futrts_mul_hi16(uint16_t a, uint16_t b)
{
    uint32_t aa = a;
    uint32_t bb = b;
    
    return aa * bb >> 16;
}
static uint32_t futrts_mul_hi32(uint32_t a, uint32_t b)
{
    return mulhi(a, b);
}
static uint64_t futrts_mul_hi64(uint64_t a, uint64_t b)
{
    return mul64hi(a, b);
}
#else
static uint8_t futrts_mul_hi8(uint8_t a, uint8_t b)
{
    uint16_t aa = a;
    uint16_t bb = b;
    
    return aa * bb >> 8;
}
static uint16_t futrts_mul_hi16(uint16_t a, uint16_t b)
{
    uint32_t aa = a;
    uint32_t bb = b;
    
    return aa * bb >> 16;
}
static uint32_t futrts_mul_hi32(uint32_t a, uint32_t b)
{
    uint64_t aa = a;
    uint64_t bb = b;
    
    return aa * bb >> 32;
}
static uint64_t futrts_mul_hi64(uint64_t a, uint64_t b)
{
    __uint128_t aa = a;
    __uint128_t bb = b;
    
    return aa * bb >> 64;
}
#endif
#if defined(__OPENCL_VERSION__)
static uint8_t futrts_mad_hi8(uint8_t a, uint8_t b, uint8_t c)
{
    return mad_hi(a, b, c);
}
static uint16_t futrts_mad_hi16(uint16_t a, uint16_t b, uint16_t c)
{
    return mad_hi(a, b, c);
}
static uint32_t futrts_mad_hi32(uint32_t a, uint32_t b, uint32_t c)
{
    return mad_hi(a, b, c);
}
static uint64_t futrts_mad_hi64(uint64_t a, uint64_t b, uint64_t c)
{
    return mad_hi(a, b, c);
}
#else
static uint8_t futrts_mad_hi8(uint8_t a, uint8_t b, uint8_t c)
{
    return futrts_mul_hi8(a, b) + c;
}
static uint16_t futrts_mad_hi16(uint16_t a, uint16_t b, uint16_t c)
{
    return futrts_mul_hi16(a, b) + c;
}
static uint32_t futrts_mad_hi32(uint32_t a, uint32_t b, uint32_t c)
{
    return futrts_mul_hi32(a, b) + c;
}
static uint64_t futrts_mad_hi64(uint64_t a, uint64_t b, uint64_t c)
{
    return futrts_mul_hi64(a, b) + c;
}
#endif
#if defined(__OPENCL_VERSION__)
static int32_t futrts_clzz8(int8_t x)
{
    return clz(x);
}
static int32_t futrts_clzz16(int16_t x)
{
    return clz(x);
}
static int32_t futrts_clzz32(int32_t x)
{
    return clz(x);
}
static int32_t futrts_clzz64(int64_t x)
{
    return clz(x);
}
#elif defined(__CUDA_ARCH__)
static int32_t futrts_clzz8(int8_t x)
{
    return __clz(zext_i8_i32(x)) - 24;
}
static int32_t futrts_clzz16(int16_t x)
{
    return __clz(zext_i16_i32(x)) - 16;
}
static int32_t futrts_clzz32(int32_t x)
{
    return __clz(x);
}
static int32_t futrts_clzz64(int64_t x)
{
    return __clzll(x);
}
#else
static int32_t futrts_clzz8(int8_t x)
{
    int n = 0;
    int bits = sizeof(x) * 8;
    
    for (int i = 0; i < bits; i++) {
        if (x < 0)
            break;
        n++;
        x <<= 1;
    }
    return n;
}
static int32_t futrts_clzz16(int16_t x)
{
    int n = 0;
    int bits = sizeof(x) * 8;
    
    for (int i = 0; i < bits; i++) {
        if (x < 0)
            break;
        n++;
        x <<= 1;
    }
    return n;
}
static int32_t futrts_clzz32(int32_t x)
{
    int n = 0;
    int bits = sizeof(x) * 8;
    
    for (int i = 0; i < bits; i++) {
        if (x < 0)
            break;
        n++;
        x <<= 1;
    }
    return n;
}
static int32_t futrts_clzz64(int64_t x)
{
    int n = 0;
    int bits = sizeof(x) * 8;
    
    for (int i = 0; i < bits; i++) {
        if (x < 0)
            break;
        n++;
        x <<= 1;
    }
    return n;
}
#endif
#if defined(__OPENCL_VERSION__)
static int32_t futrts_ctzz8(int8_t x)
{
    int i = 0;
    
    for (; i < 8 && (x & 1) == 0; i++, x >>= 1)
        ;
    return i;
}
static int32_t futrts_ctzz16(int16_t x)
{
    int i = 0;
    
    for (; i < 16 && (x & 1) == 0; i++, x >>= 1)
        ;
    return i;
}
static int32_t futrts_ctzz32(int32_t x)
{
    int i = 0;
    
    for (; i < 32 && (x & 1) == 0; i++, x >>= 1)
        ;
    return i;
}
static int32_t futrts_ctzz64(int64_t x)
{
    int i = 0;
    
    for (; i < 64 && (x & 1) == 0; i++, x >>= 1)
        ;
    return i;
}
#elif defined(__CUDA_ARCH__)
static int32_t futrts_ctzz8(int8_t x)
{
    int y = __ffs(x);
    
    return y == 0 ? 8 : y - 1;
}
static int32_t futrts_ctzz16(int16_t x)
{
    int y = __ffs(x);
    
    return y == 0 ? 16 : y - 1;
}
static int32_t futrts_ctzz32(int32_t x)
{
    int y = __ffs(x);
    
    return y == 0 ? 32 : y - 1;
}
static int32_t futrts_ctzz64(int64_t x)
{
    int y = __ffsll(x);
    
    return y == 0 ? 64 : y - 1;
}
#else
static int32_t futrts_ctzz8(int8_t x)
{
    return x == 0 ? 8 : __builtin_ctz((uint32_t) x);
}
static int32_t futrts_ctzz16(int16_t x)
{
    return x == 0 ? 16 : __builtin_ctz((uint32_t) x);
}
static int32_t futrts_ctzz32(int32_t x)
{
    return x == 0 ? 32 : __builtin_ctz(x);
}
static int32_t futrts_ctzz64(int64_t x)
{
    return x == 0 ? 64 : __builtin_ctzll(x);
}
#endif
static inline float fdiv32(float x, float y)
{
    return x / y;
}
static inline float fadd32(float x, float y)
{
    return x + y;
}
static inline float fsub32(float x, float y)
{
    return x - y;
}
static inline float fmul32(float x, float y)
{
    return x * y;
}
static inline float fmin32(float x, float y)
{
    return fmin(x, y);
}
static inline float fmax32(float x, float y)
{
    return fmax(x, y);
}
static inline float fpow32(float x, float y)
{
    return pow(x, y);
}
static inline bool cmplt32(float x, float y)
{
    return x < y;
}
static inline bool cmple32(float x, float y)
{
    return x <= y;
}
static inline float sitofp_i8_f32(int8_t x)
{
    return (float) x;
}
static inline float sitofp_i16_f32(int16_t x)
{
    return (float) x;
}
static inline float sitofp_i32_f32(int32_t x)
{
    return (float) x;
}
static inline float sitofp_i64_f32(int64_t x)
{
    return (float) x;
}
static inline float uitofp_i8_f32(uint8_t x)
{
    return (float) x;
}
static inline float uitofp_i16_f32(uint16_t x)
{
    return (float) x;
}
static inline float uitofp_i32_f32(uint32_t x)
{
    return (float) x;
}
static inline float uitofp_i64_f32(uint64_t x)
{
    return (float) x;
}
static inline int8_t fptosi_f32_i8(float x)
{
    return (int8_t) x;
}
static inline int16_t fptosi_f32_i16(float x)
{
    return (int16_t) x;
}
static inline int32_t fptosi_f32_i32(float x)
{
    return (int32_t) x;
}
static inline int64_t fptosi_f32_i64(float x)
{
    return (int64_t) x;
}
static inline uint8_t fptoui_f32_i8(float x)
{
    return (uint8_t) x;
}
static inline uint16_t fptoui_f32_i16(float x)
{
    return (uint16_t) x;
}
static inline uint32_t fptoui_f32_i32(float x)
{
    return (uint32_t) x;
}
static inline uint64_t fptoui_f32_i64(float x)
{
    return (uint64_t) x;
}
static inline bool futrts_isnan32(float x)
{
    return isnan(x);
}
static inline bool futrts_isinf32(float x)
{
    return isinf(x);
}
#ifdef __OPENCL_VERSION__
static inline float futrts_log32(float x)
{
    return log(x);
}
static inline float futrts_log2_32(float x)
{
    return log2(x);
}
static inline float futrts_log10_32(float x)
{
    return log10(x);
}
static inline float futrts_sqrt32(float x)
{
    return sqrt(x);
}
static inline float futrts_exp32(float x)
{
    return exp(x);
}
static inline float futrts_cos32(float x)
{
    return cos(x);
}
static inline float futrts_sin32(float x)
{
    return sin(x);
}
static inline float futrts_tan32(float x)
{
    return tan(x);
}
static inline float futrts_acos32(float x)
{
    return acos(x);
}
static inline float futrts_asin32(float x)
{
    return asin(x);
}
static inline float futrts_atan32(float x)
{
    return atan(x);
}
static inline float futrts_cosh32(float x)
{
    return cosh(x);
}
static inline float futrts_sinh32(float x)
{
    return sinh(x);
}
static inline float futrts_tanh32(float x)
{
    return tanh(x);
}
static inline float futrts_acosh32(float x)
{
    return acosh(x);
}
static inline float futrts_asinh32(float x)
{
    return asinh(x);
}
static inline float futrts_atanh32(float x)
{
    return atanh(x);
}
static inline float futrts_atan2_32(float x, float y)
{
    return atan2(x, y);
}
static inline float futrts_gamma32(float x)
{
    return tgamma(x);
}
static inline float futrts_lgamma32(float x)
{
    return lgamma(x);
}
static inline float fmod32(float x, float y)
{
    return fmod(x, y);
}
static inline float futrts_round32(float x)
{
    return rint(x);
}
static inline float futrts_floor32(float x)
{
    return floor(x);
}
static inline float futrts_ceil32(float x)
{
    return ceil(x);
}
static inline float futrts_lerp32(float v0, float v1, float t)
{
    return mix(v0, v1, t);
}
static inline float futrts_mad32(float a, float b, float c)
{
    return mad(a, b, c);
}
static inline float futrts_fma32(float a, float b, float c)
{
    return fma(a, b, c);
}
#else
static inline float futrts_log32(float x)
{
    return logf(x);
}
static inline float futrts_log2_32(float x)
{
    return log2f(x);
}
static inline float futrts_log10_32(float x)
{
    return log10f(x);
}
static inline float futrts_sqrt32(float x)
{
    return sqrtf(x);
}
static inline float futrts_exp32(float x)
{
    return expf(x);
}
static inline float futrts_cos32(float x)
{
    return cosf(x);
}
static inline float futrts_sin32(float x)
{
    return sinf(x);
}
static inline float futrts_tan32(float x)
{
    return tanf(x);
}
static inline float futrts_acos32(float x)
{
    return acosf(x);
}
static inline float futrts_asin32(float x)
{
    return asinf(x);
}
static inline float futrts_atan32(float x)
{
    return atanf(x);
}
static inline float futrts_cosh32(float x)
{
    return coshf(x);
}
static inline float futrts_sinh32(float x)
{
    return sinhf(x);
}
static inline float futrts_tanh32(float x)
{
    return tanhf(x);
}
static inline float futrts_acosh32(float x)
{
    return acoshf(x);
}
static inline float futrts_asinh32(float x)
{
    return asinhf(x);
}
static inline float futrts_atanh32(float x)
{
    return atanhf(x);
}
static inline float futrts_atan2_32(float x, float y)
{
    return atan2f(x, y);
}
static inline float futrts_gamma32(float x)
{
    return tgammaf(x);
}
static inline float futrts_lgamma32(float x)
{
    return lgammaf(x);
}
static inline float fmod32(float x, float y)
{
    return fmodf(x, y);
}
static inline float futrts_round32(float x)
{
    return rintf(x);
}
static inline float futrts_floor32(float x)
{
    return floorf(x);
}
static inline float futrts_ceil32(float x)
{
    return ceilf(x);
}
static inline float futrts_lerp32(float v0, float v1, float t)
{
    return v0 + (v1 - v0) * t;
}
static inline float futrts_mad32(float a, float b, float c)
{
    return a * b + c;
}
static inline float futrts_fma32(float a, float b, float c)
{
    return fmaf(a, b, c);
}
#endif
static inline int32_t futrts_to_bits32(float x)
{
    union {
        float f;
        int32_t t;
    } p;
    
    p.f = x;
    return p.t;
}
static inline float futrts_from_bits32(int32_t x)
{
    union {
        int32_t f;
        float t;
    } p;
    
    p.f = x;
    return p.t;
}
static inline float fsignum32(float x)
{
    return futrts_isnan32(x) ? x : (x > 0) - (x < 0);
}
// Start of atomics.h

inline int32_t atomic_xchg_i32_global(volatile __global int32_t *p, int32_t x) {
#ifdef FUTHARK_CUDA
  return atomicExch((int32_t*)p, x);
#else
  return atomic_xor(p, x);
#endif
}

inline int32_t atomic_xchg_i32_local(volatile __local int32_t *p, int32_t x) {
#ifdef FUTHARK_CUDA
  return atomicExch((int32_t*)p, x);
#else
  return atomic_xor(p, x);
#endif
}

inline int32_t atomic_cmpxchg_i32_global(volatile __global int32_t *p,
                                         int32_t cmp, int32_t val) {
#ifdef FUTHARK_CUDA
  return atomicCAS((int32_t*)p, cmp, val);
#else
  return atomic_cmpxchg(p, cmp, val);
#endif
}

inline int32_t atomic_cmpxchg_i32_local(volatile __local int32_t *p,
                                        int32_t cmp, int32_t val) {
#ifdef FUTHARK_CUDA
  return atomicCAS((int32_t*)p, cmp, val);
#else
  return atomic_cmpxchg(p, cmp, val);
#endif
}

inline int32_t atomic_add_i32_global(volatile __global int32_t *p, int32_t x) {
#ifdef FUTHARK_CUDA
  return atomicAdd((int32_t*)p, x);
#else
  return atomic_add(p, x);
#endif
}

inline int32_t atomic_add_i32_local(volatile __local int32_t *p, int32_t x) {
#ifdef FUTHARK_CUDA
  return atomicAdd((int32_t*)p, x);
#else
  return atomic_add(p, x);
#endif
}

inline float atomic_fadd_f32_global(volatile __global float *p, float x) {
#ifdef FUTHARK_CUDA
  return atomicAdd((float*)p, x);
#else
  union { int32_t i; float f; } old;
  union { int32_t i; float f; } assumed;
  old.f = *p;
  do {
    assumed.f = old.f;
    old.f = old.f + x;
    old.i = atomic_cmpxchg_i32_global((volatile __global int32_t*)p, assumed.i, old.i);
  } while (assumed.i != old.i);
  return old.f;
#endif
}

inline float atomic_fadd_f32_local(volatile __local float *p, float x) {
#ifdef FUTHARK_CUDA
  return atomicAdd((float*)p, x);
#else
  union { int32_t i; float f; } old;
  union { int32_t i; float f; } assumed;
  old.f = *p;
  do {
    assumed.f = old.f;
    old.f = old.f + x;
    old.i = atomic_cmpxchg_i32_local((volatile __local int32_t*)p, assumed.i, old.i);
  } while (assumed.i != old.i);
  return old.f;
#endif
}

inline int32_t atomic_smax_i32_global(volatile __global int32_t *p, int32_t x) {
#ifdef FUTHARK_CUDA
  return atomicMax((int32_t*)p, x);
#else
  return atomic_max(p, x);
#endif
}

inline int32_t atomic_smax_i32_local(volatile __local int32_t *p, int32_t x) {
#ifdef FUTHARK_CUDA
  return atomicMax((int32_t*)p, x);
#else
  return atomic_max(p, x);
#endif
}

inline int32_t atomic_smin_i32_global(volatile __global int32_t *p, int32_t x) {
#ifdef FUTHARK_CUDA
  return atomicMin((int32_t*)p, x);
#else
  return atomic_min(p, x);
#endif
}

inline int32_t atomic_smin_i32_local(volatile __local int32_t *p, int32_t x) {
#ifdef FUTHARK_CUDA
  return atomicMin((int32_t*)p, x);
#else
  return atomic_min(p, x);
#endif
}

inline uint32_t atomic_umax_i32_global(volatile __global uint32_t *p, uint32_t x) {
#ifdef FUTHARK_CUDA
  return atomicMax((uint32_t*)p, x);
#else
  return atomic_max(p, x);
#endif
}

inline uint32_t atomic_umax_i32_local(volatile __local uint32_t *p, uint32_t x) {
#ifdef FUTHARK_CUDA
  return atomicMax((uint32_t*)p, x);
#else
  return atomic_max(p, x);
#endif
}

inline uint32_t atomic_umin_i32_global(volatile __global uint32_t *p, uint32_t x) {
#ifdef FUTHARK_CUDA
  return atomicMin((uint32_t*)p, x);
#else
  return atomic_min(p, x);
#endif
}

inline uint32_t atomic_umin_i32_local(volatile __local uint32_t *p, uint32_t x) {
#ifdef FUTHARK_CUDA
  return atomicMin((uint32_t*)p, x);
#else
  return atomic_min(p, x);
#endif
}

inline int32_t atomic_and_i32_global(volatile __global int32_t *p, int32_t x) {
#ifdef FUTHARK_CUDA
  return atomicAnd((int32_t*)p, x);
#else
  return atomic_and(p, x);
#endif
}

inline int32_t atomic_and_i32_local(volatile __local int32_t *p, int32_t x) {
#ifdef FUTHARK_CUDA
  return atomicAnd((int32_t*)p, x);
#else
  return atomic_and(p, x);
#endif
}

inline int32_t atomic_or_i32_global(volatile __global int32_t *p, int32_t x) {
#ifdef FUTHARK_CUDA
  return atomicOr((int32_t*)p, x);
#else
  return atomic_or(p, x);
#endif
}

inline int32_t atomic_or_i32_local(volatile __local int32_t *p, int32_t x) {
#ifdef FUTHARK_CUDA
  return atomicOr((int32_t*)p, x);
#else
  return atomic_or(p, x);
#endif
}

inline int32_t atomic_xor_i32_global(volatile __global int32_t *p, int32_t x) {
#ifdef FUTHARK_CUDA
  return atomicXor((int32_t*)p, x);
#else
  return atomic_xor(p, x);
#endif
}

inline int32_t atomic_xor_i32_local(volatile __local int32_t *p, int32_t x) {
#ifdef FUTHARK_CUDA
  return atomicXor((int32_t*)p, x);
#else
  return atomic_xor(p, x);
#endif
}

// Start of 64 bit atomics

inline int64_t atomic_xchg_i64_global(volatile __global int64_t *p, int64_t x) {
#ifdef FUTHARK_CUDA
  return atomicExch((uint64_t*)p, x);
#else
  return atom_xor(p, x);
#endif
}

inline int64_t atomic_xchg_i64_local(volatile __local int64_t *p, int64_t x) {
#ifdef FUTHARK_CUDA
  return atomicExch((uint64_t*)p, x);
#else
  return atom_xor(p, x);
#endif
}

inline int64_t atomic_cmpxchg_i64_global(volatile __global int64_t *p,
                                         int64_t cmp, int64_t val) {
#ifdef FUTHARK_CUDA
  return atomicCAS((uint64_t*)p, cmp, val);
#else
  return atom_cmpxchg(p, cmp, val);
#endif
}

inline int64_t atomic_cmpxchg_i64_local(volatile __local int64_t *p,
                                        int64_t cmp, int64_t val) {
#ifdef FUTHARK_CUDA
  return atomicCAS((uint64_t*)p, cmp, val);
#else
  return atom_cmpxchg(p, cmp, val);
#endif
}

inline int64_t atomic_add_i64_global(volatile __global int64_t *p, int64_t x) {
#ifdef FUTHARK_CUDA
  return atomicAdd((uint64_t*)p, x);
#else
  return atom_add(p, x);
#endif
}

inline int64_t atomic_add_i64_local(volatile __local int64_t *p, int64_t x) {
#ifdef FUTHARK_CUDA
  return atomicAdd((uint64_t*)p, x);
#else
  return atom_add(p, x);
#endif
}

#ifdef FUTHARK_F64_ENABLED

inline double atomic_fadd_f64_global(volatile __global double *p, double x) {
#if defined(FUTHARK_CUDA) && __CUDA_ARCH__ >= 600
  return atomicAdd((double*)p, x);
#else
  union { int64_t i; double f; } old;
  union { int64_t i; double f; } assumed;
  old.f = *p;
  do {
    assumed.f = old.f;
    old.f = old.f + x;
    old.i = atomic_cmpxchg_i64_global((volatile __global int64_t*)p, assumed.i, old.i);
  } while (assumed.i != old.i);
  return old.f;
#endif
}

inline double atomic_fadd_f64_local(volatile __local double *p, double x) {
#if defined(FUTHARK_CUDA) && __CUDA_ARCH__ >= 600
  return atomicAdd((double*)p, x);
#else
  union { int64_t i; double f; } old;
  union { int64_t i; double f; } assumed;
  old.f = *p;
  do {
    assumed.f = old.f;
    old.f = old.f + x;
    old.i = atomic_cmpxchg_i64_local((volatile __local int64_t*)p, assumed.i, old.i);
  } while (assumed.i != old.i);
  return old.f;
#endif
}

#endif

inline int64_t atomic_smax_i64_global(volatile __global int64_t *p, int64_t x) {
#ifdef FUTHARK_CUDA
  return atomicMax((int64_t*)p, x);
#else
  return atom_max(p, x);
#endif
}

inline int64_t atomic_smax_i64_local(volatile __local int64_t *p, int64_t x) {
#ifdef FUTHARK_CUDA
  return atomicMax((int64_t*)p, x);
#else
  return atom_max(p, x);
#endif
}

inline int64_t atomic_smin_i64_global(volatile __global int64_t *p, int64_t x) {
#ifdef FUTHARK_CUDA
  return atomicMin((int64_t*)p, x);
#else
  return atom_min(p, x);
#endif
}

inline int64_t atomic_smin_i64_local(volatile __local int64_t *p, int64_t x) {
#ifdef FUTHARK_CUDA
  return atomicMin((int64_t*)p, x);
#else
  return atom_min(p, x);
#endif
}

inline uint64_t atomic_umax_i64_global(volatile __global uint64_t *p, uint64_t x) {
#ifdef FUTHARK_CUDA
  return atomicMax((uint64_t*)p, x);
#else
  return atom_max(p, x);
#endif
}

inline uint64_t atomic_umax_i64_local(volatile __local uint64_t *p, uint64_t x) {
#ifdef FUTHARK_CUDA
  return atomicMax((uint64_t*)p, x);
#else
  return atom_max(p, x);
#endif
}

inline uint64_t atomic_umin_i64_global(volatile __global uint64_t *p, uint64_t x) {
#ifdef FUTHARK_CUDA
  return atomicMin((uint64_t*)p, x);
#else
  return atom_min(p, x);
#endif
}

inline uint64_t atomic_umin_i64_local(volatile __local uint64_t *p, uint64_t x) {
#ifdef FUTHARK_CUDA
  return atomicMin((uint64_t*)p, x);
#else
  return atom_min(p, x);
#endif
}

inline int64_t atomic_and_i64_global(volatile __global int64_t *p, int64_t x) {
#ifdef FUTHARK_CUDA
  return atomicAnd((int64_t*)p, x);
#else
  return atom_and(p, x);
#endif
}

inline int64_t atomic_and_i64_local(volatile __local int64_t *p, int64_t x) {
#ifdef FUTHARK_CUDA
  return atomicAnd((int64_t*)p, x);
#else
  return atom_and(p, x);
#endif
}

inline int64_t atomic_or_i64_global(volatile __global int64_t *p, int64_t x) {
#ifdef FUTHARK_CUDA
  return atomicOr((int64_t*)p, x);
#else
  return atom_or(p, x);
#endif
}

inline int64_t atomic_or_i64_local(volatile __local int64_t *p, int64_t x) {
#ifdef FUTHARK_CUDA
  return atomicOr((int64_t*)p, x);
#else
  return atom_or(p, x);
#endif
}

inline int64_t atomic_xor_i64_global(volatile __global int64_t *p, int64_t x) {
#ifdef FUTHARK_CUDA
  return atomicXor((int64_t*)p, x);
#else
  return atom_xor(p, x);
#endif
}

inline int64_t atomic_xor_i64_local(volatile __local int64_t *p, int64_t x) {
#ifdef FUTHARK_CUDA
  return atomicXor((int64_t*)p, x);
#else
  return atom_xor(p, x);
#endif
}

// End of atomics.h




__kernel void builtinzhreplicate_f32zireplicate_6146(int32_t num_elems_6143,
                                                     float val_6144, __global
                                                     unsigned char *mem_6142)
{
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    int32_t replicate_gtid_6146;
    int32_t replicate_ltid_6147;
    int32_t replicate_gid_6148;
    
    replicate_gtid_6146 = get_global_id(0);
    replicate_ltid_6147 = get_local_id(0);
    replicate_gid_6148 = get_group_id(0);
    if (slt64(replicate_gtid_6146, num_elems_6143)) {
        ((__global float *) mem_6142)[sext_i32_i64(replicate_gtid_6146)] =
            val_6144;
    }
    
  error_0:
    return;
}
__kernel void gpu_map_transpose_f32(__local volatile
                                    int64_t *block_9_backing_aligned_0,
                                    int32_t destoffset_1, int32_t srcoffset_3,
                                    int32_t num_arrays_4, int32_t x_elems_5,
                                    int32_t y_elems_6, int32_t mulx_7,
                                    int32_t muly_8, __global
                                    unsigned char *destmem_0, __global
                                    unsigned char *srcmem_2)
{
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict block_9_backing_0 = (__local volatile
                                                         char *) block_9_backing_aligned_0;
    __local char *block_9;
    
    block_9 = (__local char *) block_9_backing_0;
    
    int32_t get_global_id_0_37;
    
    get_global_id_0_37 = get_global_id(0);
    
    int32_t get_local_id_0_38;
    
    get_local_id_0_38 = get_local_id(0);
    
    int32_t get_local_id_1_39;
    
    get_local_id_1_39 = get_local_id(1);
    
    int32_t get_group_id_0_40;
    
    get_group_id_0_40 = get_group_id(0);
    
    int32_t get_group_id_1_41;
    
    get_group_id_1_41 = get_group_id(1);
    
    int32_t get_group_id_2_42;
    
    get_group_id_2_42 = get_group_id(2);
    
    int32_t our_array_offset_30 = get_group_id_2_42 * x_elems_5 * y_elems_6;
    int32_t odata_offset_33 = squot32(destoffset_1, 4) + our_array_offset_30;
    int32_t idata_offset_34 = squot32(srcoffset_3, 4) + our_array_offset_30;
    int32_t x_index_31 = get_global_id_0_37;
    int32_t y_index_32 = get_group_id_1_41 * 32 + get_local_id_1_39;
    
    if (slt32(x_index_31, x_elems_5)) {
        for (int32_t j_43 = 0; j_43 < 4; j_43++) {
            int32_t index_in_35 = (y_index_32 + j_43 * 8) * x_elems_5 +
                    x_index_31;
            
            if (slt32(y_index_32 + j_43 * 8, y_elems_6)) {
                ((__local float *) block_9)[sext_i32_i64((get_local_id_1_39 +
                                                          j_43 * 8) * 33 +
                                            get_local_id_0_38)] = ((__global
                                                                    float *) srcmem_2)[sext_i32_i64(idata_offset_34 +
                                                                                       index_in_35)];
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    x_index_31 = get_group_id_1_41 * 32 + get_local_id_0_38;
    y_index_32 = get_group_id_0_40 * 32 + get_local_id_1_39;
    if (slt32(x_index_31, y_elems_6)) {
        for (int32_t j_43 = 0; j_43 < 4; j_43++) {
            int32_t index_out_36 = (y_index_32 + j_43 * 8) * y_elems_6 +
                    x_index_31;
            
            if (slt32(y_index_32 + j_43 * 8, x_elems_5)) {
                ((__global float *) destmem_0)[sext_i32_i64(odata_offset_33 +
                                               index_out_36)] = ((__local
                                                                  float *) block_9)[sext_i32_i64(get_local_id_0_38 *
                                                                                    33 +
                                                                                    get_local_id_1_39 +
                                                                                    j_43 *
                                                                                    8)];
            }
        }
    }
    
  error_0:
    return;
}
__kernel void gpu_map_transpose_f32_low_height(__local volatile
                                               int64_t *block_9_backing_aligned_0,
                                               int32_t destoffset_1,
                                               int32_t srcoffset_3,
                                               int32_t num_arrays_4,
                                               int32_t x_elems_5,
                                               int32_t y_elems_6,
                                               int32_t mulx_7, int32_t muly_8,
                                               __global
                                               unsigned char *destmem_0,
                                               __global unsigned char *srcmem_2)
{
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict block_9_backing_0 = (__local volatile
                                                         char *) block_9_backing_aligned_0;
    __local char *block_9;
    
    block_9 = (__local char *) block_9_backing_0;
    
    int32_t get_global_id_0_37;
    
    get_global_id_0_37 = get_global_id(0);
    
    int32_t get_local_id_0_38;
    
    get_local_id_0_38 = get_local_id(0);
    
    int32_t get_local_id_1_39;
    
    get_local_id_1_39 = get_local_id(1);
    
    int32_t get_group_id_0_40;
    
    get_group_id_0_40 = get_group_id(0);
    
    int32_t get_group_id_1_41;
    
    get_group_id_1_41 = get_group_id(1);
    
    int32_t get_group_id_2_42;
    
    get_group_id_2_42 = get_group_id(2);
    
    int32_t our_array_offset_30 = get_group_id_2_42 * x_elems_5 * y_elems_6;
    int32_t odata_offset_33 = squot32(destoffset_1, 4) + our_array_offset_30;
    int32_t idata_offset_34 = squot32(srcoffset_3, 4) + our_array_offset_30;
    int32_t x_index_31 = get_group_id_0_40 * 16 * mulx_7 + get_local_id_0_38 +
            srem32(get_local_id_1_39, mulx_7) * 16;
    int32_t y_index_32 = get_group_id_1_41 * 16 + squot32(get_local_id_1_39,
                                                          mulx_7);
    int32_t index_in_35 = y_index_32 * x_elems_5 + x_index_31;
    
    if (slt32(x_index_31, x_elems_5) && slt32(y_index_32, y_elems_6)) {
        ((__local float *) block_9)[sext_i32_i64(get_local_id_1_39 * 17 +
                                    get_local_id_0_38)] = ((__global
                                                            float *) srcmem_2)[sext_i32_i64(idata_offset_34 +
                                                                               index_in_35)];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    x_index_31 = get_group_id_1_41 * 16 + squot32(get_local_id_0_38, mulx_7);
    y_index_32 = get_group_id_0_40 * 16 * mulx_7 + get_local_id_1_39 +
        srem32(get_local_id_0_38, mulx_7) * 16;
    
    int32_t index_out_36 = y_index_32 * y_elems_6 + x_index_31;
    
    if (slt32(x_index_31, y_elems_6) && slt32(y_index_32, x_elems_5)) {
        ((__global float *) destmem_0)[sext_i32_i64(odata_offset_33 +
                                       index_out_36)] = ((__local
                                                          float *) block_9)[sext_i32_i64(get_local_id_0_38 *
                                                                            17 +
                                                                            get_local_id_1_39)];
    }
    
  error_0:
    return;
}
__kernel void gpu_map_transpose_f32_low_width(__local volatile
                                              int64_t *block_9_backing_aligned_0,
                                              int32_t destoffset_1,
                                              int32_t srcoffset_3,
                                              int32_t num_arrays_4,
                                              int32_t x_elems_5,
                                              int32_t y_elems_6, int32_t mulx_7,
                                              int32_t muly_8, __global
                                              unsigned char *destmem_0, __global
                                              unsigned char *srcmem_2)
{
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict block_9_backing_0 = (__local volatile
                                                         char *) block_9_backing_aligned_0;
    __local char *block_9;
    
    block_9 = (__local char *) block_9_backing_0;
    
    int32_t get_global_id_0_37;
    
    get_global_id_0_37 = get_global_id(0);
    
    int32_t get_local_id_0_38;
    
    get_local_id_0_38 = get_local_id(0);
    
    int32_t get_local_id_1_39;
    
    get_local_id_1_39 = get_local_id(1);
    
    int32_t get_group_id_0_40;
    
    get_group_id_0_40 = get_group_id(0);
    
    int32_t get_group_id_1_41;
    
    get_group_id_1_41 = get_group_id(1);
    
    int32_t get_group_id_2_42;
    
    get_group_id_2_42 = get_group_id(2);
    
    int32_t our_array_offset_30 = get_group_id_2_42 * x_elems_5 * y_elems_6;
    int32_t odata_offset_33 = squot32(destoffset_1, 4) + our_array_offset_30;
    int32_t idata_offset_34 = squot32(srcoffset_3, 4) + our_array_offset_30;
    int32_t x_index_31 = get_group_id_0_40 * 16 + squot32(get_local_id_0_38,
                                                          muly_8);
    int32_t y_index_32 = get_group_id_1_41 * 16 * muly_8 + get_local_id_1_39 +
            srem32(get_local_id_0_38, muly_8) * 16;
    int32_t index_in_35 = y_index_32 * x_elems_5 + x_index_31;
    
    if (slt32(x_index_31, x_elems_5) && slt32(y_index_32, y_elems_6)) {
        ((__local float *) block_9)[sext_i32_i64(get_local_id_1_39 * 17 +
                                    get_local_id_0_38)] = ((__global
                                                            float *) srcmem_2)[sext_i32_i64(idata_offset_34 +
                                                                               index_in_35)];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    x_index_31 = get_group_id_1_41 * 16 * muly_8 + get_local_id_0_38 +
        srem32(get_local_id_1_39, muly_8) * 16;
    y_index_32 = get_group_id_0_40 * 16 + squot32(get_local_id_1_39, muly_8);
    
    int32_t index_out_36 = y_index_32 * y_elems_6 + x_index_31;
    
    if (slt32(x_index_31, y_elems_6) && slt32(y_index_32, x_elems_5)) {
        ((__global float *) destmem_0)[sext_i32_i64(odata_offset_33 +
                                       index_out_36)] = ((__local
                                                          float *) block_9)[sext_i32_i64(get_local_id_0_38 *
                                                                            17 +
                                                                            get_local_id_1_39)];
    }
    
  error_0:
    return;
}
__kernel void gpu_map_transpose_f32_small(__local volatile
                                          int64_t *block_9_backing_aligned_0,
                                          int32_t destoffset_1,
                                          int32_t srcoffset_3,
                                          int32_t num_arrays_4,
                                          int32_t x_elems_5, int32_t y_elems_6,
                                          int32_t mulx_7, int32_t muly_8,
                                          __global unsigned char *destmem_0,
                                          __global unsigned char *srcmem_2)
{
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    __local volatile char *restrict block_9_backing_0 = (__local volatile
                                                         char *) block_9_backing_aligned_0;
    __local char *block_9;
    
    block_9 = (__local char *) block_9_backing_0;
    
    int32_t get_global_id_0_37;
    
    get_global_id_0_37 = get_global_id(0);
    
    int32_t get_local_id_0_38;
    
    get_local_id_0_38 = get_local_id(0);
    
    int32_t get_local_id_1_39;
    
    get_local_id_1_39 = get_local_id(1);
    
    int32_t get_group_id_0_40;
    
    get_group_id_0_40 = get_group_id(0);
    
    int32_t get_group_id_1_41;
    
    get_group_id_1_41 = get_group_id(1);
    
    int32_t get_group_id_2_42;
    
    get_group_id_2_42 = get_group_id(2);
    
    int32_t our_array_offset_30 = squot32(get_global_id_0_37, y_elems_6 *
                                          x_elems_5) * (y_elems_6 * x_elems_5);
    int32_t x_index_31 = squot32(srem32(get_global_id_0_37, y_elems_6 *
                                        x_elems_5), y_elems_6);
    int32_t y_index_32 = srem32(get_global_id_0_37, y_elems_6);
    int32_t odata_offset_33 = squot32(destoffset_1, 4) + our_array_offset_30;
    int32_t idata_offset_34 = squot32(srcoffset_3, 4) + our_array_offset_30;
    int32_t index_in_35 = y_index_32 * x_elems_5 + x_index_31;
    int32_t index_out_36 = x_index_31 * y_elems_6 + y_index_32;
    
    if (slt32(get_global_id_0_37, x_elems_5 * y_elems_6 * num_arrays_4)) {
        ((__global float *) destmem_0)[sext_i32_i64(odata_offset_33 +
                                       index_out_36)] = ((__global
                                                          float *) srcmem_2)[sext_i32_i64(idata_offset_34 +
                                                                             index_in_35)];
    }
    
  error_0:
    return;
}
__kernel void mainzisegmap_6042(__global int *global_failure,
                                int64_t num_steps_5691, int64_t num_groups_6054,
                                __global unsigned char *mem_6102, __global
                                unsigned char *mem_6104, __global
                                unsigned char *mem_6134)
{
    #define segmap_group_sizze_6053 (mainzisegmap_group_sizze_6044)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_6156;
    int32_t local_tid_6157;
    int64_t group_sizze_6160;
    int32_t wave_sizze_6159;
    int32_t group_tid_6158;
    
    global_tid_6156 = get_global_id(0);
    local_tid_6157 = get_local_id(0);
    group_sizze_6160 = get_local_size(0);
    wave_sizze_6159 = LOCKSTEP_WIDTH;
    group_tid_6158 = get_group_id(0);
    
    int32_t phys_tid_6042;
    
    phys_tid_6042 = global_tid_6156;
    
    int32_t phys_group_id_6161;
    
    phys_group_id_6161 = get_group_id(0);
    for (int32_t i_6162 = 0; i_6162 <
         sdiv_up32(sext_i64_i32(sdiv_up64(num_steps_5691,
                                          segmap_group_sizze_6053)) -
                   phys_group_id_6161, sext_i64_i32(num_groups_6054));
         i_6162++) {
        int32_t virt_group_id_6163 = phys_group_id_6161 + i_6162 *
                sext_i64_i32(num_groups_6054);
        int64_t gtid_6041 = sext_i32_i64(virt_group_id_6163) *
                segmap_group_sizze_6053 + sext_i32_i64(local_tid_6157);
        
        if (slt64(gtid_6041, num_steps_5691)) {
            float x_6057 = ((__global float *) mem_6102)[gtid_6041];
            float x_6058 = ((__global float *) mem_6104)[gtid_6041];
            float mem_6131[2];
            
            mem_6131[(int64_t) 0] = x_6057;
            mem_6131[(int64_t) 1] = x_6058;
            for (int64_t i_6164 = 0; i_6164 < (int64_t) 2; i_6164++) {
                ((__global float *) mem_6134)[i_6164 * num_steps_5691 +
                                              gtid_6041] = mem_6131[i_6164];
            }
        }
        barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
    }
    
  error_0:
    return;
    #undef segmap_group_sizze_6053
}
__kernel void runge_kutta_fwdzisegmap_6082(__global int *global_failure,
                                           int64_t num_steps_5768,
                                           int64_t num_groups_6094, __global
                                           unsigned char *mem_6102, __global
                                           unsigned char *mem_6104, __global
                                           unsigned char *mem_6134)
{
    #define segmap_group_sizze_6093 (runge_kutta_fwdzisegmap_group_sizze_6084)
    
    const int block_dim0 = 0;
    const int block_dim1 = 1;
    const int block_dim2 = 2;
    
    if (*global_failure >= 0)
        return;
    
    int32_t global_tid_6158;
    int32_t local_tid_6159;
    int64_t group_sizze_6162;
    int32_t wave_sizze_6161;
    int32_t group_tid_6160;
    
    global_tid_6158 = get_global_id(0);
    local_tid_6159 = get_local_id(0);
    group_sizze_6162 = get_local_size(0);
    wave_sizze_6161 = LOCKSTEP_WIDTH;
    group_tid_6160 = get_group_id(0);
    
    int32_t phys_tid_6082;
    
    phys_tid_6082 = global_tid_6158;
    
    int32_t phys_group_id_6163;
    
    phys_group_id_6163 = get_group_id(0);
    for (int32_t i_6164 = 0; i_6164 <
         sdiv_up32(sext_i64_i32(sdiv_up64(num_steps_5768,
                                          segmap_group_sizze_6093)) -
                   phys_group_id_6163, sext_i64_i32(num_groups_6094));
         i_6164++) {
        int32_t virt_group_id_6165 = phys_group_id_6163 + i_6164 *
                sext_i64_i32(num_groups_6094);
        int64_t gtid_6081 = sext_i32_i64(virt_group_id_6165) *
                segmap_group_sizze_6093 + sext_i32_i64(local_tid_6159);
        
        if (slt64(gtid_6081, num_steps_5768)) {
            float x_6097 = ((__global float *) mem_6102)[gtid_6081];
            float x_6098 = ((__global float *) mem_6104)[gtid_6081];
            float mem_6131[2];
            
            mem_6131[(int64_t) 0] = x_6097;
            mem_6131[(int64_t) 1] = x_6098;
            for (int64_t i_6166 = 0; i_6166 < (int64_t) 2; i_6166++) {
                ((__global float *) mem_6134)[i_6166 * num_steps_5768 +
                                              gtid_6081] = mem_6131[i_6166];
            }
        }
        barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
    }
    
  error_0:
    return;
    #undef segmap_group_sizze_6093
}
"""
# Start of values.py.

# Hacky parser/reader/writer for values written in Futhark syntax.
# Used for reading stdin when compiling standalone programs with the
# Python code generator.

import numpy as np
import string
import struct
import sys

class ReaderInput:
    def __init__(self, f):
        self.f = f
        self.lookahead_buffer = []

    def get_char(self):
        if len(self.lookahead_buffer) == 0:
            return self.f.read(1)
        else:
            c = self.lookahead_buffer[0]
            self.lookahead_buffer = self.lookahead_buffer[1:]
            return c

    def unget_char(self, c):
        self.lookahead_buffer = [c] + self.lookahead_buffer

    def get_chars(self, n):
        n1 = min(n, len(self.lookahead_buffer))
        s = b''.join(self.lookahead_buffer[:n1])
        self.lookahead_buffer = self.lookahead_buffer[n1:]
        n2 = n - n1
        if n2 > 0:
            s += self.f.read(n2)
        return s

    def peek_char(self):
        c = self.get_char()
        if c:
            self.unget_char(c)
        return c

def skip_spaces(f):
    c = f.get_char()
    while c != None:
        if c.isspace():
            c = f.get_char()
        elif c == b'-':
          # May be line comment.
          if f.peek_char() == b'-':
            # Yes, line comment. Skip to end of line.
            while (c != b'\n' and c != None):
              c = f.get_char()
          else:
            break
        else:
          break
    if c:
        f.unget_char(c)

def parse_specific_char(f, expected):
    got = f.get_char()
    if got != expected:
        f.unget_char(got)
        raise ValueError
    return True

def parse_specific_string(f, s):
    # This funky mess is intended, and is caused by the fact that if `type(b) ==
    # bytes` then `type(b[0]) == int`, but we need to match each element with a
    # `bytes`, so therefore we make each character an array element
    b = s.encode('utf8')
    bs = [b[i:i+1] for i in range(len(b))]
    read = []
    try:
        for c in bs:
            parse_specific_char(f, c)
            read.append(c)
        return True
    except ValueError:
        for c in read[::-1]:
            f.unget_char(c)
        raise

def optional(p, *args):
    try:
        return p(*args)
    except ValueError:
        return None

def optional_specific_string(f, s):
    c = f.peek_char()
    # This funky mess is intended, and is caused by the fact that if `type(b) ==
    # bytes` then `type(b[0]) == int`, but we need to match each element with a
    # `bytes`, so therefore we make each character an array element
    b = s.encode('utf8')
    bs = [b[i:i+1] for i in range(len(b))]
    if c == bs[0]:
        return parse_specific_string(f, s)
    else:
        return False

def sepBy(p, sep, *args):
    elems = []
    x = optional(p, *args)
    if x != None:
        elems += [x]
        while optional(sep, *args) != None:
            x = p(*args)
            elems += [x]
    return elems

# Assumes '0x' has already been read
def parse_hex_int(f):
    s = b''
    c = f.get_char()
    while c != None:
        if c in b'01234556789ABCDEFabcdef':
            s += c
            c = f.get_char()
        elif c == b'_':
            c = f.get_char() # skip _
        else:
            f.unget_char(c)
            break
    return str(int(s, 16)).encode('utf8') # ugh

def parse_int(f):
    s = b''
    c = f.get_char()
    if c == b'0' and f.peek_char() in b'xX':
        c = f.get_char() # skip X
        return parse_hex_int(f)
    else:
        while c != None:
            if c.isdigit():
                s += c
                c = f.get_char()
            elif c == b'_':
                c = f.get_char() # skip _
            else:
                f.unget_char(c)
                break
        if len(s) == 0:
            raise ValueError
        return s

def parse_int_signed(f):
    s = b''
    c = f.get_char()

    if c == b'-' and f.peek_char().isdigit():
      return c + parse_int(f)
    else:
      if c != b'+':
          f.unget_char(c)
      return parse_int(f)

def read_str_comma(f):
    skip_spaces(f)
    parse_specific_char(f, b',')
    return b','

def read_str_int(f, s):
    skip_spaces(f)
    x = int(parse_int_signed(f))
    optional_specific_string(f, s)
    return x

def read_str_uint(f, s):
    skip_spaces(f)
    x = int(parse_int(f))
    optional_specific_string(f, s)
    return x

def read_str_i8(f):
    return np.int8(read_str_int(f, 'i8'))
def read_str_i16(f):
    return np.int16(read_str_int(f, 'i16'))
def read_str_i32(f):
    return np.int32(read_str_int(f, 'i32'))
def read_str_i64(f):
    return np.int64(read_str_int(f, 'i64'))

def read_str_u8(f):
    return np.uint8(read_str_int(f, 'u8'))
def read_str_u16(f):
    return np.uint16(read_str_int(f, 'u16'))
def read_str_u32(f):
    return np.uint32(read_str_int(f, 'u32'))
def read_str_u64(f):
    return np.uint64(read_str_int(f, 'u64'))

def read_char(f):
    skip_spaces(f)
    parse_specific_char(f, b'\'')
    c = f.get_char()
    parse_specific_char(f, b'\'')
    return c

def read_str_hex_float(f, sign):
    int_part = parse_hex_int(f)
    parse_specific_char(f, b'.')
    frac_part = parse_hex_int(f)
    parse_specific_char(f, b'p')
    exponent = parse_int(f)

    int_val = int(int_part, 16)
    frac_val = float(int(frac_part, 16)) / (16 ** len(frac_part))
    exp_val = int(exponent)

    total_val = (int_val + frac_val) * (2.0 ** exp_val)
    if sign == b'-':
        total_val = -1 * total_val

    return float(total_val)


def read_str_decimal(f):
    skip_spaces(f)
    c = f.get_char()
    if (c == b'-'):
      sign = b'-'
    else:
      f.unget_char(c)
      sign = b''

    # Check for hexadecimal float
    c = f.get_char()
    if (c == '0' and (f.peek_char() in ['x', 'X'])):
        f.get_char()
        return read_str_hex_float(f, sign)
    else:
        f.unget_char(c)

    bef = optional(parse_int, f)
    if bef == None:
        bef = b'0'
        parse_specific_char(f, b'.')
        aft = parse_int(f)
    elif optional(parse_specific_char, f, b'.'):
        aft = parse_int(f)
    else:
        aft = b'0'
    if (optional(parse_specific_char, f, b'E') or
        optional(parse_specific_char, f, b'e')):
        expt = parse_int_signed(f)
    else:
        expt = b'0'
    return float(sign + bef + b'.' + aft + b'E' + expt)

def read_str_f32(f):
    skip_spaces(f)
    try:
        parse_specific_string(f, 'f32.nan')
        return np.float32(np.nan)
    except ValueError:
        try:
            parse_specific_string(f, 'f32.inf')
            return np.float32(np.inf)
        except ValueError:
            try:
               parse_specific_string(f, '-f32.inf')
               return np.float32(-np.inf)
            except ValueError:
               x = read_str_decimal(f)
               optional_specific_string(f, 'f32')
               return x

def read_str_f64(f):
    skip_spaces(f)
    try:
        parse_specific_string(f, 'f64.nan')
        return np.float64(np.nan)
    except ValueError:
        try:
            parse_specific_string(f, 'f64.inf')
            return np.float64(np.inf)
        except ValueError:
            try:
               parse_specific_string(f, '-f64.inf')
               return np.float64(-np.inf)
            except ValueError:
               x = read_str_decimal(f)
               optional_specific_string(f, 'f64')
               return x

def read_str_bool(f):
    skip_spaces(f)
    if f.peek_char() == b't':
        parse_specific_string(f, 'true')
        return True
    elif f.peek_char() == b'f':
        parse_specific_string(f, 'false')
        return False
    else:
        raise ValueError

def read_str_empty_array(f, type_name, rank):
    parse_specific_string(f, 'empty')
    parse_specific_char(f, b'(')
    dims = []
    for i in range(rank):
        parse_specific_string(f, '[')
        dims += [int(parse_int(f))]
        parse_specific_string(f, ']')
    if np.product(dims) != 0:
        raise ValueError
    parse_specific_string(f, type_name)
    parse_specific_char(f, b')')

    return tuple(dims)

def read_str_array_elems(f, elem_reader, type_name, rank):
    skip_spaces(f)
    try:
        parse_specific_char(f, b'[')
    except ValueError:
        return read_str_empty_array(f, type_name, rank)
    else:
        xs = sepBy(elem_reader, read_str_comma, f)
        skip_spaces(f)
        parse_specific_char(f, b']')
        return xs

def read_str_array_helper(f, elem_reader, type_name, rank):
    def nested_row_reader(_):
        return read_str_array_helper(f, elem_reader, type_name, rank-1)
    if rank == 1:
        row_reader = elem_reader
    else:
        row_reader = nested_row_reader
    return read_str_array_elems(f, row_reader, type_name, rank)

def expected_array_dims(l, rank):
  if rank > 1:
      n = len(l)
      if n == 0:
          elem = []
      else:
          elem = l[0]
      return [n] + expected_array_dims(elem, rank-1)
  else:
      return [len(l)]

def verify_array_dims(l, dims):
    if dims[0] != len(l):
        raise ValueError
    if len(dims) > 1:
        for x in l:
            verify_array_dims(x, dims[1:])

def read_str_array(f, elem_reader, type_name, rank, bt):
    elems = read_str_array_helper(f, elem_reader, type_name, rank)
    if type(elems) == tuple:
        # Empty array
        return np.empty(elems, dtype=bt)
    else:
        dims = expected_array_dims(elems, rank)
        verify_array_dims(elems, dims)
        return np.array(elems, dtype=bt)

################################################################################

READ_BINARY_VERSION = 2

# struct format specified at
# https://docs.python.org/2/library/struct.html#format-characters

def mk_bin_scalar_reader(t):
    def bin_reader(f):
        fmt = FUTHARK_PRIMTYPES[t]['bin_format']
        size = FUTHARK_PRIMTYPES[t]['size']
        return struct.unpack('<' + fmt, f.get_chars(size))[0]
    return bin_reader

read_bin_i8 = mk_bin_scalar_reader('i8')
read_bin_i16 = mk_bin_scalar_reader('i16')
read_bin_i32 = mk_bin_scalar_reader('i32')
read_bin_i64 = mk_bin_scalar_reader('i64')

read_bin_u8 = mk_bin_scalar_reader('u8')
read_bin_u16 = mk_bin_scalar_reader('u16')
read_bin_u32 = mk_bin_scalar_reader('u32')
read_bin_u64 = mk_bin_scalar_reader('u64')

read_bin_f32 = mk_bin_scalar_reader('f32')
read_bin_f64 = mk_bin_scalar_reader('f64')

read_bin_bool = mk_bin_scalar_reader('bool')

def read_is_binary(f):
    skip_spaces(f)
    c = f.get_char()
    if c == b'b':
        bin_version = read_bin_u8(f)
        if bin_version != READ_BINARY_VERSION:
            panic(1, "binary-input: File uses version %i, but I only understand version %i.\n",
                  bin_version, READ_BINARY_VERSION)
        return True
    else:
        f.unget_char(c)
        return False

FUTHARK_PRIMTYPES = {
    'i8':  {'binname' : b"  i8",
            'size' : 1,
            'bin_reader': read_bin_i8,
            'str_reader': read_str_i8,
            'bin_format': 'b',
            'numpy_type': np.int8 },

    'i16': {'binname' : b" i16",
            'size' : 2,
            'bin_reader': read_bin_i16,
            'str_reader': read_str_i16,
            'bin_format': 'h',
            'numpy_type': np.int16 },

    'i32': {'binname' : b" i32",
            'size' : 4,
            'bin_reader': read_bin_i32,
            'str_reader': read_str_i32,
            'bin_format': 'i',
            'numpy_type': np.int32 },

    'i64': {'binname' : b" i64",
            'size' : 8,
            'bin_reader': read_bin_i64,
            'str_reader': read_str_i64,
            'bin_format': 'q',
            'numpy_type': np.int64},

    'u8':  {'binname' : b"  u8",
            'size' : 1,
            'bin_reader': read_bin_u8,
            'str_reader': read_str_u8,
            'bin_format': 'B',
            'numpy_type': np.uint8 },

    'u16': {'binname' : b" u16",
            'size' : 2,
            'bin_reader': read_bin_u16,
            'str_reader': read_str_u16,
            'bin_format': 'H',
            'numpy_type': np.uint16 },

    'u32': {'binname' : b" u32",
            'size' : 4,
            'bin_reader': read_bin_u32,
            'str_reader': read_str_u32,
            'bin_format': 'I',
            'numpy_type': np.uint32 },

    'u64': {'binname' : b" u64",
            'size' : 8,
            'bin_reader': read_bin_u64,
            'str_reader': read_str_u64,
            'bin_format': 'Q',
            'numpy_type': np.uint64 },

    'f32': {'binname' : b" f32",
            'size' : 4,
            'bin_reader': read_bin_f32,
            'str_reader': read_str_f32,
            'bin_format': 'f',
            'numpy_type': np.float32 },

    'f64': {'binname' : b" f64",
            'size' : 8,
            'bin_reader': read_bin_f64,
            'str_reader': read_str_f64,
            'bin_format': 'd',
            'numpy_type': np.float64 },

    'bool': {'binname' : b"bool",
             'size' : 1,
             'bin_reader': read_bin_bool,
             'str_reader': read_str_bool,
             'bin_format': 'b',
             'numpy_type': np.bool }
}

def read_bin_read_type(f):
    read_binname = f.get_chars(4)

    for (k,v) in FUTHARK_PRIMTYPES.items():
        if v['binname'] == read_binname:
            return k
    panic(1, "binary-input: Did not recognize the type '%s'.\n", read_binname)

def numpy_type_to_type_name(t):
    for (k,v) in FUTHARK_PRIMTYPES.items():
        if v['numpy_type'] == t:
            return k
    raise Exception('Unknown Numpy type: {}'.format(t))

def read_bin_ensure_scalar(f, expected_type):
  dims = read_bin_i8(f)

  if dims != 0:
      panic(1, "binary-input: Expected scalar (0 dimensions), but got array with %i dimensions.\n", dims)

  bin_type = read_bin_read_type(f)
  if bin_type != expected_type:
      panic(1, "binary-input: Expected scalar of type %s but got scalar of type %s.\n",
            expected_type, bin_type)

# ------------------------------------------------------------------------------
# General interface for reading Primitive Futhark Values
# ------------------------------------------------------------------------------

def read_scalar(f, ty):
    if read_is_binary(f):
        read_bin_ensure_scalar(f, ty)
        return FUTHARK_PRIMTYPES[ty]['bin_reader'](f)
    return FUTHARK_PRIMTYPES[ty]['str_reader'](f)

def read_array(f, expected_type, rank):
    if not read_is_binary(f):
        str_reader = FUTHARK_PRIMTYPES[expected_type]['str_reader']
        return read_str_array(f, str_reader, expected_type, rank,
                              FUTHARK_PRIMTYPES[expected_type]['numpy_type'])

    bin_rank = read_bin_u8(f)

    if bin_rank != rank:
        panic(1, "binary-input: Expected %i dimensions, but got array with %i dimensions.\n",
              rank, bin_rank)

    bin_type_enum = read_bin_read_type(f)
    if expected_type != bin_type_enum:
        panic(1, "binary-input: Expected %iD-array with element type '%s' but got %iD-array with element type '%s'.\n",
              rank, expected_type, bin_rank, bin_type_enum)

    shape = []
    elem_count = 1
    for i in range(rank):
        bin_size = read_bin_u64(f)
        elem_count *= bin_size
        shape.append(bin_size)

    bin_fmt = FUTHARK_PRIMTYPES[bin_type_enum]['bin_format']

    # We first read the expected number of types into a bytestring,
    # then use np.fromstring.  This is because np.fromfile does not
    # work on things that are insufficiently file-like, like a network
    # stream.
    bytes = f.get_chars(elem_count * FUTHARK_PRIMTYPES[expected_type]['size'])
    arr = np.fromstring(bytes, dtype=FUTHARK_PRIMTYPES[bin_type_enum]['numpy_type'])
    arr.shape = shape

    return arr

if sys.version_info >= (3,0):
    input_reader = ReaderInput(sys.stdin.buffer)
else:
    input_reader = ReaderInput(sys.stdin)

import re

def read_value(type_desc, reader=input_reader):
    """Read a value of the given type.  The type is a string
representation of the Futhark type."""
    m = re.match(r'((?:\[\])*)([a-z0-9]+)$', type_desc)
    if m:
        dims = int(len(m.group(1))/2)
        basetype = m.group(2)
        assert basetype in FUTHARK_PRIMTYPES, "Unknown type: {}".format(type_desc)
        if dims > 0:
            return read_array(reader, basetype, dims)
        else:
            return read_scalar(reader, basetype)
        return (dims, basetype)

def end_of_input(entry, f=input_reader):
    skip_spaces(f)
    if f.get_char() != b'':
        panic(1, "Expected EOF on stdin after reading input for \"%s\".", entry)

def write_value_text(v, out=sys.stdout):
    if type(v) == np.uint8:
        out.write("%uu8" % v)
    elif type(v) == np.uint16:
        out.write("%uu16" % v)
    elif type(v) == np.uint32:
        out.write("%uu32" % v)
    elif type(v) == np.uint64:
        out.write("%uu64" % v)
    elif type(v) == np.int8:
        out.write("%di8" % v)
    elif type(v) == np.int16:
        out.write("%di16" % v)
    elif type(v) == np.int32:
        out.write("%di32" % v)
    elif type(v) == np.int64:
        out.write("%di64" % v)
    elif type(v) in [np.bool, np.bool_]:
        if v:
            out.write("true")
        else:
            out.write("false")
    elif type(v) == np.float32:
        if np.isnan(v):
            out.write('f32.nan')
        elif np.isinf(v):
            if v >= 0:
                out.write('f32.inf')
            else:
                out.write('-f32.inf')
        else:
            out.write("%.6ff32" % v)
    elif type(v) == np.float64:
        if np.isnan(v):
            out.write('f64.nan')
        elif np.isinf(v):
            if v >= 0:
                out.write('f64.inf')
            else:
                out.write('-f64.inf')
        else:
            out.write("%.6ff64" % v)
    elif type(v) == np.ndarray:
        if np.product(v.shape) == 0:
            tname = numpy_type_to_type_name(v.dtype)
            out.write('empty({}{})'.format(''.join(['[{}]'.format(d)
                                                    for d in v.shape]), tname))
        else:
            first = True
            out.write('[')
            for x in v:
                if not first: out.write(', ')
                first = False
                write_value(x, out=out)
            out.write(']')
    else:
        raise Exception("Cannot print value of type {}: {}".format(type(v), v))

type_strs = { np.dtype('int8'): b'  i8',
              np.dtype('int16'): b' i16',
              np.dtype('int32'): b' i32',
              np.dtype('int64'): b' i64',
              np.dtype('uint8'): b'  u8',
              np.dtype('uint16'): b' u16',
              np.dtype('uint32'): b' u32',
              np.dtype('uint64'): b' u64',
              np.dtype('float32'): b' f32',
              np.dtype('float64'): b' f64',
              np.dtype('bool'): b'bool'}

def construct_binary_value(v):
    t = v.dtype
    shape = v.shape

    elems = 1
    for d in shape:
        elems *= d

    num_bytes = 1 + 1 + 1 + 4 + len(shape) * 8 + elems * t.itemsize
    bytes = bytearray(num_bytes)
    bytes[0] = np.int8(ord('b'))
    bytes[1] = 2
    bytes[2] = np.int8(len(shape))
    bytes[3:7] = type_strs[t]

    for i in range(len(shape)):
        bytes[7+i*8:7+(i+1)*8] = np.int64(shape[i]).tostring()

    bytes[7+len(shape)*8:] = np.ascontiguousarray(v).tostring()

    return bytes

def write_value_binary(v, out=sys.stdout):
    if sys.version_info >= (3,0):
        out = out.buffer
    out.write(construct_binary_value(v))

def write_value(v, out=sys.stdout, binary=False):
    if binary:
        return write_value_binary(v, out=out)
    else:
        return write_value_text(v, out=out)

# End of values.py.
# Start of memory.py.

import ctypes as ct

def addressOffset(x, offset, bt):
  return ct.cast(ct.addressof(x.contents)+int(offset), ct.POINTER(bt))

def allocateMem(size):
  return ct.cast((ct.c_byte * max(0,size))(), ct.POINTER(ct.c_byte))

# Copy an array if its is not-None.  This is important for treating
# Numpy arrays as flat memory, but has some overhead.
def normaliseArray(x):
  if (x.base is x) or (x.base is None):
    return x
  else:
    return x.copy()

def unwrapArray(x):
  return normaliseArray(x).ctypes.data_as(ct.POINTER(ct.c_byte))

def createArray(x, shape):
  # HACK: np.ctypeslib.as_array may fail if the shape contains zeroes,
  # for some reason.
  if any(map(lambda x: x == 0, shape)):
      return np.ndarray(shape, dtype=x._type_)
  else:
      return np.ctypeslib.as_array(x, shape=shape)

def indexArray(x, offset, bt, nptype):
  return nptype(addressOffset(x, offset*ct.sizeof(bt), bt)[0])

def writeScalarArray(x, offset, v):
  ct.memmove(ct.addressof(x.contents)+int(offset)*ct.sizeof(v), ct.addressof(v), ct.sizeof(v))

# An opaque Futhark value.
class opaque(object):
  def __init__(self, desc, *payload):
    self.data = payload
    self.desc = desc

  def __repr__(self):
    return "<opaque Futhark value of type {}>".format(self.desc)

# End of memory.py.
# Start of panic.py.

def panic(exitcode, fmt, *args):
    sys.stderr.write('%s: ' % sys.argv[0])
    sys.stderr.write(fmt % args)
    sys.stderr.write('\n')
    sys.exit(exitcode)

# End of panic.py.
# Start of tuning.py

def read_tuning_file(kvs, f):
    for line in f.read().splitlines():
        size, value = line.split('=')
        kvs[size] = int(value)
    return kvs

# End of tuning.py.
# Start of scalar.py.

import numpy as np
import math
import struct

def intlit(t, x):
  if t == np.int8:
    return np.int8(x)
  elif t == np.int16:
    return np.int16(x)
  elif t == np.int32:
    return np.int32(x)
  else:
    return np.int64(x)

def signed(x):
  if type(x) == np.uint8:
    return np.int8(x)
  elif type(x) == np.uint16:
    return np.int16(x)
  elif type(x) == np.uint32:
    return np.int32(x)
  else:
    return np.int64(x)

def unsigned(x):
  if type(x) == np.int8:
    return np.uint8(x)
  elif type(x) == np.int16:
    return np.uint16(x)
  elif type(x) == np.int32:
    return np.uint32(x)
  else:
    return np.uint64(x)

def shlN(x,y):
  return x << y

def ashrN(x,y):
  return x >> y

# Python is so slow that we just make all the unsafe operations safe,
# always.

def sdivN(x,y):
  if y == 0:
    return intlit(type(x), 0)
  else:
    return x // y

def sdiv_upN(x,y):
  if y == 0:
    return intlit(type(x), 0)
  else:
    return (x+y-intlit(type(x), 1)) // y

def smodN(x,y):
  if y == 0:
    return intlit(type(x), 0)
  else:
    return x % y

def udivN(x,y):
  if y == 0:
    return intlit(type(x), 0)
  else:
    return signed(unsigned(x) // unsigned(y))

def udiv_upN(x,y):
  if y == 0:
    return intlit(type(x), 0)
  else:
    return signed((unsigned(x)+unsigned(y)-unsigned(intlit(type(x),1))) // unsigned(y))

def umodN(x,y):
  if y == 0:
    return intlit(type(x), 0)
  else:
    return signed(unsigned(x) % unsigned(y))

def squotN(x,y):
  if y == 0:
    return intlit(type(x), 0)
  else:
    return np.floor_divide(np.abs(x), np.abs(y)) * np.sign(x) * np.sign(y)

def sremN(x,y):
  if y == 0:
    return intlit(type(x), 0)
  else:
    return np.remainder(np.abs(x), np.abs(y)) * np.sign(x)

def sminN(x,y):
  return min(x,y)

def smaxN(x,y):
  return max(x,y)

def uminN(x,y):
  return signed(min(unsigned(x),unsigned(y)))

def umaxN(x,y):
  return signed(max(unsigned(x),unsigned(y)))

def fminN(x,y):
  return min(x,y)

def fmaxN(x,y):
  return max(x,y)

def powN(x,y):
  return x ** y

def fpowN(x,y):
  return x ** y

def sleN(x,y):
  return x <= y

def sltN(x,y):
  return x < y

def uleN(x,y):
  return unsigned(x) <= unsigned(y)

def ultN(x,y):
  return unsigned(x) < unsigned(y)

def lshr8(x,y):
  return np.int8(np.uint8(x) >> np.uint8(y))

def lshr16(x,y):
  return np.int16(np.uint16(x) >> np.uint16(y))

def lshr32(x,y):
  return np.int32(np.uint32(x) >> np.uint32(y))

def lshr64(x,y):
  return np.int64(np.uint64(x) >> np.uint64(y))

def sext_T_i8(x):
  return np.int8(x)

def sext_T_i16(x):
  return np.int16(x)

def sext_T_i32(x):
  return np.int32(x)

def sext_T_i64(x):
  return np.int64(x)

def itob_T_bool(x):
  return np.bool(x)

def btoi_bool_i8(x):
  return np.int8(x)

def btoi_bool_i16(x):
  return np.int8(x)

def btoi_bool_i32(x):
  return np.int8(x)

def btoi_bool_i64(x):
  return np.int8(x)

def zext_i8_i8(x):
  return np.int8(np.uint8(x))

def zext_i8_i16(x):
  return np.int16(np.uint8(x))

def zext_i8_i32(x):
  return np.int32(np.uint8(x))

def zext_i8_i64(x):
  return np.int64(np.uint8(x))

def zext_i16_i8(x):
  return np.int8(np.uint16(x))

def zext_i16_i16(x):
  return np.int16(np.uint16(x))

def zext_i16_i32(x):
  return np.int32(np.uint16(x))

def zext_i16_i64(x):
  return np.int64(np.uint16(x))

def zext_i32_i8(x):
  return np.int8(np.uint32(x))

def zext_i32_i16(x):
  return np.int16(np.uint32(x))

def zext_i32_i32(x):
  return np.int32(np.uint32(x))

def zext_i32_i64(x):
  return np.int64(np.uint32(x))

def zext_i64_i8(x):
  return np.int8(np.uint64(x))

def zext_i64_i16(x):
  return np.int16(np.uint64(x))

def zext_i64_i32(x):
  return np.int32(np.uint64(x))

def zext_i64_i64(x):
  return np.int64(np.uint64(x))

sdiv8 = sdiv16 = sdiv32 = sdiv64 = sdivN
sdiv_up8 = sdiv1_up6 = sdiv_up32 = sdiv_up64 = sdiv_upN
sdiv_safe8 = sdiv1_safe6 = sdiv_safe32 = sdiv_safe64 = sdivN
sdiv_up_safe8 = sdiv_up1_safe6 = sdiv_up_safe32 = sdiv_up_safe64 = sdiv_upN
smod8 = smod16 = smod32 = smod64 = smodN
smod_safe8 = smod_safe16 = smod_safe32 = smod_safe64 = smodN
udiv8 = udiv16 = udiv32 = udiv64 = udivN
udiv_up8 = udiv_up16 = udiv_up32 = udiv_up64 = udivN
udiv_safe8 = udiv_safe16 = udiv_safe32 = udiv_safe64 = udiv_upN
udiv_up_safe8 = udiv_up_safe16 = udiv_up_safe32 = udiv_up_safe64 = udiv_upN
umod8 = umod16 = umod32 = umod64 = umodN
umod_safe8 = umod_safe16 = umod_safe32 = umod_safe64 = umodN
squot8 = squot16 = squot32 = squot64 = squotN
squot_safe8 = squot_safe16 = squot_safe32 = squot_safe64 = squotN
srem8 = srem16 = srem32 = srem64 = sremN
srem_safe8 = srem_safe16 = srem_safe32 = srem_safe64 = sremN

shl8 = shl16 = shl32 = shl64 = shlN
ashr8 = ashr16 = ashr32 = ashr64 = ashrN
smax8 = smax16 = smax32 = smax64 = smaxN
smin8 = smin16 = smin32 = smin64 = sminN
umax8 = umax16 = umax32 = umax64 = umaxN
umin8 = umin16 = umin32 = umin64 = uminN
pow8 = pow16 = pow32 = pow64 = powN
fpow32 = fpow64 = fpowN
fmax32 = fmax64 = fmaxN
fmin32 = fmin64 = fminN
sle8 = sle16 = sle32 = sle64 = sleN
slt8 = slt16 = slt32 = slt64 = sltN
ule8 = ule16 = ule32 = ule64 = uleN
ult8 = ult16 = ult32 = ult64 = ultN
sext_i8_i8 = sext_i16_i8 = sext_i32_i8 = sext_i64_i8 = sext_T_i8
sext_i8_i16 = sext_i16_i16 = sext_i32_i16 = sext_i64_i16 = sext_T_i16
sext_i8_i32 = sext_i16_i32 = sext_i32_i32 = sext_i64_i32 = sext_T_i32
sext_i8_i64 = sext_i16_i64 = sext_i32_i64 = sext_i64_i64 = sext_T_i64
itob_i8_bool = itob_i16_bool = itob_i32_bool = itob_i64_bool = itob_T_bool

def clz_T(x):
  n = np.int32(0)
  bits = x.itemsize * 8
  for i in range(bits):
    if x < 0:
      break
    n += 1
    x <<= np.int8(1)
  return n

def ctz_T(x):
  n = np.int32(0)
  bits = x.itemsize * 8
  for i in range(bits):
    if (x & 1) == 1:
      break
    n += 1
    x >>= np.int8(1)
  return n

def popc_T(x):
  c = np.int32(0)
  while x != 0:
    x &= x - np.int8(1)
    c += np.int8(1)
  return c

futhark_popc8 = futhark_popc16 = futhark_popc32 = futhark_popc64 = popc_T
futhark_clzz8 = futhark_clzz16 = futhark_clzz32 = futhark_clzz64 = clz_T
futhark_ctzz8 = futhark_ctzz16 = futhark_ctzz32 = futhark_ctzz64 = ctz_T

def ssignum(x):
  return np.sign(x)

def usignum(x):
  if x < 0:
    return ssignum(-x)
  else:
    return ssignum(x)

def sitofp_T_f32(x):
  return np.float32(x)
sitofp_i8_f32 = sitofp_i16_f32 = sitofp_i32_f32 = sitofp_i64_f32 = sitofp_T_f32

def sitofp_T_f64(x):
  return np.float64(x)
sitofp_i8_f64 = sitofp_i16_f64 = sitofp_i32_f64 = sitofp_i64_f64 = sitofp_T_f64

def uitofp_T_f32(x):
  return np.float32(unsigned(x))
uitofp_i8_f32 = uitofp_i16_f32 = uitofp_i32_f32 = uitofp_i64_f32 = uitofp_T_f32

def uitofp_T_f64(x):
  return np.float64(unsigned(x))
uitofp_i8_f64 = uitofp_i16_f64 = uitofp_i32_f64 = uitofp_i64_f64 = uitofp_T_f64

def fptosi_T_i8(x):
  return np.int8(np.trunc(x))
fptosi_f32_i8 = fptosi_f64_i8 = fptosi_T_i8

def fptosi_T_i16(x):
  return np.int16(np.trunc(x))
fptosi_f32_i16 = fptosi_f64_i16 = fptosi_T_i16

def fptosi_T_i32(x):
  return np.int32(np.trunc(x))
fptosi_f32_i32 = fptosi_f64_i32 = fptosi_T_i32

def fptosi_T_i64(x):
  return np.int64(np.trunc(x))
fptosi_f32_i64 = fptosi_f64_i64 = fptosi_T_i64

def fptoui_T_i8(x):
  return np.uint8(np.trunc(x))
fptoui_f32_i8 = fptoui_f64_i8 = fptoui_T_i8

def fptoui_T_i16(x):
  return np.uint16(np.trunc(x))
fptoui_f32_i16 = fptoui_f64_i16 = fptoui_T_i16

def fptoui_T_i32(x):
  return np.uint32(np.trunc(x))
fptoui_f32_i32 = fptoui_f64_i32 = fptoui_T_i32

def fptoui_T_i64(x):
  return np.uint64(np.trunc(x))
fptoui_f32_i64 = fptoui_f64_i64 = fptoui_T_i64

def fpconv_f32_f64(x):
  return np.float64(x)

def fpconv_f64_f32(x):
  return np.float32(x)

def futhark_mul_hi8(a, b):
  a = np.uint64(np.uint8(a))
  b = np.uint64(np.uint8(b))
  return np.int8((a*b) >> np.uint64(8))

def futhark_mul_hi16(a, b):
  a = np.uint64(np.uint16(a))
  b = np.uint64(np.uint16(b))
  return np.int16((a*b) >> np.uint64(16))

def futhark_mul_hi32(a, b):
  a = np.uint64(np.uint32(a))
  b = np.uint64(np.uint32(b))
  return np.int32((a*b) >> np.uint64(32))

# This one is done with arbitrary-precision integers.
def futhark_mul_hi64(a, b):
  a = int(np.uint64(a))
  b = int(np.uint64(b))
  return np.int64(np.uint64(a*b >> 64))

def futhark_mad_hi8(a, b, c):
  return futhark_mul_hi8(a,b) + c

def futhark_mad_hi16(a, b, c):
  return futhark_mul_hi16(a,b) + c

def futhark_mad_hi32(a, b, c):
  return futhark_mul_hi32(a,b) + c

def futhark_mad_hi64(a, b, c):
  return futhark_mul_hi64(a,b) + c

def futhark_log64(x):
  return np.float64(np.log(x))

def futhark_log2_64(x):
  return np.float64(np.log2(x))

def futhark_log10_64(x):
  return np.float64(np.log10(x))

def futhark_sqrt64(x):
  return np.sqrt(x)

def futhark_exp64(x):
  return np.exp(x)

def futhark_cos64(x):
  return np.cos(x)

def futhark_sin64(x):
  return np.sin(x)

def futhark_tan64(x):
  return np.tan(x)

def futhark_acos64(x):
  return np.arccos(x)

def futhark_asin64(x):
  return np.arcsin(x)

def futhark_atan64(x):
  return np.arctan(x)

def futhark_cosh64(x):
  return np.cosh(x)

def futhark_sinh64(x):
  return np.sinh(x)

def futhark_tanh64(x):
  return np.tanh(x)

def futhark_acosh64(x):
  return np.arccosh(x)

def futhark_asinh64(x):
  return np.arcsinh(x)

def futhark_atanh64(x):
  return np.arctanh(x)

def futhark_atan2_64(x, y):
  return np.arctan2(x, y)

def futhark_gamma64(x):
  return np.float64(math.gamma(x))

def futhark_lgamma64(x):
  return np.float64(math.lgamma(x))

def futhark_round64(x):
  return np.round(x)

def futhark_ceil64(x):
  return np.ceil(x)

def futhark_floor64(x):
  return np.floor(x)

def futhark_isnan64(x):
  return np.isnan(x)

def futhark_isinf64(x):
  return np.isinf(x)

def futhark_to_bits64(x):
  s = struct.pack('>d', x)
  return np.int64(struct.unpack('>q', s)[0])

def futhark_from_bits64(x):
  s = struct.pack('>q', x)
  return np.float64(struct.unpack('>d', s)[0])

def futhark_log32(x):
  return np.float32(np.log(x))

def futhark_log2_32(x):
  return np.float32(np.log2(x))

def futhark_log10_32(x):
  return np.float32(np.log10(x))

def futhark_sqrt32(x):
  return np.float32(np.sqrt(x))

def futhark_exp32(x):
  return np.exp(x)

def futhark_cos32(x):
  return np.cos(x)

def futhark_sin32(x):
  return np.sin(x)

def futhark_tan32(x):
  return np.tan(x)

def futhark_acos32(x):
  return np.arccos(x)

def futhark_asin32(x):
  return np.arcsin(x)

def futhark_atan32(x):
  return np.arctan(x)

def futhark_cosh32(x):
  return np.cosh(x)

def futhark_sinh32(x):
  return np.sinh(x)

def futhark_tanh32(x):
  return np.tanh(x)

def futhark_acosh32(x):
  return np.arccosh(x)

def futhark_asinh32(x):
  return np.arcsinh(x)

def futhark_atanh32(x):
  return np.arctanh(x)

def futhark_atan2_32(x, y):
  return np.arctan2(x, y)

def futhark_gamma32(x):
  return np.float32(math.gamma(x))

def futhark_lgamma32(x):
  return np.float32(math.lgamma(x))

def futhark_round32(x):
  return np.round(x)

def futhark_ceil32(x):
  return np.ceil(x)

def futhark_floor32(x):
  return np.floor(x)

def futhark_isnan32(x):
  return np.isnan(x)

def futhark_isinf32(x):
  return np.isinf(x)

def futhark_to_bits32(x):
  s = struct.pack('>f', x)
  return np.int32(struct.unpack('>l', s)[0])

def futhark_from_bits32(x):
  s = struct.pack('>l', x)
  return np.float32(struct.unpack('>f', s)[0])

def futhark_lerp32(v0, v1, t):
  return v0 + (v1-v0)*t

def futhark_lerp64(v0, v1, t):
  return v0 + (v1-v0)*t

def futhark_mad32(a, b, c):
  return a * b + c

def futhark_mad64(a, b, c):
  return a * b + c

def futhark_fma32(a, b, c):
  return a * b + c

def futhark_fma64(a, b, c):
  return a * b + c

# End of scalar.py.
# Start of server.py

import sys
import time

class Server:
    def __init__(self, ctx):
        self._ctx = ctx
        self._vars = {}

    class Failure(BaseException):
        def __init__(self, msg):
            self.msg = msg

    def _get_arg(self, args, i):
        if i < len(args):
            return args[i]
        else:
            raise self.Failure('Insufficient command args')

    def _get_entry_point(self, entry):
        if entry in self._ctx.entry_points:
            return self._ctx.entry_points[entry]
        else:
            raise self.Failure('Unknown entry point: %s' % entry)

    def _check_var(self, vname):
        if not vname in self._vars:
            raise self.Failure('Unknown variable: %s' % vname)

    def _get_var(self, vname):
        self._check_var(vname)
        return self._vars[vname]

    def _cmd_inputs(self, args):
        entry = self._get_arg(args, 0)
        for t in self._get_entry_point(entry)[0]:
            print(t)

    def _cmd_outputs(self, args):
        entry = self._get_arg(args, 0)
        for t in self._get_entry_point(entry)[1]:
            print(t)

    def _cmd_dummy(self, args):
        pass

    def _cmd_free(self, args):
        for vname in args:
            self._check_var(vname)
            del self._vars[vname]

    def _cmd_call(self, args):
        entry = self._get_entry_point(self._get_arg(args, 0))
        num_ins = len(entry[0])
        num_outs = len(entry[1])
        exp_len = 1 + num_outs + num_ins

        if len(args) != exp_len:
            raise self.Failure('Invalid argument count, expected %d' % exp_len)

        out_vnames = args[1:num_outs+1]

        for out_vname in out_vnames:
            if out_vname in self._vars:
                raise self.Failure('Variable already exists: %s' % out_vname)

        in_vnames = args[1+num_outs:]
        ins = [ self._get_var(in_vname) for in_vname in in_vnames ]

        try:
            (runtime, vals) = getattr(self._ctx, args[0])(*ins)
        except Exception as e:
            raise self.Failure(str(e))

        print('runtime: %d' % runtime)

        if num_outs == 1:
            self._vars[out_vnames[0]] = vals
        else:
            for (out_vname, val) in zip(out_vnames, vals):
                self._vars[out_vname] = val

    def _cmd_store(self, args):
        fname = self._get_arg(args, 0)

        with open(fname, 'wb') as f:
            for i in range(1, len(args)):
                vname = args[i]
                value = self._get_var(vname)
                # In case we are using the PyOpenCL backend, we first
                # need to convert OpenCL arrays to ordinary NumPy
                # arrays.  We do this in a nasty way.
                if isinstance(value, np.number) or isinstance(value, np.bool) or isinstance(value, np.bool_) or isinstance(value, np.ndarray):
                    # Ordinary NumPy value.
                    f.write(construct_binary_value(self._vars[vname]))
                else:
                    # Assuming PyOpenCL array.
                    f.write(construct_binary_value(self._vars[vname].get()))

    def _cmd_restore(self, args):
        if len(args) % 2 == 0:
            raise self.Failure('Invalid argument count')

        fname = args[0]
        args = args[1:]

        with open(fname, 'rb') as f:
            reader = ReaderInput(f)
            while args != []:
                vname = args[0]
                typename = args[1]
                args = args[2:]

                if vname in self._vars:
                    raise self.Failure('Variable already exists: %s' % vname)

                try:
                    self._vars[vname] = read_value(typename, reader)
                except ValueError:
                    raise self.Failure('Failed to restore variable %s.\n'
                                       'Possibly malformed data in %s.\n'
                                       % (vname, fname))

            skip_spaces(reader)
            if reader.get_char() != b'':
                raise self.Failure('Expected EOF after reading values')

    _commands = { 'inputs': _cmd_inputs,
                  'outputs': _cmd_outputs,
                  'call': _cmd_call,
                  'restore': _cmd_restore,
                  'store': _cmd_store,
                  'free': _cmd_free,
                  'clear': _cmd_dummy,
                  'pause_profiling': _cmd_dummy,
                  'unpause_profiling': _cmd_dummy,
                  'report': _cmd_dummy
                 }

    def _process_line(self, line):
        words = line.split()
        if words == []:
            raise self.Failure('Empty line')
        else:
            cmd = words[0]
            args = words[1:]
            if cmd in self._commands:
                self._commands[cmd](self, args)
            else:
                raise self.Failure('Unknown command: %s' % cmd)

    def run(self):
        while True:
            print('%%% OK', flush=True)
            line = sys.stdin.readline()
            if line == '':
                return
            try:
                self._process_line(line)
            except self.Failure as e:
                print('%%% FAILURE')
                print(e.msg)

# End of server.py
class lotka_volterra:
  entry_points = {"main": (["f32", "i64", "f32", "f32", "f32", "f32", "f32",
                            "f32"], ["[][]f32"]), "runge_kutta_fwd": (["f32",
                                                                       "i64",
                                                                       "f32",
                                                                       "f32",
                                                                       "f32",
                                                                       "f32",
                                                                       "f32",
                                                                       "f32",
                                                                       "f32",
                                                                       "f32",
                                                                       "f32",
                                                                       "f32",
                                                                       "f32",
                                                                       "f32"],
                                                                      ["[][]f32"])}
  def __init__(self, command_queue=None, interactive=False,
               platform_pref=preferred_platform, device_pref=preferred_device,
               default_group_size=default_group_size,
               default_num_groups=default_num_groups,
               default_tile_size=default_tile_size,
               default_reg_tile_size=default_reg_tile_size,
               default_threshold=default_threshold, sizes=sizes):
    size_heuristics=[("NVIDIA CUDA", cl.device_type.GPU, "lockstep_width",
      lambda device: np.int32(32)), ("AMD Accelerated Parallel Processing",
                                     cl.device_type.GPU, "lockstep_width",
                                     lambda device: np.int32(32)), ("",
                                                                    cl.device_type.GPU,
                                                                    "lockstep_width",
                                                                    lambda device: np.int32(1)),
     ("", cl.device_type.GPU, "num_groups",
      lambda device: (np.int32(4) * device.get_info(getattr(cl.device_info,
                                                            "MAX_COMPUTE_UNITS")))),
     ("", cl.device_type.GPU, "group_size", lambda device: np.int32(256)), ("",
                                                                            cl.device_type.GPU,
                                                                            "tile_size",
                                                                            lambda device: np.int32(32)),
     ("", cl.device_type.GPU, "reg_tile_size", lambda device: np.int32(2)), ("",
                                                                             cl.device_type.GPU,
                                                                             "threshold",
                                                                             lambda device: np.int32(32768)),
     ("", cl.device_type.CPU, "lockstep_width", lambda device: np.int32(1)), ("",
                                                                              cl.device_type.CPU,
                                                                              "num_groups",
                                                                              lambda device: device.get_info(getattr(cl.device_info,
                                                                                                                     "MAX_COMPUTE_UNITS"))),
     ("", cl.device_type.CPU, "group_size", lambda device: np.int32(32)), ("",
                                                                           cl.device_type.CPU,
                                                                           "tile_size",
                                                                           lambda device: np.int32(4)),
     ("", cl.device_type.CPU, "reg_tile_size", lambda device: np.int32(1)), ("",
                                                                             cl.device_type.CPU,
                                                                             "threshold",
                                                                             lambda device: device.get_info(getattr(cl.device_info,
                                                                                                                    "MAX_COMPUTE_UNITS")))]
    self.global_failure_args_max = 0
    self.failure_msgs=[]
    program = initialise_opencl_object(self,
                                       program_src=fut_opencl_src,
                                       command_queue=command_queue,
                                       interactive=interactive,
                                       platform_pref=platform_pref,
                                       device_pref=device_pref,
                                       default_group_size=default_group_size,
                                       default_num_groups=default_num_groups,
                                       default_tile_size=default_tile_size,
                                       default_reg_tile_size=default_reg_tile_size,
                                       default_threshold=default_threshold,
                                       size_heuristics=size_heuristics,
                                       required_types=["i32", "i64", "f32"],
                                       user_sizes=sizes,
                                       all_sizes={"builtin#replicate_f32.group_size_6149": {"class": "group_size",
                                                                                  "value": None},
                                        "main.segmap_group_size_6044": {"class": "group_size", "value": None},
                                        "main.segmap_num_groups_6046": {"class": "num_groups", "value": None},
                                        "runge_kutta_fwd.segmap_group_size_6084": {"class": "group_size",
                                                                                   "value": None},
                                        "runge_kutta_fwd.segmap_num_groups_6086": {"class": "num_groups",
                                                                                   "value": None}})
    self.builtinzhreplicate_f32zireplicate_6146_var = program.builtinzhreplicate_f32zireplicate_6146
    self.gpu_map_transpose_f32_var = program.gpu_map_transpose_f32
    self.gpu_map_transpose_f32_low_height_var = program.gpu_map_transpose_f32_low_height
    self.gpu_map_transpose_f32_low_width_var = program.gpu_map_transpose_f32_low_width
    self.gpu_map_transpose_f32_small_var = program.gpu_map_transpose_f32_small
    self.mainzisegmap_6042_var = program.mainzisegmap_6042
    self.runge_kutta_fwdzisegmap_6082_var = program.runge_kutta_fwdzisegmap_6082
    self.constants = {}
  def futhark_builtinzhgpu_map_transpose_f32(self, destmem_0, destoffset_1,
                                             srcmem_2, srcoffset_3,
                                             num_arrays_4, x_elems_5,
                                             y_elems_6):
    if ((num_arrays_4 == np.int32(0)) or ((x_elems_5 == np.int32(0)) or (y_elems_6 == np.int32(0)))):
      pass
    else:
      muly_8 = squot32(np.int32(16), x_elems_5)
      mulx_7 = squot32(np.int32(16), y_elems_6)
      if ((num_arrays_4 == np.int32(1)) and ((x_elems_5 == np.int32(1)) or (y_elems_6 == np.int32(1)))):
        if (sext_i32_i64(((x_elems_5 * y_elems_6) * np.int32(4))) != 0):
          cl.enqueue_copy(self.queue, destmem_0, srcmem_2,
                          dest_offset=np.long(sext_i32_i64(destoffset_1)),
                          src_offset=np.long(sext_i32_i64(srcoffset_3)),
                          byte_count=np.long(sext_i32_i64(((x_elems_5 * y_elems_6) * np.int32(4)))))
        if synchronous:
          sync(self)
      else:
        if (sle32(x_elems_5, np.int32(8)) and slt32(np.int32(16), y_elems_6)):
          if ((((1 * (np.long(sdiv_up32(x_elems_5,
                                        np.int32(16))) * np.long(np.int32(16)))) * (np.long(sdiv_up32(sdiv_up32(y_elems_6,
                                                                                                                muly_8),
                                                                                                      np.int32(16))) * np.long(np.int32(16)))) * (np.long(num_arrays_4) * np.long(np.int32(1)))) != 0):
            self.gpu_map_transpose_f32_low_width_var.set_args(cl.LocalMemory(np.long(np.int64(1088))),
                                                              np.int32(destoffset_1),
                                                              np.int32(srcoffset_3),
                                                              np.int32(num_arrays_4),
                                                              np.int32(x_elems_5),
                                                              np.int32(y_elems_6),
                                                              np.int32(mulx_7),
                                                              np.int32(muly_8),
                                                              destmem_0,
                                                              srcmem_2)
            cl.enqueue_nd_range_kernel(self.queue,
                                       self.gpu_map_transpose_f32_low_width_var,
                                       ((np.long(sdiv_up32(x_elems_5,
                                                           np.int32(16))) * np.long(np.int32(16))),
                                        (np.long(sdiv_up32(sdiv_up32(y_elems_6,
                                                                     muly_8),
                                                           np.int32(16))) * np.long(np.int32(16))),
                                        (np.long(num_arrays_4) * np.long(np.int32(1)))),
                                       (np.long(np.int32(16)),
                                        np.long(np.int32(16)),
                                        np.long(np.int32(1))))
            if synchronous:
              sync(self)
        else:
          if (sle32(y_elems_6, np.int32(8)) and slt32(np.int32(16), x_elems_5)):
            if ((((1 * (np.long(sdiv_up32(sdiv_up32(x_elems_5, mulx_7),
                                          np.int32(16))) * np.long(np.int32(16)))) * (np.long(sdiv_up32(y_elems_6,
                                                                                                        np.int32(16))) * np.long(np.int32(16)))) * (np.long(num_arrays_4) * np.long(np.int32(1)))) != 0):
              self.gpu_map_transpose_f32_low_height_var.set_args(cl.LocalMemory(np.long(np.int64(1088))),
                                                                 np.int32(destoffset_1),
                                                                 np.int32(srcoffset_3),
                                                                 np.int32(num_arrays_4),
                                                                 np.int32(x_elems_5),
                                                                 np.int32(y_elems_6),
                                                                 np.int32(mulx_7),
                                                                 np.int32(muly_8),
                                                                 destmem_0,
                                                                 srcmem_2)
              cl.enqueue_nd_range_kernel(self.queue,
                                         self.gpu_map_transpose_f32_low_height_var,
                                         ((np.long(sdiv_up32(sdiv_up32(x_elems_5,
                                                                       mulx_7),
                                                             np.int32(16))) * np.long(np.int32(16))),
                                          (np.long(sdiv_up32(y_elems_6,
                                                             np.int32(16))) * np.long(np.int32(16))),
                                          (np.long(num_arrays_4) * np.long(np.int32(1)))),
                                         (np.long(np.int32(16)),
                                          np.long(np.int32(16)),
                                          np.long(np.int32(1))))
              if synchronous:
                sync(self)
          else:
            if (sle32(x_elems_5, np.int32(8)) and sle32(y_elems_6,
                                                        np.int32(8))):
              if ((1 * (np.long(sdiv_up32(((num_arrays_4 * x_elems_5) * y_elems_6),
                                          np.int32(256))) * np.long(np.int32(256)))) != 0):
                self.gpu_map_transpose_f32_small_var.set_args(cl.LocalMemory(np.long(np.int64(1))),
                                                              np.int32(destoffset_1),
                                                              np.int32(srcoffset_3),
                                                              np.int32(num_arrays_4),
                                                              np.int32(x_elems_5),
                                                              np.int32(y_elems_6),
                                                              np.int32(mulx_7),
                                                              np.int32(muly_8),
                                                              destmem_0,
                                                              srcmem_2)
                cl.enqueue_nd_range_kernel(self.queue,
                                           self.gpu_map_transpose_f32_small_var,
                                           ((np.long(sdiv_up32(((num_arrays_4 * x_elems_5) * y_elems_6),
                                                               np.int32(256))) * np.long(np.int32(256))),),
                                           (np.long(np.int32(256)),))
                if synchronous:
                  sync(self)
            else:
              if ((((1 * (np.long(sdiv_up32(x_elems_5,
                                            np.int32(32))) * np.long(np.int32(32)))) * (np.long(sdiv_up32(y_elems_6,
                                                                                                          np.int32(32))) * np.long(np.int32(8)))) * (np.long(num_arrays_4) * np.long(np.int32(1)))) != 0):
                self.gpu_map_transpose_f32_var.set_args(cl.LocalMemory(np.long(np.int64(4224))),
                                                        np.int32(destoffset_1),
                                                        np.int32(srcoffset_3),
                                                        np.int32(num_arrays_4),
                                                        np.int32(x_elems_5),
                                                        np.int32(y_elems_6),
                                                        np.int32(mulx_7),
                                                        np.int32(muly_8),
                                                        destmem_0, srcmem_2)
                cl.enqueue_nd_range_kernel(self.queue,
                                           self.gpu_map_transpose_f32_var,
                                           ((np.long(sdiv_up32(x_elems_5,
                                                               np.int32(32))) * np.long(np.int32(32))),
                                            (np.long(sdiv_up32(y_elems_6,
                                                               np.int32(32))) * np.long(np.int32(8))),
                                            (np.long(num_arrays_4) * np.long(np.int32(1)))),
                                           (np.long(np.int32(32)),
                                            np.long(np.int32(8)),
                                            np.long(np.int32(1))))
                if synchronous:
                  sync(self)
    return ()
  def futhark_builtinzhreplicate_f32(self, mem_6142, num_elems_6143, val_6144):
    group_sizze_6149 = self.sizes["builtin#replicate_f32.group_size_6149"]
    num_groups_6150 = sdiv_up64(num_elems_6143, group_sizze_6149)
    if ((1 * (np.long(num_groups_6150) * np.long(group_sizze_6149))) != 0):
      self.builtinzhreplicate_f32zireplicate_6146_var.set_args(np.int32(num_elems_6143),
                                                               np.float32(val_6144),
                                                               mem_6142)
      cl.enqueue_nd_range_kernel(self.queue,
                                 self.builtinzhreplicate_f32zireplicate_6146_var,
                                 ((np.long(num_groups_6150) * np.long(group_sizze_6149)),),
                                 (np.long(group_sizze_6149),))
      if synchronous:
        sync(self)
    return ()
  def futhark_main(self, step_sizze_5690, num_steps_5691, init_prey_5692,
                   init_pred_5693, growth_prey_5694, predation_5695,
                   growth_pred_5696, decline_pred_5697):
    bounds_invalid_upwards_5698 = slt64(num_steps_5691, np.int64(0))
    valid_5699 = not(bounds_invalid_upwards_5698)
    range_valid_c_5700 = True
    assert valid_5699, ("Error: %s%d%s%d%s%d%s\n\nBacktrace:\n-> #0  /prelude/array.fut:90:3-10\n   #1  /prelude/array.fut:108:18-23\n   #2  lotka_volterra.fut:54:1-62:122\n" % ("Range ",
                                                                                                                                                                                  np.int64(0),
                                                                                                                                                                                  "..",
                                                                                                                                                                                  np.int64(1),
                                                                                                                                                                                  "..<",
                                                                                                                                                                                  num_steps_5691,
                                                                                                                                                                                  " is invalid."))
    bytes_6101 = (np.int64(4) * num_steps_5691)
    mem_6102 = opencl_alloc(self, bytes_6101, "mem_6102")
    self.futhark_builtinzhreplicate_f32(mem_6102, num_steps_5691,
                                        init_prey_5692)
    mem_6104 = opencl_alloc(self, bytes_6101, "mem_6104")
    self.futhark_builtinzhreplicate_f32(mem_6104, num_steps_5691,
                                        init_pred_5693)
    x_5703 = (step_sizze_5690 / np.float32(2.0))
    x_5704 = (step_sizze_5690 / np.float32(6.0))
    curr_state_5710 = init_prey_5692
    curr_state_5711 = init_pred_5693
    i_5709 = np.int64(0)
    one_6168 = np.int64(1)
    for counter_6167 in range(num_steps_5691):
      y_5714 = (predation_5695 * curr_state_5711)
      x_5715 = (growth_prey_5694 - y_5714)
      dprey_5716 = (curr_state_5710 * x_5715)
      x_5717 = (growth_pred_5696 * curr_state_5710)
      x_5718 = (x_5717 - decline_pred_5697)
      dpred_5719 = (curr_state_5711 * x_5718)
      y_5720 = (x_5703 * dprey_5716)
      defunc_1_fn_arg_5721 = (curr_state_5711 + y_5720)
      defunc_0_fn_arg_5722 = (curr_state_5710 + y_5720)
      y_5723 = (predation_5695 * defunc_1_fn_arg_5721)
      x_5724 = (growth_prey_5694 - y_5723)
      dprey_5725 = (defunc_0_fn_arg_5722 * x_5724)
      x_5726 = (growth_pred_5696 * defunc_0_fn_arg_5722)
      x_5727 = (x_5726 - decline_pred_5697)
      dpred_5728 = (defunc_1_fn_arg_5721 * x_5727)
      y_5729 = (x_5703 * dprey_5725)
      defunc_1_fn_arg_5730 = (curr_state_5711 + y_5729)
      defunc_0_fn_arg_5731 = (curr_state_5710 + y_5729)
      y_5732 = (predation_5695 * defunc_1_fn_arg_5730)
      x_5733 = (growth_prey_5694 - y_5732)
      dprey_5734 = (defunc_0_fn_arg_5731 * x_5733)
      x_5735 = (growth_pred_5696 * defunc_0_fn_arg_5731)
      x_5736 = (x_5735 - decline_pred_5697)
      dpred_5737 = (defunc_1_fn_arg_5730 * x_5736)
      y_5738 = (step_sizze_5690 * dprey_5734)
      defunc_1_fn_arg_5739 = (curr_state_5711 + y_5738)
      defunc_0_fn_arg_5740 = (curr_state_5710 + y_5738)
      y_5741 = (predation_5695 * defunc_1_fn_arg_5739)
      x_5742 = (growth_prey_5694 - y_5741)
      dprey_5743 = (defunc_0_fn_arg_5740 * x_5742)
      x_5744 = (growth_pred_5696 * defunc_0_fn_arg_5740)
      x_5745 = (x_5744 - decline_pred_5697)
      dpred_5746 = (defunc_1_fn_arg_5739 * x_5745)
      y_5747 = (np.float32(2.0) * dprey_5725)
      x_5748 = (dprey_5716 + y_5747)
      y_5749 = (np.float32(2.0) * dprey_5734)
      x_5750 = (x_5748 + y_5749)
      y_5751 = (dprey_5743 + x_5750)
      y_5752 = (x_5704 * y_5751)
      loopres_5753 = (curr_state_5710 + y_5752)
      y_5754 = (np.float32(2.0) * dpred_5728)
      x_5755 = (dpred_5719 + y_5754)
      y_5756 = (np.float32(2.0) * dpred_5737)
      x_5757 = (x_5755 + y_5756)
      y_5758 = (dpred_5746 + x_5757)
      y_5759 = (x_5704 * y_5758)
      loopres_5760 = (curr_state_5711 + y_5759)
      cl.enqueue_copy(self.queue, mem_6102, np.array(loopres_5753,
                                                     dtype=ct.c_float),
                      device_offset=(np.long(i_5709) * 4),
                      is_blocking=synchronous)
      cl.enqueue_copy(self.queue, mem_6104, np.array(loopres_5760,
                                                     dtype=ct.c_float),
                      device_offset=(np.long(i_5709) * 4),
                      is_blocking=synchronous)
      curr_state_tmp_6151 = loopres_5753
      curr_state_tmp_6152 = loopres_5760
      curr_state_5710 = curr_state_tmp_6151
      curr_state_5711 = curr_state_tmp_6152
      i_5709 += one_6168
    states_5705 = curr_state_5710
    states_5706 = curr_state_5711
    segmap_group_sizze_6053 = self.sizes["main.segmap_group_size_6044"]
    max_num_groups_6155 = self.sizes["main.segmap_num_groups_6046"]
    num_groups_6054 = sext_i64_i32(smax64(np.int64(1),
                                          smin64(sdiv_up64(num_steps_5691,
                                                           segmap_group_sizze_6053),
                                                 sext_i32_i64(max_num_groups_6155))))
    binop_x_6133 = (np.int64(2) * num_steps_5691)
    bytes_6132 = (np.int64(4) * binop_x_6133)
    mem_6134 = opencl_alloc(self, bytes_6132, "mem_6134")
    if ((1 * (np.long(num_groups_6054) * np.long(segmap_group_sizze_6053))) != 0):
      self.mainzisegmap_6042_var.set_args(self.global_failure,
                                          np.int64(num_steps_5691),
                                          np.int64(num_groups_6054), mem_6102,
                                          mem_6104, mem_6134)
      cl.enqueue_nd_range_kernel(self.queue, self.mainzisegmap_6042_var,
                                 ((np.long(num_groups_6054) * np.long(segmap_group_sizze_6053)),),
                                 (np.long(segmap_group_sizze_6053),))
      if synchronous:
        sync(self)
    mem_6102 = None
    mem_6104 = None
    mem_6137 = opencl_alloc(self, bytes_6132, "mem_6137")
    self.futhark_builtinzhgpu_map_transpose_f32(mem_6137, np.int64(0), mem_6134,
                                                np.int64(0), np.int64(1),
                                                num_steps_5691, np.int64(2))
    mem_6134 = None
    out_mem_6141 = mem_6137
    return out_mem_6141
  def futhark_runge_kutta_fwd(self, step_sizze_5767, num_steps_5768,
                              init_prey_5769, init_pred_5770, growth_prey_5771,
                              predation_5772, growth_pred_5773,
                              decline_pred_5774, init_prey_tan_5775,
                              init_pred_tan_5776, growth_prey_tan_5777,
                              predation_tan_5778, growth_pred_tan_5779,
                              decline_pred_tan_5780):
    bounds_invalid_upwards_5781 = slt64(num_steps_5768, np.int64(0))
    valid_5782 = not(bounds_invalid_upwards_5781)
    range_valid_c_5783 = True
    assert valid_5782, ("Error: %s%d%s%d%s%d%s\n\nBacktrace:\n-> #0  /prelude/array.fut:90:3-10\n   #1  /prelude/array.fut:108:18-23\n   #2  lotka_volterra.fut:64:1-92:35\n" % ("Range ",
                                                                                                                                                                                 np.int64(0),
                                                                                                                                                                                 "..",
                                                                                                                                                                                 np.int64(1),
                                                                                                                                                                                 "..<",
                                                                                                                                                                                 num_steps_5768,
                                                                                                                                                                                 " is invalid."))
    x_5784 = (step_sizze_5767 / np.float32(2.0))
    x_5785 = (step_sizze_5767 / np.float32(6.0))
    bytes_6101 = (np.int64(4) * num_steps_5768)
    mem_6102 = opencl_alloc(self, bytes_6101, "mem_6102")
    self.futhark_builtinzhreplicate_f32(mem_6102, num_steps_5768,
                                        init_prey_tan_5775)
    mem_6104 = opencl_alloc(self, bytes_6101, "mem_6104")
    self.futhark_builtinzhreplicate_f32(mem_6104, num_steps_5768,
                                        init_pred_tan_5776)
    binop_y_5888 = (np.float32(-1.0) * decline_pred_tan_5780)
    curr_state_5801 = init_prey_5769
    curr_state_tan_5870 = init_prey_tan_5775
    curr_state_5802 = init_pred_5770
    curr_state_tan_5871 = init_pred_tan_5776
    i_5800 = np.int64(0)
    one_6170 = np.int64(1)
    for counter_6169 in range(num_steps_5768):
      y_5805 = (predation_5772 * curr_state_5802)
      binop_x_5875 = (predation_tan_5778 * curr_state_5802)
      binop_y_5876 = (predation_5772 * curr_state_tan_5871)
      y_tan_5874 = (binop_x_5875 + binop_y_5876)
      x_5806 = (growth_prey_5771 - y_5805)
      binop_y_5879 = (np.float32(-1.0) * y_tan_5874)
      x_tan_5877 = (growth_prey_tan_5777 + binop_y_5879)
      dprey_5807 = (curr_state_5801 * x_5806)
      binop_x_5881 = (x_5806 * curr_state_tan_5870)
      binop_y_5882 = (curr_state_5801 * x_tan_5877)
      dprey_tan_5880 = (binop_x_5881 + binop_y_5882)
      x_5808 = (growth_pred_5773 * curr_state_5801)
      binop_x_5884 = (growth_pred_tan_5779 * curr_state_5801)
      binop_y_5885 = (growth_pred_5773 * curr_state_tan_5870)
      x_tan_5883 = (binop_x_5884 + binop_y_5885)
      x_5809 = (x_5808 - decline_pred_5774)
      x_tan_5886 = (x_tan_5883 + binop_y_5888)
      dpred_5810 = (curr_state_5802 * x_5809)
      binop_x_5890 = (x_5809 * curr_state_tan_5871)
      binop_y_5891 = (curr_state_5802 * x_tan_5886)
      dpred_tan_5889 = (binop_x_5890 + binop_y_5891)
      y_5811 = (x_5784 * dprey_5807)
      binop_y_5895 = (x_5784 * dprey_tan_5880)
      defunc_1_fn_arg_5812 = (curr_state_5802 + y_5811)
      defunc_1_fn_arg_tan_5896 = (curr_state_tan_5871 + binop_y_5895)
      defunc_0_fn_arg_5813 = (curr_state_5801 + y_5811)
      defunc_0_fn_arg_tan_5899 = (curr_state_tan_5870 + binop_y_5895)
      y_5814 = (predation_5772 * defunc_1_fn_arg_5812)
      binop_x_5903 = (predation_tan_5778 * defunc_1_fn_arg_5812)
      binop_y_5904 = (predation_5772 * defunc_1_fn_arg_tan_5896)
      y_tan_5902 = (binop_x_5903 + binop_y_5904)
      x_5815 = (growth_prey_5771 - y_5814)
      binop_y_5907 = (np.float32(-1.0) * y_tan_5902)
      x_tan_5905 = (growth_prey_tan_5777 + binop_y_5907)
      dprey_5816 = (defunc_0_fn_arg_5813 * x_5815)
      binop_x_5909 = (x_5815 * defunc_0_fn_arg_tan_5899)
      binop_y_5910 = (defunc_0_fn_arg_5813 * x_tan_5905)
      dprey_tan_5908 = (binop_x_5909 + binop_y_5910)
      x_5817 = (growth_pred_5773 * defunc_0_fn_arg_5813)
      binop_x_5912 = (growth_pred_tan_5779 * defunc_0_fn_arg_5813)
      binop_y_5913 = (growth_pred_5773 * defunc_0_fn_arg_tan_5899)
      x_tan_5911 = (binop_x_5912 + binop_y_5913)
      x_5818 = (x_5817 - decline_pred_5774)
      x_tan_5914 = (binop_y_5888 + x_tan_5911)
      dpred_5819 = (defunc_1_fn_arg_5812 * x_5818)
      binop_x_5918 = (x_5818 * defunc_1_fn_arg_tan_5896)
      binop_y_5919 = (defunc_1_fn_arg_5812 * x_tan_5914)
      dpred_tan_5917 = (binop_x_5918 + binop_y_5919)
      y_5820 = (x_5784 * dprey_5816)
      binop_y_5922 = (x_5784 * dprey_tan_5908)
      defunc_1_fn_arg_5821 = (curr_state_5802 + y_5820)
      defunc_1_fn_arg_tan_5923 = (curr_state_tan_5871 + binop_y_5922)
      defunc_0_fn_arg_5822 = (curr_state_5801 + y_5820)
      defunc_0_fn_arg_tan_5926 = (curr_state_tan_5870 + binop_y_5922)
      y_5823 = (predation_5772 * defunc_1_fn_arg_5821)
      binop_x_5930 = (predation_tan_5778 * defunc_1_fn_arg_5821)
      binop_y_5931 = (predation_5772 * defunc_1_fn_arg_tan_5923)
      y_tan_5929 = (binop_x_5930 + binop_y_5931)
      x_5824 = (growth_prey_5771 - y_5823)
      binop_y_5934 = (np.float32(-1.0) * y_tan_5929)
      x_tan_5932 = (growth_prey_tan_5777 + binop_y_5934)
      dprey_5825 = (defunc_0_fn_arg_5822 * x_5824)
      binop_x_5936 = (x_5824 * defunc_0_fn_arg_tan_5926)
      binop_y_5937 = (defunc_0_fn_arg_5822 * x_tan_5932)
      dprey_tan_5935 = (binop_x_5936 + binop_y_5937)
      x_5826 = (growth_pred_5773 * defunc_0_fn_arg_5822)
      binop_x_5939 = (growth_pred_tan_5779 * defunc_0_fn_arg_5822)
      binop_y_5940 = (growth_pred_5773 * defunc_0_fn_arg_tan_5926)
      x_tan_5938 = (binop_x_5939 + binop_y_5940)
      x_5827 = (x_5826 - decline_pred_5774)
      x_tan_5941 = (binop_y_5888 + x_tan_5938)
      dpred_5828 = (defunc_1_fn_arg_5821 * x_5827)
      binop_x_5945 = (x_5827 * defunc_1_fn_arg_tan_5923)
      binop_y_5946 = (defunc_1_fn_arg_5821 * x_tan_5941)
      dpred_tan_5944 = (binop_x_5945 + binop_y_5946)
      y_5829 = (step_sizze_5767 * dprey_5825)
      binop_y_5950 = (step_sizze_5767 * dprey_tan_5935)
      defunc_1_fn_arg_5830 = (curr_state_5802 + y_5829)
      defunc_1_fn_arg_tan_5951 = (curr_state_tan_5871 + binop_y_5950)
      defunc_0_fn_arg_5831 = (curr_state_5801 + y_5829)
      defunc_0_fn_arg_tan_5954 = (curr_state_tan_5870 + binop_y_5950)
      y_5832 = (predation_5772 * defunc_1_fn_arg_5830)
      binop_x_5958 = (predation_tan_5778 * defunc_1_fn_arg_5830)
      binop_y_5959 = (predation_5772 * defunc_1_fn_arg_tan_5951)
      y_tan_5957 = (binop_x_5958 + binop_y_5959)
      x_5833 = (growth_prey_5771 - y_5832)
      binop_y_5962 = (np.float32(-1.0) * y_tan_5957)
      x_tan_5960 = (growth_prey_tan_5777 + binop_y_5962)
      dprey_5834 = (defunc_0_fn_arg_5831 * x_5833)
      binop_x_5964 = (x_5833 * defunc_0_fn_arg_tan_5954)
      binop_y_5965 = (defunc_0_fn_arg_5831 * x_tan_5960)
      dprey_tan_5963 = (binop_x_5964 + binop_y_5965)
      x_5835 = (growth_pred_5773 * defunc_0_fn_arg_5831)
      binop_x_5967 = (growth_pred_tan_5779 * defunc_0_fn_arg_5831)
      binop_y_5968 = (growth_pred_5773 * defunc_0_fn_arg_tan_5954)
      x_tan_5966 = (binop_x_5967 + binop_y_5968)
      x_5836 = (x_5835 - decline_pred_5774)
      x_tan_5969 = (binop_y_5888 + x_tan_5966)
      dpred_5837 = (defunc_1_fn_arg_5830 * x_5836)
      binop_x_5973 = (x_5836 * defunc_1_fn_arg_tan_5951)
      binop_y_5974 = (defunc_1_fn_arg_5830 * x_tan_5969)
      dpred_tan_5972 = (binop_x_5973 + binop_y_5974)
      y_5838 = (np.float32(2.0) * dprey_5816)
      binop_y_5977 = (np.float32(2.0) * dprey_tan_5908)
      x_5839 = (dprey_5807 + y_5838)
      x_tan_5978 = (dprey_tan_5880 + binop_y_5977)
      y_5840 = (np.float32(2.0) * dprey_5825)
      binop_y_5983 = (np.float32(2.0) * dprey_tan_5935)
      x_5841 = (x_5839 + y_5840)
      x_tan_5984 = (x_tan_5978 + binop_y_5983)
      y_5842 = (dprey_5834 + x_5841)
      y_tan_5987 = (dprey_tan_5963 + x_tan_5984)
      y_5843 = (x_5785 * y_5842)
      binop_y_5993 = (x_5785 * y_tan_5987)
      loopres_5844 = (curr_state_5801 + y_5843)
      loopres_tan_5994 = (curr_state_tan_5870 + binop_y_5993)
      y_5845 = (np.float32(2.0) * dpred_5819)
      binop_y_5999 = (np.float32(2.0) * dpred_tan_5917)
      x_5846 = (dpred_5810 + y_5845)
      x_tan_6000 = (dpred_tan_5889 + binop_y_5999)
      y_5847 = (np.float32(2.0) * dpred_5828)
      binop_y_6005 = (np.float32(2.0) * dpred_tan_5944)
      x_5848 = (x_5846 + y_5847)
      x_tan_6006 = (x_tan_6000 + binop_y_6005)
      y_5849 = (dpred_5837 + x_5848)
      y_tan_6009 = (dpred_tan_5972 + x_tan_6006)
      y_5850 = (x_5785 * y_5849)
      binop_y_6014 = (x_5785 * y_tan_6009)
      loopres_5851 = (curr_state_5802 + y_5850)
      loopres_tan_6015 = (curr_state_tan_5871 + binop_y_6014)
      cl.enqueue_copy(self.queue, mem_6102, np.array(loopres_tan_5994,
                                                     dtype=ct.c_float),
                      device_offset=(np.long(i_5800) * 4),
                      is_blocking=synchronous)
      cl.enqueue_copy(self.queue, mem_6104, np.array(loopres_tan_6015,
                                                     dtype=ct.c_float),
                      device_offset=(np.long(i_5800) * 4),
                      is_blocking=synchronous)
      curr_state_tmp_6151 = loopres_5844
      curr_state_tan_tmp_6152 = loopres_tan_5994
      curr_state_tmp_6153 = loopres_5851
      curr_state_tan_tmp_6154 = loopres_tan_6015
      curr_state_5801 = curr_state_tmp_6151
      curr_state_tan_5870 = curr_state_tan_tmp_6152
      curr_state_5802 = curr_state_tmp_6153
      curr_state_tan_5871 = curr_state_tan_tmp_6154
      i_5800 += one_6170
    states_5796 = curr_state_5801
    states_tan_5866 = curr_state_tan_5870
    states_5797 = curr_state_5802
    states_tan_5867 = curr_state_tan_5871
    segmap_group_sizze_6093 = self.sizes["runge_kutta_fwd.segmap_group_size_6084"]
    max_num_groups_6157 = self.sizes["runge_kutta_fwd.segmap_num_groups_6086"]
    num_groups_6094 = sext_i64_i32(smax64(np.int64(1),
                                          smin64(sdiv_up64(num_steps_5768,
                                                           segmap_group_sizze_6093),
                                                 sext_i32_i64(max_num_groups_6157))))
    binop_x_6133 = (np.int64(2) * num_steps_5768)
    bytes_6132 = (np.int64(4) * binop_x_6133)
    mem_6134 = opencl_alloc(self, bytes_6132, "mem_6134")
    if ((1 * (np.long(num_groups_6094) * np.long(segmap_group_sizze_6093))) != 0):
      self.runge_kutta_fwdzisegmap_6082_var.set_args(self.global_failure,
                                                     np.int64(num_steps_5768),
                                                     np.int64(num_groups_6094),
                                                     mem_6102, mem_6104,
                                                     mem_6134)
      cl.enqueue_nd_range_kernel(self.queue,
                                 self.runge_kutta_fwdzisegmap_6082_var,
                                 ((np.long(num_groups_6094) * np.long(segmap_group_sizze_6093)),),
                                 (np.long(segmap_group_sizze_6093),))
      if synchronous:
        sync(self)
    mem_6102 = None
    mem_6104 = None
    mem_6137 = opencl_alloc(self, bytes_6132, "mem_6137")
    self.futhark_builtinzhgpu_map_transpose_f32(mem_6137, np.int64(0), mem_6134,
                                                np.int64(0), np.int64(1),
                                                num_steps_5768, np.int64(2))
    mem_6134 = None
    out_mem_6141 = mem_6137
    return out_mem_6141
  def main(self, step_sizze_5690_ext, num_steps_5691_ext, init_prey_5692_ext,
           init_pred_5693_ext, growth_prey_5694_ext, predation_5695_ext,
           growth_pred_5696_ext, decline_pred_5697_ext):
    try:
      step_sizze_5690 = np.float32(ct.c_float(step_sizze_5690_ext))
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #0 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("f32",
                                                                                                                            type(step_sizze_5690_ext),
                                                                                                                            step_sizze_5690_ext))
    try:
      num_steps_5691 = np.int64(ct.c_int64(num_steps_5691_ext))
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #1 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("i64",
                                                                                                                            type(num_steps_5691_ext),
                                                                                                                            num_steps_5691_ext))
    try:
      init_prey_5692 = np.float32(ct.c_float(init_prey_5692_ext))
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #2 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("f32",
                                                                                                                            type(init_prey_5692_ext),
                                                                                                                            init_prey_5692_ext))
    try:
      init_pred_5693 = np.float32(ct.c_float(init_pred_5693_ext))
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #3 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("f32",
                                                                                                                            type(init_pred_5693_ext),
                                                                                                                            init_pred_5693_ext))
    try:
      growth_prey_5694 = np.float32(ct.c_float(growth_prey_5694_ext))
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #4 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("f32",
                                                                                                                            type(growth_prey_5694_ext),
                                                                                                                            growth_prey_5694_ext))
    try:
      predation_5695 = np.float32(ct.c_float(predation_5695_ext))
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #5 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("f32",
                                                                                                                            type(predation_5695_ext),
                                                                                                                            predation_5695_ext))
    try:
      growth_pred_5696 = np.float32(ct.c_float(growth_pred_5696_ext))
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #6 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("f32",
                                                                                                                            type(growth_pred_5696_ext),
                                                                                                                            growth_pred_5696_ext))
    try:
      decline_pred_5697 = np.float32(ct.c_float(decline_pred_5697_ext))
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #7 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("f32",
                                                                                                                            type(decline_pred_5697_ext),
                                                                                                                            decline_pred_5697_ext))
    time_start = time.time()
    with np.errstate(divide="ignore", over="ignore", under="ignore",
                     invalid="ignore"):
      out_mem_6141 = self.futhark_main(step_sizze_5690, num_steps_5691,
                                       init_prey_5692, init_pred_5693,
                                       growth_prey_5694, predation_5695,
                                       growth_pred_5696, decline_pred_5697)
    runtime = (int((time.time() * 1000000)) - int((time_start * 1000000)))
    sync(self)
    return cl.array.Array(self.queue, (num_steps_5691, np.int64(2)), ct.c_float,
                          data=out_mem_6141)
  def runge_kutta_fwd(self, step_sizze_5767_ext, num_steps_5768_ext,
                      init_prey_5769_ext, init_pred_5770_ext,
                      growth_prey_5771_ext, predation_5772_ext,
                      growth_pred_5773_ext, decline_pred_5774_ext,
                      init_prey_tan_5775_ext, init_pred_tan_5776_ext,
                      growth_prey_tan_5777_ext, predation_tan_5778_ext,
                      growth_pred_tan_5779_ext, decline_pred_tan_5780_ext):
    try:
      step_sizze_5767 = np.float32(ct.c_float(step_sizze_5767_ext))
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #0 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("f32",
                                                                                                                            type(step_sizze_5767_ext),
                                                                                                                            step_sizze_5767_ext))
    try:
      num_steps_5768 = np.int64(ct.c_int64(num_steps_5768_ext))
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #1 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("i64",
                                                                                                                            type(num_steps_5768_ext),
                                                                                                                            num_steps_5768_ext))
    try:
      init_prey_5769 = np.float32(ct.c_float(init_prey_5769_ext))
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #2 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("f32",
                                                                                                                            type(init_prey_5769_ext),
                                                                                                                            init_prey_5769_ext))
    try:
      init_pred_5770 = np.float32(ct.c_float(init_pred_5770_ext))
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #3 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("f32",
                                                                                                                            type(init_pred_5770_ext),
                                                                                                                            init_pred_5770_ext))
    try:
      growth_prey_5771 = np.float32(ct.c_float(growth_prey_5771_ext))
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #4 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("f32",
                                                                                                                            type(growth_prey_5771_ext),
                                                                                                                            growth_prey_5771_ext))
    try:
      predation_5772 = np.float32(ct.c_float(predation_5772_ext))
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #5 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("f32",
                                                                                                                            type(predation_5772_ext),
                                                                                                                            predation_5772_ext))
    try:
      growth_pred_5773 = np.float32(ct.c_float(growth_pred_5773_ext))
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #6 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("f32",
                                                                                                                            type(growth_pred_5773_ext),
                                                                                                                            growth_pred_5773_ext))
    try:
      decline_pred_5774 = np.float32(ct.c_float(decline_pred_5774_ext))
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #7 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("f32",
                                                                                                                            type(decline_pred_5774_ext),
                                                                                                                            decline_pred_5774_ext))
    try:
      init_prey_tan_5775 = np.float32(ct.c_float(init_prey_tan_5775_ext))
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #8 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("f32",
                                                                                                                            type(init_prey_tan_5775_ext),
                                                                                                                            init_prey_tan_5775_ext))
    try:
      init_pred_tan_5776 = np.float32(ct.c_float(init_pred_tan_5776_ext))
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #9 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("f32",
                                                                                                                            type(init_pred_tan_5776_ext),
                                                                                                                            init_pred_tan_5776_ext))
    try:
      growth_prey_tan_5777 = np.float32(ct.c_float(growth_prey_tan_5777_ext))
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #10 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("f32",
                                                                                                                             type(growth_prey_tan_5777_ext),
                                                                                                                             growth_prey_tan_5777_ext))
    try:
      predation_tan_5778 = np.float32(ct.c_float(predation_tan_5778_ext))
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #11 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("f32",
                                                                                                                             type(predation_tan_5778_ext),
                                                                                                                             predation_tan_5778_ext))
    try:
      growth_pred_tan_5779 = np.float32(ct.c_float(growth_pred_tan_5779_ext))
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #12 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("f32",
                                                                                                                             type(growth_pred_tan_5779_ext),
                                                                                                                             growth_pred_tan_5779_ext))
    try:
      decline_pred_tan_5780 = np.float32(ct.c_float(decline_pred_tan_5780_ext))
    except (TypeError, AssertionError) as e:
      raise TypeError("Argument #13 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("f32",
                                                                                                                             type(decline_pred_tan_5780_ext),
                                                                                                                             decline_pred_tan_5780_ext))
    time_start = time.time()
    with np.errstate(divide="ignore", over="ignore", under="ignore",
                     invalid="ignore"):
      out_mem_6141 = self.futhark_runge_kutta_fwd(step_sizze_5767,
                                                  num_steps_5768,
                                                  init_prey_5769,
                                                  init_pred_5770,
                                                  growth_prey_5771,
                                                  predation_5772,
                                                  growth_pred_5773,
                                                  decline_pred_5774,
                                                  init_prey_tan_5775,
                                                  init_pred_tan_5776,
                                                  growth_prey_tan_5777,
                                                  predation_tan_5778,
                                                  growth_pred_tan_5779,
                                                  decline_pred_tan_5780)
    runtime = (int((time.time() * 1000000)) - int((time_start * 1000000)))
    sync(self)
    return cl.array.Array(self.queue, (num_steps_5768, np.int64(2)), ct.c_float,
                          data=out_mem_6141)