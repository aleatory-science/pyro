import sys
import numpy as np
import ctypes as ct
import time
import argparse
sizes = {}
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
  def __init__(self):
    pass
    self.constants = {}
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
    bytes_6024 = (np.int64(4) * num_steps_5691)
    mem_6025 = allocateMem(bytes_6024)
    i_6083 = np.int64(0)
    one_6093 = np.int64(1)
    for counter_6092 in range(num_steps_5691):
      writeScalarArray(mem_6025, i_6083, ct.c_float(init_prey_5692))
      i_6083 += one_6093
    mem_6027 = allocateMem(bytes_6024)
    i_6084 = np.int64(0)
    one_6095 = np.int64(1)
    for counter_6094 in range(num_steps_5691):
      writeScalarArray(mem_6027, i_6084, ct.c_float(init_pred_5693))
      i_6084 += one_6095
    x_5703 = (step_sizze_5690 / np.float32(2.0))
    x_5704 = (step_sizze_5690 / np.float32(6.0))
    curr_state_5710 = init_prey_5692
    curr_state_5711 = init_pred_5693
    i_5709 = np.int64(0)
    one_6097 = np.int64(1)
    for counter_6096 in range(num_steps_5691):
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
      writeScalarArray(mem_6025, i_5709, ct.c_float(loopres_5753))
      writeScalarArray(mem_6027, i_5709, ct.c_float(loopres_5760))
      curr_state_tmp_6085 = loopres_5753
      curr_state_tmp_6086 = loopres_5760
      curr_state_5710 = curr_state_tmp_6085
      curr_state_5711 = curr_state_tmp_6086
      i_5709 += one_6097
    states_5705 = curr_state_5710
    states_5706 = curr_state_5711
    binop_x_6053 = (np.int64(2) * num_steps_5691)
    bytes_6052 = (np.int64(4) * binop_x_6053)
    mem_6054 = allocateMem(bytes_6052)
    mem_6069 = allocateMem(np.int64(8))
    i_6022 = np.int64(0)
    one_6099 = np.int64(1)
    for counter_6098 in range(num_steps_5691):
      x_5764 = indexArray(mem_6025, i_6022, ct.c_float, np.float32)
      x_5765 = indexArray(mem_6027, i_6022, ct.c_float, np.float32)
      writeScalarArray(mem_6069, np.int64(0), ct.c_float(x_5764))
      writeScalarArray(mem_6069, np.int64(1), ct.c_float(x_5765))
      ct.memmove(addressOffset(mem_6054, ((i_6022 * np.int64(2)) * np.int64(4)),
                               ct.c_byte), addressOffset(mem_6069, np.int64(0),
                                                         ct.c_byte),
                 (np.int64(2) * np.int32(4)))
      i_6022 += one_6099
    mem_6025 = None
    mem_6027 = None
    mem_6069 = None
    mem_6080 = allocateMem(bytes_6052)
    ct.memmove(addressOffset(mem_6080, np.int64(0), ct.c_byte),
               addressOffset(mem_6054, np.int64(0), ct.c_byte),
               ((num_steps_5691 * np.int64(2)) * np.int32(4)))
    mem_6054 = None
    out_mem_6082 = mem_6080
    return out_mem_6082
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
    bytes_6024 = (np.int64(4) * num_steps_5768)
    mem_6025 = allocateMem(bytes_6024)
    i_6083 = np.int64(0)
    one_6101 = np.int64(1)
    for counter_6100 in range(num_steps_5768):
      writeScalarArray(mem_6025, i_6083, ct.c_float(init_prey_tan_5775))
      i_6083 += one_6101
    mem_6027 = allocateMem(bytes_6024)
    i_6084 = np.int64(0)
    one_6103 = np.int64(1)
    for counter_6102 in range(num_steps_5768):
      writeScalarArray(mem_6027, i_6084, ct.c_float(init_pred_tan_5776))
      i_6084 += one_6103
    binop_y_5888 = (np.float32(-1.0) * decline_pred_tan_5780)
    curr_state_5801 = init_prey_5769
    curr_state_tan_5870 = init_prey_tan_5775
    curr_state_5802 = init_pred_5770
    curr_state_tan_5871 = init_pred_tan_5776
    i_5800 = np.int64(0)
    one_6105 = np.int64(1)
    for counter_6104 in range(num_steps_5768):
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
      writeScalarArray(mem_6025, i_5800, ct.c_float(loopres_tan_5994))
      writeScalarArray(mem_6027, i_5800, ct.c_float(loopres_tan_6015))
      curr_state_tmp_6085 = loopres_5844
      curr_state_tan_tmp_6086 = loopres_tan_5994
      curr_state_tmp_6087 = loopres_5851
      curr_state_tan_tmp_6088 = loopres_tan_6015
      curr_state_5801 = curr_state_tmp_6085
      curr_state_tan_5870 = curr_state_tan_tmp_6086
      curr_state_5802 = curr_state_tmp_6087
      curr_state_tan_5871 = curr_state_tan_tmp_6088
      i_5800 += one_6105
    states_5796 = curr_state_5801
    states_tan_5866 = curr_state_tan_5870
    states_5797 = curr_state_5802
    states_tan_5867 = curr_state_tan_5871
    binop_x_6053 = (np.int64(2) * num_steps_5768)
    bytes_6052 = (np.int64(4) * binop_x_6053)
    mem_6054 = allocateMem(bytes_6052)
    mem_6069 = allocateMem(np.int64(8))
    i_6022 = np.int64(0)
    one_6107 = np.int64(1)
    for counter_6106 in range(num_steps_5768):
      x_5855 = indexArray(mem_6025, i_6022, ct.c_float, np.float32)
      x_5856 = indexArray(mem_6027, i_6022, ct.c_float, np.float32)
      writeScalarArray(mem_6069, np.int64(0), ct.c_float(x_5855))
      writeScalarArray(mem_6069, np.int64(1), ct.c_float(x_5856))
      ct.memmove(addressOffset(mem_6054, ((i_6022 * np.int64(2)) * np.int64(4)),
                               ct.c_byte), addressOffset(mem_6069, np.int64(0),
                                                         ct.c_byte),
                 (np.int64(2) * np.int32(4)))
      i_6022 += one_6107
    mem_6025 = None
    mem_6027 = None
    mem_6069 = None
    mem_6080 = allocateMem(bytes_6052)
    ct.memmove(addressOffset(mem_6080, np.int64(0), ct.c_byte),
               addressOffset(mem_6054, np.int64(0), ct.c_byte),
               ((num_steps_5768 * np.int64(2)) * np.int32(4)))
    mem_6054 = None
    out_mem_6082 = mem_6080
    return out_mem_6082
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
      out_mem_6082 = self.futhark_main(step_sizze_5690, num_steps_5691,
                                       init_prey_5692, init_pred_5693,
                                       growth_prey_5694, predation_5695,
                                       growth_pred_5696, decline_pred_5697)
    runtime = (int((time.time() * 1000000)) - int((time_start * 1000000)))
    return createArray(ct.cast(out_mem_6082, ct.POINTER(ct.c_float)),
                       (num_steps_5691, np.int64(2)))
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
      out_mem_6082 = self.futhark_runge_kutta_fwd(step_sizze_5767,
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
    return createArray(ct.cast(out_mem_6082, ct.POINTER(ct.c_float)),
                       (num_steps_5768, np.int64(2)))