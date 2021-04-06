from pathlib import Path

from cffi import FFI

ffibuilder = FFI()

curr_dir = Path(__file__).parent

ffibuilder.set_source('_lv_cffi', (curr_dir / 'lotka_volterra.c').open().read(), libraries=[])

ffibuilder.cdef("""
int futhark_entry_main(struct futhark_context *ctx,
                       struct futhark_f32_2d **out0, const float in0, const
                       int64_t in1, const float in2, const float in3, const
                       float in4, const float in5, const float in6, const
                       float in7); 
""")

if __name__ == '__main__':
    ffibuilder.compile(verbose=True)
