#define _GNU_SOURCE
#ifdef __GNUC__
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wunused-variable"
#pragma GCC diagnostic ignored "-Wparentheses"
#pragma GCC diagnostic ignored "-Wunused-label"
#endif
#ifdef __clang__
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wparentheses"
#pragma clang diagnostic ignored "-Wunused-label"
#endif
// Headers

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <stdio.h>
#include <float.h>

#ifdef __cplusplus
extern "C" {
#endif

// Initialisation

struct futhark_context_config ;
struct futhark_context_config *futhark_context_config_new(void);
void futhark_context_config_free(struct futhark_context_config *cfg);
void futhark_context_config_set_debugging(struct futhark_context_config *cfg,
                                          int flag);
void futhark_context_config_set_logging(struct futhark_context_config *cfg,
                                        int flag);
struct futhark_context ;
struct futhark_context *futhark_context_new(struct futhark_context_config *cfg);
void futhark_context_free(struct futhark_context *ctx);
int futhark_context_config_set_size(struct futhark_context_config *cfg, const
                                    char *size_name, size_t size_value);
int futhark_get_num_sizes(void);
const char *futhark_get_size_name(int);
const char *futhark_get_size_class(int);

// Arrays

struct futhark_f32_2d ;
struct futhark_f32_2d *futhark_new_f32_2d(struct futhark_context *ctx, const
                                          float *data, int64_t dim0,
                                          int64_t dim1);
struct futhark_f32_2d *futhark_new_raw_f32_2d(struct futhark_context *ctx, const
                                              char *data, int offset,
                                              int64_t dim0, int64_t dim1);
int futhark_free_f32_2d(struct futhark_context *ctx,
                        struct futhark_f32_2d *arr);
int futhark_values_f32_2d(struct futhark_context *ctx,
                          struct futhark_f32_2d *arr, float *data);
char *futhark_values_raw_f32_2d(struct futhark_context *ctx,
                                struct futhark_f32_2d *arr);
const int64_t *futhark_shape_f32_2d(struct futhark_context *ctx,
                                    struct futhark_f32_2d *arr);

// Opaque values


// Entry points

int futhark_entry_main(struct futhark_context *ctx,
                       struct futhark_f32_2d **out0, const float in0, const
                       int64_t in1, const float in2, const float in3, const
                       float in4, const float in5, const float in6, const
                       float in7);
int futhark_entry_runge_kutta_fwd(struct futhark_context *ctx,
                                  struct futhark_f32_2d **out0, const float in0,
                                  const int64_t in1, const float in2, const
                                  float in3, const float in4, const float in5,
                                  const float in6, const float in7, const
                                  float in8, const float in9, const float in10,
                                  const float in11, const float in12, const
                                  float in13);

// Miscellaneous

int futhark_context_sync(struct futhark_context *ctx);
char *futhark_context_report(struct futhark_context *ctx);
char *futhark_context_get_error(struct futhark_context *ctx);
void futhark_context_set_logging_file(struct futhark_context *ctx, FILE *f);
void futhark_context_pause_profiling(struct futhark_context *ctx);
void futhark_context_unpause_profiling(struct futhark_context *ctx);
int futhark_context_clear_caches(struct futhark_context *ctx);
#define FUTHARK_BACKEND_c
#ifdef __cplusplus
}
#endif
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <stdint.h>
#undef NDEBUG
#include <assert.h>
#include <stdarg.h>
// Start of util.h.
//
// Various helper functions that are useful in all generated C code.

#include <errno.h>
#include <string.h>

static const char *fut_progname = "(embedded Futhark)";

static void futhark_panic(int eval, const char *fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  fprintf(stderr, "%s: ", fut_progname);
  vfprintf(stderr, fmt, ap);
  va_end(ap);
  exit(eval);
}

// For generating arbitrary-sized error messages.  It is the callers
// responsibility to free the buffer at some point.
static char* msgprintf(const char *s, ...) {
  va_list vl;
  va_start(vl, s);
  size_t needed = 1 + (size_t)vsnprintf(NULL, 0, s, vl);
  char *buffer = (char*) malloc(needed);
  va_start(vl, s); // Must re-init.
  vsnprintf(buffer, needed, s, vl);
  return buffer;
}


static inline void check_err(int errval, int sets_errno, const char *fun, int line,
                            const char *msg, ...) {
  if (errval) {
    char errnum[10];

    va_list vl;
    va_start(vl, msg);

    fprintf(stderr, "ERROR: ");
    vfprintf(stderr, msg, vl);
    fprintf(stderr, " in %s() at line %d with error code %s\n",
            fun, line,
            sets_errno ? strerror(errno) : errnum);
    exit(errval);
  }
}

#define CHECK_ERR(err, msg...) check_err(err, 0, __func__, __LINE__, msg)
#define CHECK_ERRNO(err, msg...) check_err(err, 1, __func__, __LINE__, msg)

// Read a file into a NUL-terminated string; returns NULL on error.
static void* slurp_file(const char *filename, size_t *size) {
  unsigned char *s;
  FILE *f = fopen(filename, "rb"); // To avoid Windows messing with linebreaks.
  if (f == NULL) return NULL;
  fseek(f, 0, SEEK_END);
  size_t src_size = ftell(f);
  fseek(f, 0, SEEK_SET);
  s = (unsigned char*) malloc(src_size + 1);
  if (fread(s, 1, src_size, f) != src_size) {
    free(s);
    s = NULL;
  } else {
    s[src_size] = '\0';
  }
  fclose(f);

  if (size) {
    *size = src_size;
  }

  return s;
}

// Dump 'n' bytes from 'buf' into the file at the designated location.
// Returns 0 on success.
static int dump_file(const char *file, const void *buf, size_t n) {
  FILE *f = fopen(file, "w");

  if (f == NULL) {
    return 1;
  }

  if (fwrite(buf, sizeof(char), n, f) != n) {
    return 1;
  }

  if (fclose(f) != 0) {
    return 1;
  }

  return 0;
}

struct str_builder {
  char *str;
  size_t capacity; // Size of buffer.
  size_t used; // Bytes used, *not* including final zero.
};

static void str_builder_init(struct str_builder *b) {
  b->capacity = 10;
  b->used = 0;
  b->str = malloc(b->capacity);
  b->str[0] = 0;
}

static void str_builder(struct str_builder *b, const char *s, ...) {
  va_list vl;
  va_start(vl, s);
  size_t needed = (size_t)vsnprintf(NULL, 0, s, vl);

  while (b->capacity < b->used + needed + 1) {
    b->capacity *= 2;
    b->str = realloc(b->str, b->capacity);
  }

  va_start(vl, s); // Must re-init.
  vsnprintf(b->str+b->used, b->capacity-b->used, s, vl);
  b->used += needed;
}

// End of util.h.

// Start of timing.h.

// The function get_wall_time() returns the wall time in microseconds
// (with an unspecified offset).

#ifdef _WIN32

#include <windows.h>

static int64_t get_wall_time(void) {
  LARGE_INTEGER time,freq;
  assert(QueryPerformanceFrequency(&freq));
  assert(QueryPerformanceCounter(&time));
  return ((double)time.QuadPart / freq.QuadPart) * 1000000;
}

#else
// Assuming POSIX

#include <time.h>
#include <sys/time.h>

static int64_t get_wall_time(void) {
  struct timeval time;
  assert(gettimeofday(&time,NULL) == 0);
  return time.tv_sec * 1000000 + time.tv_usec;
}

static int64_t get_wall_time_ns(void) {
  struct timespec time;
  assert(clock_gettime(CLOCK_REALTIME, &time) == 0);
  return time.tv_sec * 1000000000 + time.tv_nsec;
}

#endif

// End of timing.h.

#ifdef _MSC_VER
#define inline __inline
#endif
#include <string.h>
#include <string.h>
#include <errno.h>
#include <assert.h>
#include <ctype.h>

// Start of lock.h.

// A very simple cross-platform implementation of locks.  Uses
// pthreads on Unix and some Windows thing there.  Futhark's
// host-level code is not multithreaded, but user code may be, so we
// need some mechanism for ensuring atomic access to API functions.
// This is that mechanism.  It is not exposed to user code at all, so
// we do not have to worry about name collisions.

#ifdef _WIN32

typedef HANDLE lock_t;

static void create_lock(lock_t *lock) {
  *lock = CreateMutex(NULL,  // Default security attributes.
                      FALSE, // Initially unlocked.
                      NULL); // Unnamed.
}

static void lock_lock(lock_t *lock) {
  assert(WaitForSingleObject(*lock, INFINITE) == WAIT_OBJECT_0);
}

static void lock_unlock(lock_t *lock) {
  assert(ReleaseMutex(*lock));
}

static void free_lock(lock_t *lock) {
  CloseHandle(*lock);
}

#else
// Assuming POSIX

#include <pthread.h>

typedef pthread_mutex_t lock_t;

static void create_lock(lock_t *lock) {
  int r = pthread_mutex_init(lock, NULL);
  assert(r == 0);
}

static void lock_lock(lock_t *lock) {
  int r = pthread_mutex_lock(lock);
  assert(r == 0);
}

static void lock_unlock(lock_t *lock) {
  int r = pthread_mutex_unlock(lock);
  assert(r == 0);
}

static void free_lock(lock_t *lock) {
  // Nothing to do for pthreads.
  (void)lock;
}

#endif

// End of lock.h.

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
static inline double fdiv64(double x, double y)
{
    return x / y;
}
static inline double fadd64(double x, double y)
{
    return x + y;
}
static inline double fsub64(double x, double y)
{
    return x - y;
}
static inline double fmul64(double x, double y)
{
    return x * y;
}
static inline double fmin64(double x, double y)
{
    return fmin(x, y);
}
static inline double fmax64(double x, double y)
{
    return fmax(x, y);
}
static inline double fpow64(double x, double y)
{
    return pow(x, y);
}
static inline bool cmplt64(double x, double y)
{
    return x < y;
}
static inline bool cmple64(double x, double y)
{
    return x <= y;
}
static inline double sitofp_i8_f64(int8_t x)
{
    return (double) x;
}
static inline double sitofp_i16_f64(int16_t x)
{
    return (double) x;
}
static inline double sitofp_i32_f64(int32_t x)
{
    return (double) x;
}
static inline double sitofp_i64_f64(int64_t x)
{
    return (double) x;
}
static inline double uitofp_i8_f64(uint8_t x)
{
    return (double) x;
}
static inline double uitofp_i16_f64(uint16_t x)
{
    return (double) x;
}
static inline double uitofp_i32_f64(uint32_t x)
{
    return (double) x;
}
static inline double uitofp_i64_f64(uint64_t x)
{
    return (double) x;
}
static inline int8_t fptosi_f64_i8(double x)
{
    return (int8_t) x;
}
static inline int16_t fptosi_f64_i16(double x)
{
    return (int16_t) x;
}
static inline int32_t fptosi_f64_i32(double x)
{
    return (int32_t) x;
}
static inline int64_t fptosi_f64_i64(double x)
{
    return (int64_t) x;
}
static inline uint8_t fptoui_f64_i8(double x)
{
    return (uint8_t) x;
}
static inline uint16_t fptoui_f64_i16(double x)
{
    return (uint16_t) x;
}
static inline uint32_t fptoui_f64_i32(double x)
{
    return (uint32_t) x;
}
static inline uint64_t fptoui_f64_i64(double x)
{
    return (uint64_t) x;
}
static inline float fpconv_f32_f32(float x)
{
    return (float) x;
}
static inline double fpconv_f32_f64(float x)
{
    return (double) x;
}
static inline float fpconv_f64_f32(double x)
{
    return (float) x;
}
static inline double fpconv_f64_f64(double x)
{
    return (double) x;
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
static inline double futrts_log64(double x)
{
    return log(x);
}
static inline double futrts_log2_64(double x)
{
    return log2(x);
}
static inline double futrts_log10_64(double x)
{
    return log10(x);
}
static inline double futrts_sqrt64(double x)
{
    return sqrt(x);
}
static inline double futrts_exp64(double x)
{
    return exp(x);
}
static inline double futrts_cos64(double x)
{
    return cos(x);
}
static inline double futrts_sin64(double x)
{
    return sin(x);
}
static inline double futrts_tan64(double x)
{
    return tan(x);
}
static inline double futrts_acos64(double x)
{
    return acos(x);
}
static inline double futrts_asin64(double x)
{
    return asin(x);
}
static inline double futrts_atan64(double x)
{
    return atan(x);
}
static inline double futrts_cosh64(double x)
{
    return cosh(x);
}
static inline double futrts_sinh64(double x)
{
    return sinh(x);
}
static inline double futrts_tanh64(double x)
{
    return tanh(x);
}
static inline double futrts_acosh64(double x)
{
    return acosh(x);
}
static inline double futrts_asinh64(double x)
{
    return asinh(x);
}
static inline double futrts_atanh64(double x)
{
    return atanh(x);
}
static inline double futrts_atan2_64(double x, double y)
{
    return atan2(x, y);
}
static inline double futrts_gamma64(double x)
{
    return tgamma(x);
}
static inline double futrts_lgamma64(double x)
{
    return lgamma(x);
}
static inline double futrts_fma64(double a, double b, double c)
{
    return fma(a, b, c);
}
static inline double futrts_round64(double x)
{
    return rint(x);
}
static inline double futrts_ceil64(double x)
{
    return ceil(x);
}
static inline double futrts_floor64(double x)
{
    return floor(x);
}
static inline bool futrts_isnan64(double x)
{
    return isnan(x);
}
static inline bool futrts_isinf64(double x)
{
    return isinf(x);
}
static inline int64_t futrts_to_bits64(double x)
{
    union {
        double f;
        int64_t t;
    } p;
    
    p.f = x;
    return p.t;
}
static inline double futrts_from_bits64(int64_t x)
{
    union {
        int64_t f;
        double t;
    } p;
    
    p.f = x;
    return p.t;
}
static inline double fmod64(double x, double y)
{
    return fmod(x, y);
}
static inline double fsignum64(double x)
{
    return futrts_isnan64(x) ? x : (x > 0) - (x < 0);
}
#ifdef __OPENCL_VERSION__
static inline double futrts_lerp64(double v0, double v1, double t)
{
    return mix(v0, v1, t);
}
static inline double futrts_mad64(double a, double b, double c)
{
    return mad(a, b, c);
}
#else
static inline double futrts_lerp64(double v0, double v1, double t)
{
    return v0 + (v1 - v0) * t;
}
static inline double futrts_mad64(double a, double b, double c)
{
    return a * b + c;
}
#endif
static int init_constants(struct futhark_context *);
static int free_constants(struct futhark_context *);
struct memblock {
    int *references;
    char *mem;
    int64_t size;
    const char *desc;
} ;
struct futhark_context_config {
    int debugging;
} ;
struct futhark_context_config *futhark_context_config_new(void)
{
    struct futhark_context_config *cfg =
                                  (struct futhark_context_config *) malloc(sizeof(struct futhark_context_config));
    
    if (cfg == NULL)
        return NULL;
    cfg->debugging = 0;
    return cfg;
}
void futhark_context_config_free(struct futhark_context_config *cfg)
{
    free(cfg);
}
void futhark_context_config_set_debugging(struct futhark_context_config *cfg,
                                          int detail)
{
    cfg->debugging = detail;
}
void futhark_context_config_set_logging(struct futhark_context_config *cfg,
                                        int detail)
{
    /* Does nothing for this backend. */
    (void) cfg;
    (void) detail;
}
struct futhark_context {
    int detail_memory;
    int debugging;
    int profiling;
    int logging;
    lock_t lock;
    char *error;
    FILE *log;
    int profiling_paused;
    int64_t peak_mem_usage_default;
    int64_t cur_mem_usage_default;
    struct {
        int dummy;
    } constants;
} ;
struct futhark_context *futhark_context_new(struct futhark_context_config *cfg)
{
    struct futhark_context *ctx =
                           (struct futhark_context *) malloc(sizeof(struct futhark_context));
    
    if (ctx == NULL)
        return NULL;
    ctx->detail_memory = cfg->debugging;
    ctx->debugging = cfg->debugging;
    ctx->profiling = cfg->debugging;
    ctx->logging = cfg->debugging;
    ctx->error = NULL;
    ctx->log = stderr;
    create_lock(&ctx->lock);
    ctx->peak_mem_usage_default = 0;
    ctx->cur_mem_usage_default = 0;
    init_constants(ctx);
    return ctx;
}
void futhark_context_free(struct futhark_context *ctx)
{
    free_constants(ctx);
    free_lock(&ctx->lock);
    free(ctx);
}
int futhark_context_sync(struct futhark_context *ctx)
{
    (void) ctx;
    return 0;
}
static const char *size_names[0];
static const char *size_vars[0];
static const char *size_classes[0];
int futhark_context_config_set_size(struct futhark_context_config *cfg, const
                                    char *size_name, size_t size_value)
{
    (void) cfg;
    (void) size_name;
    (void) size_value;
    return 1;
}
static int memblock_unref(struct futhark_context *ctx, struct memblock *block,
                          const char *desc)
{
    if (block->references != NULL) {
        *block->references -= 1;
        if (ctx->detail_memory)
            fprintf(ctx->log,
                    "Unreferencing block %s (allocated as %s) in %s: %d references remaining.\n",
                    desc, block->desc, "default space", *block->references);
        if (*block->references == 0) {
            ctx->cur_mem_usage_default -= block->size;
            free(block->mem);
            free(block->references);
            if (ctx->detail_memory)
                fprintf(ctx->log,
                        "%lld bytes freed (now allocated: %lld bytes)\n",
                        (long long) block->size,
                        (long long) ctx->cur_mem_usage_default);
        }
        block->references = NULL;
    }
    return 0;
}
static int memblock_alloc(struct futhark_context *ctx, struct memblock *block,
                          int64_t size, const char *desc)
{
    if (size < 0)
        futhark_panic(1,
                      "Negative allocation of %lld bytes attempted for %s in %s.\n",
                      (long long) size, desc, "default space",
                      ctx->cur_mem_usage_default);
    
    int ret = memblock_unref(ctx, block, desc);
    
    ctx->cur_mem_usage_default += size;
    if (ctx->detail_memory)
        fprintf(ctx->log,
                "Allocating %lld bytes for %s in %s (then allocated: %lld bytes)",
                (long long) size, desc, "default space",
                (long long) ctx->cur_mem_usage_default);
    if (ctx->cur_mem_usage_default > ctx->peak_mem_usage_default) {
        ctx->peak_mem_usage_default = ctx->cur_mem_usage_default;
        if (ctx->detail_memory)
            fprintf(ctx->log, " (new peak).\n");
    } else if (ctx->detail_memory)
        fprintf(ctx->log, ".\n");
    block->mem = (char *) malloc(size);
    block->references = (int *) malloc(sizeof(int));
    *block->references = 1;
    block->size = size;
    block->desc = desc;
    return ret;
}
static int memblock_set(struct futhark_context *ctx, struct memblock *lhs,
                        struct memblock *rhs, const char *lhs_desc)
{
    int ret = memblock_unref(ctx, lhs, lhs_desc);
    
    if (rhs->references != NULL)
        (*rhs->references)++;
    *lhs = *rhs;
    return ret;
}
int futhark_get_num_sizes(void)
{
    return sizeof(size_names) / sizeof(size_names[0]);
}
const char *futhark_get_size_name(int i)
{
    return size_names[i];
}
const char *futhark_get_size_class(int i)
{
    return size_classes[i];
}
char *futhark_context_report(struct futhark_context *ctx)
{
    struct str_builder builder;
    
    str_builder_init(&builder);
    if (ctx->detail_memory || ctx->profiling || ctx->logging) {
        { }
    }
    if (ctx->profiling) { }
    return builder.str;
}
char *futhark_context_get_error(struct futhark_context *ctx)
{
    char *error = ctx->error;
    
    ctx->error = NULL;
    return error;
}
void futhark_context_set_logging_file(struct futhark_context *ctx, FILE *f)
{
    ctx->log = f;
}
void futhark_context_pause_profiling(struct futhark_context *ctx)
{
    ctx->profiling_paused = 1;
}
void futhark_context_unpause_profiling(struct futhark_context *ctx)
{
    ctx->profiling_paused = 0;
}
int futhark_context_clear_caches(struct futhark_context *ctx)
{
    lock_lock(&ctx->lock);
    ctx->peak_mem_usage_default = 0;
    lock_unlock(&ctx->lock);
    return ctx->error != NULL;
}
static int futrts_main(struct futhark_context *ctx,
                       struct memblock *out_mem_p_6092, float step_sizze_5690,
                       int64_t num_steps_5691, float init_prey_5692,
                       float init_pred_5693, float growth_prey_5694,
                       float predation_5695, float growth_pred_5696,
                       float decline_pred_5697);
static int futrts_runge_kutta_fwd(struct futhark_context *ctx,
                                  struct memblock *out_mem_p_6097,
                                  float step_sizze_5767, int64_t num_steps_5768,
                                  float init_prey_5769, float init_pred_5770,
                                  float growth_prey_5771, float predation_5772,
                                  float growth_pred_5773,
                                  float decline_pred_5774,
                                  float init_prey_tan_5775,
                                  float init_pred_tan_5776,
                                  float growth_prey_tan_5777,
                                  float predation_tan_5778,
                                  float growth_pred_tan_5779,
                                  float decline_pred_tan_5780);
static int init_constants(struct futhark_context *ctx)
{
    (void) ctx;
    
    int err = 0;
    
    
  cleanup:
    return err;
}
static int free_constants(struct futhark_context *ctx)
{
    (void) ctx;
    return 0;
}
static int futrts_main(struct futhark_context *ctx,
                       struct memblock *out_mem_p_6092, float step_sizze_5690,
                       int64_t num_steps_5691, float init_prey_5692,
                       float init_pred_5693, float growth_prey_5694,
                       float predation_5695, float growth_pred_5696,
                       float decline_pred_5697)
{
    (void) ctx;
    
    int err = 0;
    size_t mem_6025_cached_sizze_6093 = 0;
    char *mem_6025 = NULL;
    size_t mem_6027_cached_sizze_6094 = 0;
    char *mem_6027 = NULL;
    size_t mem_6054_cached_sizze_6095 = 0;
    char *mem_6054 = NULL;
    size_t mem_6069_cached_sizze_6096 = 0;
    char *mem_6069 = NULL;
    struct memblock out_mem_6082;
    
    out_mem_6082.references = NULL;
    
    bool bounds_invalid_upwards_5698 = slt64(num_steps_5691, (int64_t) 0);
    bool valid_5699 = !bounds_invalid_upwards_5698;
    bool range_valid_c_5700;
    
    if (!valid_5699) {
        ctx->error = msgprintf("Error: %s%lld%s%lld%s%lld%s\n\nBacktrace:\n%s",
                               "Range ", (int64_t) 0, "..", (int64_t) 1, "..<",
                               num_steps_5691, " is invalid.",
                               "-> #0  /prelude/array.fut:90:3-10\n   #1  /prelude/array.fut:108:18-23\n   #2  lotka_volterra.fut:54:1-62:122\n");
        if (memblock_unref(ctx, &out_mem_6082, "out_mem_6082") != 0)
            return 1;
        return 1;
    }
    
    int64_t bytes_6024 = (int64_t) 4 * num_steps_5691;
    
    if (mem_6025_cached_sizze_6093 < (size_t) bytes_6024) {
        mem_6025 = realloc(mem_6025, bytes_6024);
        mem_6025_cached_sizze_6093 = bytes_6024;
    }
    for (int64_t i_6083 = 0; i_6083 < num_steps_5691; i_6083++) {
        ((float *) mem_6025)[i_6083] = init_prey_5692;
    }
    if (mem_6027_cached_sizze_6094 < (size_t) bytes_6024) {
        mem_6027 = realloc(mem_6027, bytes_6024);
        mem_6027_cached_sizze_6094 = bytes_6024;
    }
    for (int64_t i_6084 = 0; i_6084 < num_steps_5691; i_6084++) {
        ((float *) mem_6027)[i_6084] = init_pred_5693;
    }
    
    float x_5703 = step_sizze_5690 / 2.0F;
    float x_5704 = step_sizze_5690 / 6.0F;
    float states_5705;
    float states_5706;
    float curr_state_5710;
    float curr_state_5711;
    
    curr_state_5710 = init_prey_5692;
    curr_state_5711 = init_pred_5693;
    for (int64_t i_5709 = 0; i_5709 < num_steps_5691; i_5709++) {
        float y_5714 = predation_5695 * curr_state_5711;
        float x_5715 = growth_prey_5694 - y_5714;
        float dprey_5716 = curr_state_5710 * x_5715;
        float x_5717 = growth_pred_5696 * curr_state_5710;
        float x_5718 = x_5717 - decline_pred_5697;
        float dpred_5719 = curr_state_5711 * x_5718;
        float y_5720 = x_5703 * dprey_5716;
        float defunc_1_fn_arg_5721 = curr_state_5711 + y_5720;
        float defunc_0_fn_arg_5722 = curr_state_5710 + y_5720;
        float y_5723 = predation_5695 * defunc_1_fn_arg_5721;
        float x_5724 = growth_prey_5694 - y_5723;
        float dprey_5725 = defunc_0_fn_arg_5722 * x_5724;
        float x_5726 = growth_pred_5696 * defunc_0_fn_arg_5722;
        float x_5727 = x_5726 - decline_pred_5697;
        float dpred_5728 = defunc_1_fn_arg_5721 * x_5727;
        float y_5729 = x_5703 * dprey_5725;
        float defunc_1_fn_arg_5730 = curr_state_5711 + y_5729;
        float defunc_0_fn_arg_5731 = curr_state_5710 + y_5729;
        float y_5732 = predation_5695 * defunc_1_fn_arg_5730;
        float x_5733 = growth_prey_5694 - y_5732;
        float dprey_5734 = defunc_0_fn_arg_5731 * x_5733;
        float x_5735 = growth_pred_5696 * defunc_0_fn_arg_5731;
        float x_5736 = x_5735 - decline_pred_5697;
        float dpred_5737 = defunc_1_fn_arg_5730 * x_5736;
        float y_5738 = step_sizze_5690 * dprey_5734;
        float defunc_1_fn_arg_5739 = curr_state_5711 + y_5738;
        float defunc_0_fn_arg_5740 = curr_state_5710 + y_5738;
        float y_5741 = predation_5695 * defunc_1_fn_arg_5739;
        float x_5742 = growth_prey_5694 - y_5741;
        float dprey_5743 = defunc_0_fn_arg_5740 * x_5742;
        float x_5744 = growth_pred_5696 * defunc_0_fn_arg_5740;
        float x_5745 = x_5744 - decline_pred_5697;
        float dpred_5746 = defunc_1_fn_arg_5739 * x_5745;
        float y_5747 = 2.0F * dprey_5725;
        float x_5748 = dprey_5716 + y_5747;
        float y_5749 = 2.0F * dprey_5734;
        float x_5750 = x_5748 + y_5749;
        float y_5751 = dprey_5743 + x_5750;
        float y_5752 = x_5704 * y_5751;
        float loopres_5753 = curr_state_5710 + y_5752;
        float y_5754 = 2.0F * dpred_5728;
        float x_5755 = dpred_5719 + y_5754;
        float y_5756 = 2.0F * dpred_5737;
        float x_5757 = x_5755 + y_5756;
        float y_5758 = dpred_5746 + x_5757;
        float y_5759 = x_5704 * y_5758;
        float loopres_5760 = curr_state_5711 + y_5759;
        
        ((float *) mem_6025)[i_5709] = loopres_5753;
        ((float *) mem_6027)[i_5709] = loopres_5760;
        
        float curr_state_tmp_6085 = loopres_5753;
        float curr_state_tmp_6086 = loopres_5760;
        
        curr_state_5710 = curr_state_tmp_6085;
        curr_state_5711 = curr_state_tmp_6086;
    }
    states_5705 = curr_state_5710;
    states_5706 = curr_state_5711;
    
    int64_t binop_x_6053 = (int64_t) 2 * num_steps_5691;
    int64_t bytes_6052 = (int64_t) 4 * binop_x_6053;
    
    if (mem_6054_cached_sizze_6095 < (size_t) bytes_6052) {
        mem_6054 = realloc(mem_6054, bytes_6052);
        mem_6054_cached_sizze_6095 = bytes_6052;
    }
    if (mem_6069_cached_sizze_6096 < (size_t) (int64_t) 8) {
        mem_6069 = realloc(mem_6069, (int64_t) 8);
        mem_6069_cached_sizze_6096 = (int64_t) 8;
    }
    for (int64_t i_6022 = 0; i_6022 < num_steps_5691; i_6022++) {
        float x_5764 = ((float *) mem_6025)[i_6022];
        float x_5765 = ((float *) mem_6027)[i_6022];
        
        ((float *) mem_6069)[(int64_t) 0] = x_5764;
        ((float *) mem_6069)[(int64_t) 1] = x_5765;
        memmove(mem_6054 + i_6022 * (int64_t) 2 * (int64_t) 4, mem_6069 +
                (int64_t) 0, (int64_t) 2 * (int64_t) sizeof(float));
    }
    
    struct memblock mem_6080;
    
    mem_6080.references = NULL;
    if (memblock_alloc(ctx, &mem_6080, bytes_6052, "mem_6080")) {
        err = 1;
        goto cleanup;
    }
    memmove(mem_6080.mem + (int64_t) 0, mem_6054 + (int64_t) 0, num_steps_5691 *
            (int64_t) 2 * (int64_t) sizeof(float));
    if (memblock_set(ctx, &out_mem_6082, &mem_6080, "mem_6080") != 0)
        return 1;
    (*out_mem_p_6092).references = NULL;
    if (memblock_set(ctx, &*out_mem_p_6092, &out_mem_6082, "out_mem_6082") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_6080, "mem_6080") != 0)
        return 1;
    if (memblock_unref(ctx, &out_mem_6082, "out_mem_6082") != 0)
        return 1;
    
  cleanup:
    { }
    free(mem_6025);
    free(mem_6027);
    free(mem_6054);
    free(mem_6069);
    return err;
}
static int futrts_runge_kutta_fwd(struct futhark_context *ctx,
                                  struct memblock *out_mem_p_6097,
                                  float step_sizze_5767, int64_t num_steps_5768,
                                  float init_prey_5769, float init_pred_5770,
                                  float growth_prey_5771, float predation_5772,
                                  float growth_pred_5773,
                                  float decline_pred_5774,
                                  float init_prey_tan_5775,
                                  float init_pred_tan_5776,
                                  float growth_prey_tan_5777,
                                  float predation_tan_5778,
                                  float growth_pred_tan_5779,
                                  float decline_pred_tan_5780)
{
    (void) ctx;
    
    int err = 0;
    size_t mem_6025_cached_sizze_6098 = 0;
    char *mem_6025 = NULL;
    size_t mem_6027_cached_sizze_6099 = 0;
    char *mem_6027 = NULL;
    size_t mem_6054_cached_sizze_6100 = 0;
    char *mem_6054 = NULL;
    size_t mem_6069_cached_sizze_6101 = 0;
    char *mem_6069 = NULL;
    struct memblock out_mem_6082;
    
    out_mem_6082.references = NULL;
    
    bool bounds_invalid_upwards_5781 = slt64(num_steps_5768, (int64_t) 0);
    bool valid_5782 = !bounds_invalid_upwards_5781;
    bool range_valid_c_5783;
    
    if (!valid_5782) {
        ctx->error = msgprintf("Error: %s%lld%s%lld%s%lld%s\n\nBacktrace:\n%s",
                               "Range ", (int64_t) 0, "..", (int64_t) 1, "..<",
                               num_steps_5768, " is invalid.",
                               "-> #0  /prelude/array.fut:90:3-10\n   #1  /prelude/array.fut:108:18-23\n   #2  lotka_volterra.fut:64:1-92:35\n");
        if (memblock_unref(ctx, &out_mem_6082, "out_mem_6082") != 0)
            return 1;
        return 1;
    }
    
    float x_5784 = step_sizze_5767 / 2.0F;
    float x_5785 = step_sizze_5767 / 6.0F;
    int64_t bytes_6024 = (int64_t) 4 * num_steps_5768;
    
    if (mem_6025_cached_sizze_6098 < (size_t) bytes_6024) {
        mem_6025 = realloc(mem_6025, bytes_6024);
        mem_6025_cached_sizze_6098 = bytes_6024;
    }
    for (int64_t i_6083 = 0; i_6083 < num_steps_5768; i_6083++) {
        ((float *) mem_6025)[i_6083] = init_prey_tan_5775;
    }
    if (mem_6027_cached_sizze_6099 < (size_t) bytes_6024) {
        mem_6027 = realloc(mem_6027, bytes_6024);
        mem_6027_cached_sizze_6099 = bytes_6024;
    }
    for (int64_t i_6084 = 0; i_6084 < num_steps_5768; i_6084++) {
        ((float *) mem_6027)[i_6084] = init_pred_tan_5776;
    }
    
    float binop_y_5888 = -1.0F * decline_pred_tan_5780;
    float states_5796;
    float states_tan_5866;
    float states_5797;
    float states_tan_5867;
    float curr_state_5801;
    float curr_state_5802;
    float curr_state_tan_5870;
    float curr_state_tan_5871;
    
    curr_state_5801 = init_prey_5769;
    curr_state_tan_5870 = init_prey_tan_5775;
    curr_state_5802 = init_pred_5770;
    curr_state_tan_5871 = init_pred_tan_5776;
    for (int64_t i_5800 = 0; i_5800 < num_steps_5768; i_5800++) {
        float y_5805 = predation_5772 * curr_state_5802;
        float binop_x_5875 = predation_tan_5778 * curr_state_5802;
        float binop_y_5876 = predation_5772 * curr_state_tan_5871;
        float y_tan_5874 = binop_x_5875 + binop_y_5876;
        float x_5806 = growth_prey_5771 - y_5805;
        float binop_y_5879 = -1.0F * y_tan_5874;
        float x_tan_5877 = growth_prey_tan_5777 + binop_y_5879;
        float dprey_5807 = curr_state_5801 * x_5806;
        float binop_x_5881 = x_5806 * curr_state_tan_5870;
        float binop_y_5882 = curr_state_5801 * x_tan_5877;
        float dprey_tan_5880 = binop_x_5881 + binop_y_5882;
        float x_5808 = growth_pred_5773 * curr_state_5801;
        float binop_x_5884 = growth_pred_tan_5779 * curr_state_5801;
        float binop_y_5885 = growth_pred_5773 * curr_state_tan_5870;
        float x_tan_5883 = binop_x_5884 + binop_y_5885;
        float x_5809 = x_5808 - decline_pred_5774;
        float x_tan_5886 = x_tan_5883 + binop_y_5888;
        float dpred_5810 = curr_state_5802 * x_5809;
        float binop_x_5890 = x_5809 * curr_state_tan_5871;
        float binop_y_5891 = curr_state_5802 * x_tan_5886;
        float dpred_tan_5889 = binop_x_5890 + binop_y_5891;
        float y_5811 = x_5784 * dprey_5807;
        float binop_y_5895 = x_5784 * dprey_tan_5880;
        float defunc_1_fn_arg_5812 = curr_state_5802 + y_5811;
        float defunc_1_fn_arg_tan_5896 = curr_state_tan_5871 + binop_y_5895;
        float defunc_0_fn_arg_5813 = curr_state_5801 + y_5811;
        float defunc_0_fn_arg_tan_5899 = curr_state_tan_5870 + binop_y_5895;
        float y_5814 = predation_5772 * defunc_1_fn_arg_5812;
        float binop_x_5903 = predation_tan_5778 * defunc_1_fn_arg_5812;
        float binop_y_5904 = predation_5772 * defunc_1_fn_arg_tan_5896;
        float y_tan_5902 = binop_x_5903 + binop_y_5904;
        float x_5815 = growth_prey_5771 - y_5814;
        float binop_y_5907 = -1.0F * y_tan_5902;
        float x_tan_5905 = growth_prey_tan_5777 + binop_y_5907;
        float dprey_5816 = defunc_0_fn_arg_5813 * x_5815;
        float binop_x_5909 = x_5815 * defunc_0_fn_arg_tan_5899;
        float binop_y_5910 = defunc_0_fn_arg_5813 * x_tan_5905;
        float dprey_tan_5908 = binop_x_5909 + binop_y_5910;
        float x_5817 = growth_pred_5773 * defunc_0_fn_arg_5813;
        float binop_x_5912 = growth_pred_tan_5779 * defunc_0_fn_arg_5813;
        float binop_y_5913 = growth_pred_5773 * defunc_0_fn_arg_tan_5899;
        float x_tan_5911 = binop_x_5912 + binop_y_5913;
        float x_5818 = x_5817 - decline_pred_5774;
        float x_tan_5914 = binop_y_5888 + x_tan_5911;
        float dpred_5819 = defunc_1_fn_arg_5812 * x_5818;
        float binop_x_5918 = x_5818 * defunc_1_fn_arg_tan_5896;
        float binop_y_5919 = defunc_1_fn_arg_5812 * x_tan_5914;
        float dpred_tan_5917 = binop_x_5918 + binop_y_5919;
        float y_5820 = x_5784 * dprey_5816;
        float binop_y_5922 = x_5784 * dprey_tan_5908;
        float defunc_1_fn_arg_5821 = curr_state_5802 + y_5820;
        float defunc_1_fn_arg_tan_5923 = curr_state_tan_5871 + binop_y_5922;
        float defunc_0_fn_arg_5822 = curr_state_5801 + y_5820;
        float defunc_0_fn_arg_tan_5926 = curr_state_tan_5870 + binop_y_5922;
        float y_5823 = predation_5772 * defunc_1_fn_arg_5821;
        float binop_x_5930 = predation_tan_5778 * defunc_1_fn_arg_5821;
        float binop_y_5931 = predation_5772 * defunc_1_fn_arg_tan_5923;
        float y_tan_5929 = binop_x_5930 + binop_y_5931;
        float x_5824 = growth_prey_5771 - y_5823;
        float binop_y_5934 = -1.0F * y_tan_5929;
        float x_tan_5932 = growth_prey_tan_5777 + binop_y_5934;
        float dprey_5825 = defunc_0_fn_arg_5822 * x_5824;
        float binop_x_5936 = x_5824 * defunc_0_fn_arg_tan_5926;
        float binop_y_5937 = defunc_0_fn_arg_5822 * x_tan_5932;
        float dprey_tan_5935 = binop_x_5936 + binop_y_5937;
        float x_5826 = growth_pred_5773 * defunc_0_fn_arg_5822;
        float binop_x_5939 = growth_pred_tan_5779 * defunc_0_fn_arg_5822;
        float binop_y_5940 = growth_pred_5773 * defunc_0_fn_arg_tan_5926;
        float x_tan_5938 = binop_x_5939 + binop_y_5940;
        float x_5827 = x_5826 - decline_pred_5774;
        float x_tan_5941 = binop_y_5888 + x_tan_5938;
        float dpred_5828 = defunc_1_fn_arg_5821 * x_5827;
        float binop_x_5945 = x_5827 * defunc_1_fn_arg_tan_5923;
        float binop_y_5946 = defunc_1_fn_arg_5821 * x_tan_5941;
        float dpred_tan_5944 = binop_x_5945 + binop_y_5946;
        float y_5829 = step_sizze_5767 * dprey_5825;
        float binop_y_5950 = step_sizze_5767 * dprey_tan_5935;
        float defunc_1_fn_arg_5830 = curr_state_5802 + y_5829;
        float defunc_1_fn_arg_tan_5951 = curr_state_tan_5871 + binop_y_5950;
        float defunc_0_fn_arg_5831 = curr_state_5801 + y_5829;
        float defunc_0_fn_arg_tan_5954 = curr_state_tan_5870 + binop_y_5950;
        float y_5832 = predation_5772 * defunc_1_fn_arg_5830;
        float binop_x_5958 = predation_tan_5778 * defunc_1_fn_arg_5830;
        float binop_y_5959 = predation_5772 * defunc_1_fn_arg_tan_5951;
        float y_tan_5957 = binop_x_5958 + binop_y_5959;
        float x_5833 = growth_prey_5771 - y_5832;
        float binop_y_5962 = -1.0F * y_tan_5957;
        float x_tan_5960 = growth_prey_tan_5777 + binop_y_5962;
        float dprey_5834 = defunc_0_fn_arg_5831 * x_5833;
        float binop_x_5964 = x_5833 * defunc_0_fn_arg_tan_5954;
        float binop_y_5965 = defunc_0_fn_arg_5831 * x_tan_5960;
        float dprey_tan_5963 = binop_x_5964 + binop_y_5965;
        float x_5835 = growth_pred_5773 * defunc_0_fn_arg_5831;
        float binop_x_5967 = growth_pred_tan_5779 * defunc_0_fn_arg_5831;
        float binop_y_5968 = growth_pred_5773 * defunc_0_fn_arg_tan_5954;
        float x_tan_5966 = binop_x_5967 + binop_y_5968;
        float x_5836 = x_5835 - decline_pred_5774;
        float x_tan_5969 = binop_y_5888 + x_tan_5966;
        float dpred_5837 = defunc_1_fn_arg_5830 * x_5836;
        float binop_x_5973 = x_5836 * defunc_1_fn_arg_tan_5951;
        float binop_y_5974 = defunc_1_fn_arg_5830 * x_tan_5969;
        float dpred_tan_5972 = binop_x_5973 + binop_y_5974;
        float y_5838 = 2.0F * dprey_5816;
        float binop_y_5977 = 2.0F * dprey_tan_5908;
        float x_5839 = dprey_5807 + y_5838;
        float x_tan_5978 = dprey_tan_5880 + binop_y_5977;
        float y_5840 = 2.0F * dprey_5825;
        float binop_y_5983 = 2.0F * dprey_tan_5935;
        float x_5841 = x_5839 + y_5840;
        float x_tan_5984 = x_tan_5978 + binop_y_5983;
        float y_5842 = dprey_5834 + x_5841;
        float y_tan_5987 = dprey_tan_5963 + x_tan_5984;
        float y_5843 = x_5785 * y_5842;
        float binop_y_5993 = x_5785 * y_tan_5987;
        float loopres_5844 = curr_state_5801 + y_5843;
        float loopres_tan_5994 = curr_state_tan_5870 + binop_y_5993;
        float y_5845 = 2.0F * dpred_5819;
        float binop_y_5999 = 2.0F * dpred_tan_5917;
        float x_5846 = dpred_5810 + y_5845;
        float x_tan_6000 = dpred_tan_5889 + binop_y_5999;
        float y_5847 = 2.0F * dpred_5828;
        float binop_y_6005 = 2.0F * dpred_tan_5944;
        float x_5848 = x_5846 + y_5847;
        float x_tan_6006 = x_tan_6000 + binop_y_6005;
        float y_5849 = dpred_5837 + x_5848;
        float y_tan_6009 = dpred_tan_5972 + x_tan_6006;
        float y_5850 = x_5785 * y_5849;
        float binop_y_6014 = x_5785 * y_tan_6009;
        float loopres_5851 = curr_state_5802 + y_5850;
        float loopres_tan_6015 = curr_state_tan_5871 + binop_y_6014;
        
        ((float *) mem_6025)[i_5800] = loopres_tan_5994;
        ((float *) mem_6027)[i_5800] = loopres_tan_6015;
        
        float curr_state_tmp_6085 = loopres_5844;
        float curr_state_tan_tmp_6086 = loopres_tan_5994;
        float curr_state_tmp_6087 = loopres_5851;
        float curr_state_tan_tmp_6088 = loopres_tan_6015;
        
        curr_state_5801 = curr_state_tmp_6085;
        curr_state_tan_5870 = curr_state_tan_tmp_6086;
        curr_state_5802 = curr_state_tmp_6087;
        curr_state_tan_5871 = curr_state_tan_tmp_6088;
    }
    states_5796 = curr_state_5801;
    states_tan_5866 = curr_state_tan_5870;
    states_5797 = curr_state_5802;
    states_tan_5867 = curr_state_tan_5871;
    
    int64_t binop_x_6053 = (int64_t) 2 * num_steps_5768;
    int64_t bytes_6052 = (int64_t) 4 * binop_x_6053;
    
    if (mem_6054_cached_sizze_6100 < (size_t) bytes_6052) {
        mem_6054 = realloc(mem_6054, bytes_6052);
        mem_6054_cached_sizze_6100 = bytes_6052;
    }
    if (mem_6069_cached_sizze_6101 < (size_t) (int64_t) 8) {
        mem_6069 = realloc(mem_6069, (int64_t) 8);
        mem_6069_cached_sizze_6101 = (int64_t) 8;
    }
    for (int64_t i_6022 = 0; i_6022 < num_steps_5768; i_6022++) {
        float x_5855 = ((float *) mem_6025)[i_6022];
        float x_5856 = ((float *) mem_6027)[i_6022];
        
        ((float *) mem_6069)[(int64_t) 0] = x_5855;
        ((float *) mem_6069)[(int64_t) 1] = x_5856;
        memmove(mem_6054 + i_6022 * (int64_t) 2 * (int64_t) 4, mem_6069 +
                (int64_t) 0, (int64_t) 2 * (int64_t) sizeof(float));
    }
    
    struct memblock mem_6080;
    
    mem_6080.references = NULL;
    if (memblock_alloc(ctx, &mem_6080, bytes_6052, "mem_6080")) {
        err = 1;
        goto cleanup;
    }
    memmove(mem_6080.mem + (int64_t) 0, mem_6054 + (int64_t) 0, num_steps_5768 *
            (int64_t) 2 * (int64_t) sizeof(float));
    if (memblock_set(ctx, &out_mem_6082, &mem_6080, "mem_6080") != 0)
        return 1;
    (*out_mem_p_6097).references = NULL;
    if (memblock_set(ctx, &*out_mem_p_6097, &out_mem_6082, "out_mem_6082") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_6080, "mem_6080") != 0)
        return 1;
    if (memblock_unref(ctx, &out_mem_6082, "out_mem_6082") != 0)
        return 1;
    
  cleanup:
    { }
    free(mem_6025);
    free(mem_6027);
    free(mem_6054);
    free(mem_6069);
    return err;
}
struct futhark_f32_2d {
    struct memblock mem;
    int64_t shape[2];
} ;
struct futhark_f32_2d *futhark_new_f32_2d(struct futhark_context *ctx, const
                                          float *data, int64_t dim0,
                                          int64_t dim1)
{
    struct futhark_f32_2d *bad = NULL;
    struct futhark_f32_2d *arr =
                          (struct futhark_f32_2d *) malloc(sizeof(struct futhark_f32_2d));
    
    if (arr == NULL)
        return bad;
    lock_lock(&ctx->lock);
    arr->mem.references = NULL;
    if (memblock_alloc(ctx, &arr->mem, (size_t) (dim0 * dim1) * sizeof(float),
                       "arr->mem"))
        return NULL;
    arr->shape[0] = dim0;
    arr->shape[1] = dim1;
    memmove(arr->mem.mem + 0, data + 0, (size_t) (dim0 * dim1) * sizeof(float));
    lock_unlock(&ctx->lock);
    return arr;
}
struct futhark_f32_2d *futhark_new_raw_f32_2d(struct futhark_context *ctx, const
                                              char *data, int offset,
                                              int64_t dim0, int64_t dim1)
{
    struct futhark_f32_2d *bad = NULL;
    struct futhark_f32_2d *arr =
                          (struct futhark_f32_2d *) malloc(sizeof(struct futhark_f32_2d));
    
    if (arr == NULL)
        return bad;
    lock_lock(&ctx->lock);
    arr->mem.references = NULL;
    if (memblock_alloc(ctx, &arr->mem, (size_t) (dim0 * dim1) * sizeof(float),
                       "arr->mem"))
        return NULL;
    arr->shape[0] = dim0;
    arr->shape[1] = dim1;
    memmove(arr->mem.mem + 0, data + offset, (size_t) (dim0 * dim1) *
            sizeof(float));
    lock_unlock(&ctx->lock);
    return arr;
}
int futhark_free_f32_2d(struct futhark_context *ctx, struct futhark_f32_2d *arr)
{
    lock_lock(&ctx->lock);
    if (memblock_unref(ctx, &arr->mem, "arr->mem") != 0)
        return 1;
    lock_unlock(&ctx->lock);
    free(arr);
    return 0;
}
int futhark_values_f32_2d(struct futhark_context *ctx,
                          struct futhark_f32_2d *arr, float *data)
{
    lock_lock(&ctx->lock);
    memmove(data + 0, arr->mem.mem + 0, (size_t) (arr->shape[0] *
                                                  arr->shape[1]) *
            sizeof(float));
    lock_unlock(&ctx->lock);
    return 0;
}
char *futhark_values_raw_f32_2d(struct futhark_context *ctx,
                                struct futhark_f32_2d *arr)
{
    (void) ctx;
    return arr->mem.mem;
}
const int64_t *futhark_shape_f32_2d(struct futhark_context *ctx,
                                    struct futhark_f32_2d *arr)
{
    (void) ctx;
    return arr->shape;
}
int futhark_entry_main(struct futhark_context *ctx,
                       struct futhark_f32_2d **out0, const float in0, const
                       int64_t in1, const float in2, const float in3, const
                       float in4, const float in5, const float in6, const
                       float in7)
{
    float step_sizze_5690;
    int64_t num_steps_5691;
    float init_prey_5692;
    float init_pred_5693;
    float growth_prey_5694;
    float predation_5695;
    float growth_pred_5696;
    float decline_pred_5697;
    struct memblock out_mem_6082;
    
    out_mem_6082.references = NULL;
    
    int ret = 0;
    
    lock_lock(&ctx->lock);
    step_sizze_5690 = in0;
    num_steps_5691 = in1;
    init_prey_5692 = in2;
    init_pred_5693 = in3;
    growth_prey_5694 = in4;
    predation_5695 = in5;
    growth_pred_5696 = in6;
    decline_pred_5697 = in7;
    if (!(true && (true && (true && (true && (true && (true && (true &&
                                                                true)))))))) {
        ret = 1;
        if (!ctx->error)
            ctx->error =
                msgprintf("Error: entry point arguments have invalid sizes.\n");
    } else {
        ret = futrts_main(ctx, &out_mem_6082, step_sizze_5690, num_steps_5691,
                          init_prey_5692, init_pred_5693, growth_prey_5694,
                          predation_5695, growth_pred_5696, decline_pred_5697);
        if (ret == 0) {
            assert((*out0 =
                    (struct futhark_f32_2d *) malloc(sizeof(struct futhark_f32_2d))) !=
                NULL);
            (*out0)->mem = out_mem_6082;
            (*out0)->shape[0] = num_steps_5691;
            (*out0)->shape[1] = 2;
        }
    }
    lock_unlock(&ctx->lock);
    return ret;
}
int futhark_entry_runge_kutta_fwd(struct futhark_context *ctx,
                                  struct futhark_f32_2d **out0, const float in0,
                                  const int64_t in1, const float in2, const
                                  float in3, const float in4, const float in5,
                                  const float in6, const float in7, const
                                  float in8, const float in9, const float in10,
                                  const float in11, const float in12, const
                                  float in13)
{
    float step_sizze_5767;
    int64_t num_steps_5768;
    float init_prey_5769;
    float init_pred_5770;
    float growth_prey_5771;
    float predation_5772;
    float growth_pred_5773;
    float decline_pred_5774;
    float init_prey_tan_5775;
    float init_pred_tan_5776;
    float growth_prey_tan_5777;
    float predation_tan_5778;
    float growth_pred_tan_5779;
    float decline_pred_tan_5780;
    struct memblock out_mem_6082;
    
    out_mem_6082.references = NULL;
    
    int ret = 0;
    
    lock_lock(&ctx->lock);
    step_sizze_5767 = in0;
    num_steps_5768 = in1;
    init_prey_5769 = in2;
    init_pred_5770 = in3;
    growth_prey_5771 = in4;
    predation_5772 = in5;
    growth_pred_5773 = in6;
    decline_pred_5774 = in7;
    init_prey_tan_5775 = in8;
    init_pred_tan_5776 = in9;
    growth_prey_tan_5777 = in10;
    predation_tan_5778 = in11;
    growth_pred_tan_5779 = in12;
    decline_pred_tan_5780 = in13;
    if (!(true && (true && (true && (true && (true && (true && (true && (true &&
                                                                         (true &&
                                                                          (true &&
                                                                           (true &&
                                                                            (true &&
                                                                             (true &&
                                                                              true)))))))))))))) {
        ret = 1;
        if (!ctx->error)
            ctx->error =
                msgprintf("Error: entry point arguments have invalid sizes.\n");
    } else {
        ret = futrts_runge_kutta_fwd(ctx, &out_mem_6082, step_sizze_5767,
                                     num_steps_5768, init_prey_5769,
                                     init_pred_5770, growth_prey_5771,
                                     predation_5772, growth_pred_5773,
                                     decline_pred_5774, init_prey_tan_5775,
                                     init_pred_tan_5776, growth_prey_tan_5777,
                                     predation_tan_5778, growth_pred_tan_5779,
                                     decline_pred_tan_5780);
        if (ret == 0) {
            assert((*out0 =
                    (struct futhark_f32_2d *) malloc(sizeof(struct futhark_f32_2d))) !=
                NULL);
            (*out0)->mem = out_mem_6082;
            (*out0)->shape[0] = num_steps_5768;
            (*out0)->shape[1] = 2;
        }
    }
    lock_unlock(&ctx->lock);
    return ret;
}
