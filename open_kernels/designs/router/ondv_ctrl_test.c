// Host check: the router control-word kernel agrees word for word with the Python
// reference (designs/layer_x/ondv_ctrl_ref.py), on both a small base and a base whose
// high word is non-zero (the pool BO sits above 4 GB on this box, so addr_hi matters).
//
//   python3 ../layer_x/ondv_ctrl_ref.py dump /tmp/ondv_a.bin 0x80000000 0,1,2,3,4,5,6,7
//   python3 ../layer_x/ondv_ctrl_ref.py dump /tmp/ondv_b.bin 0x1_40000000 255,0,17,3,9,64,128,200
//   ./ondv_ctrl_test /tmp/ondv_a.bin 0x80000000 0,1,2,3,4,5,6,7
//   ./ondv_ctrl_test /tmp/ondv_b.bin 0x140000000 255,0,17,3,9,64,128,200
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ondv_ctrl.h"

extern "C" void ondv_ctrl(const uint8_t *rout, const uint32_t *cfg, int32_t *out);

static int read_i32(const char *path, int32_t *buf, size_t n) {
  FILE *f = fopen(path, "rb");
  if (!f) return -1;
  size_t got = fread(buf, sizeof(int32_t), n, f);
  fclose(f);
  return got == n ? 0 : -1;
}

int main(int argc, char **argv) {
  if (argc != 4) {
    fprintf(stderr, "usage: %s <expected.bin> <base> <idx0,..,idx7>\n", argv[0]);
    return 2;
  }
  const char *path = argv[1];
  const unsigned long long base = strtoull(argv[2], NULL, 0);
  int32_t idx[8];
  int k = 0;
  char *s = strdup(argv[3]);
  for (char *t = strtok(s, ","); t && k < 8; t = strtok(NULL, ",")) idx[k++] = (int32_t)strtol(t, NULL, 0);
  free(s);
  if (k != 8) {
    fprintf(stderr, "need 8 indices\n");
    return 2;
  }

  enum { N = 8 * 8 * 13 };
  int32_t want[N];
  if (read_i32(path, want, N) != 0) {
    fprintf(stderr, "cannot read %zu int32 from %s\n", (size_t)N, path);
    return 2;
  }
  int32_t got[N];
  ondv_ctrl_impl(idx, (uint32_t)base, (uint32_t)(base >> 32), got);

  // the ExternalFunction entry reads idx out of the router's 4 KB output at +1024 B and
  // the base out of a 2-word config element -- checked against ondv_ctrl_impl directly
  {
    static uint8_t rout[4096];
    static uint32_t cfg[2];
    int32_t via_entry[N];
    for (int i = 0; i < 8; ++i) ((int32_t *)(rout + 1024))[i] = idx[i];
    cfg[0] = (uint32_t)base;
    cfg[1] = (uint32_t)(base >> 32);
    ondv_ctrl(rout, cfg, via_entry);
    for (int i = 0; i < N; ++i)
      if (via_entry[i] != got[i]) {
        fprintf(stderr, "ondv_ctrl_test: FAIL (entry differs from impl at word %d)\n", i);
        return 1;
      }
  }

  int bad = 0;
  for (int i = 0; i < N; ++i) {
    if (want[i] != got[i]) {
      if (bad < 8)
        fprintf(stderr, "word %3d (offset %u, k%u c%u): got 0x%08X want 0x%08X\n", i, i % 13,
                i / 104, (i / 13) % 8, (unsigned)got[i], (unsigned)want[i]);
      ++bad;
    }
  }
  if (bad) {
    fprintf(stderr, "ondv_ctrl_test: FAIL (%d/%d words differ)\n", bad, N);
    return 1;
  }
  // a couple of structural assertions the reference cannot express
  if ((got[1] & 3u) != 0u || (got[4] & 3u) != 0u || (got[7] & 3u) != 0u) {
    fprintf(stderr, "ondv_ctrl_test: FAIL (w1 addr_low has low bits set)\n");
    return 1;
  }
  printf("ondv_ctrl_test: PASS (%d words, base=0x%llX)\n", N, base);
  return 0;
}
