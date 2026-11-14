// Test-only C ABI for the actual host prefill implementation; no duplicated math.
#include "open_qwen36/block_host.hpp"

extern "C" int prefill_delta(const size_t* c, const float* qkv, const float* z,
    const float* xn, const float* cw, const float* wa, const float* wb,
    const float* a, const float* dt, const float* nw, float* conv, float* state, float* out) {
    try {
        open_qwen36::host::DeltaGeom g;
        g.conv_f32 = true;
        g.T=c[0]; g.t_real=c[1]; g.hid=c[2]; g.key_heads=c[3]; g.value_heads=c[4];
        g.head_dim=c[5]; g.taps=c[6]; g.lanes=c[7]; g.s_rows=c[8];
        open_qwen36::host::deltanet_block(g,qkv,z,xn,cw,wa,wb,a,dt,nw,conv,state,out);
        return 0;
    } catch (...) { return 1; }
}

extern "C" int prefill_attention(const size_t* c, const float* q, const float* k,
    const float* v, const float* gate, const float* qn, const float* kn,
    const double* freq, uint16_t* kv, float* out) {
    try {
        open_qwen36::host::AttnGeom g;
        g.kv_pair=true;
        g.T=c[0]; g.t_real=c[1]; g.nh=c[2]; g.kvh=c[3]; g.hd=c[4]; g.rot=c[5]; g.pos0=c[6];
        open_qwen36::host::attention_block(g,q,k,v,gate,qn,kn,freq,kv,4*g.kvh*g.hd,out);
        return 0;
    } catch (...) { return 1; }
}
