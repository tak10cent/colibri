/* Qwen3-MoE inference engine in pure C.
 *
 * Targets the Qwen/Qwen3.6-35B-A3B model (model_type = "qwen3_moe") and the
 * broader Qwen3-MoE family.  Architecture differences from GLM-5.2 that make
 * that engine incompatible:
 *   - Standard GQA attention (not MLA/LoRA).  Separate q_proj/k_proj/v_proj.
 *   - Per-head RMS norms on Q and K (q_norm, k_norm, shape [head_dim]).
 *   - RoPE theta ~1 000 000 (not 10 000).
 *   - Softmax router (not sigmoid) with optional norm_topk_prob.
 *   - One shared expert applied to ALL tokens (tensor name shared_expert,
 *     singular, not shared_experts).
 *   - No MLA latent compression, no DSA indexer, no MTP speculative head.
 *   - Qwen tokenizer (<|im_start|>/<|im_end|> chat format).
 *
 * Implements the same \x01\x01READY\x01\x01 / \x01\x01END\x01\x01 / STAT /
 * \x02PROMPT / \x02RESET wire protocol as glm.c so it works unchanged with
 * openai_server.py and the `coli serve` / `coli chat` commands.
 *
 * Build:  make qwen
 * Run:    SNAP=<dir> ./qwen <cap> [<expert_bits>]
 * Serve:  SNAP=<dir> SERVE=1 KV_SLOTS=1 ./qwen <cap>
 *
 * Limitations:
 *   - Text-only: vision inputs are NOT supported.
 *   - Single decode thread (no batch-decode mux).
 *   - No CUDA/Metal GPU acceleration.
 *   - No speculative / draft decoding.
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <limits.h>
#include <pthread.h>
#if defined(__APPLE__) || defined(__linux__)
#include <sys/resource.h>
#include <sys/mman.h>
#endif
#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif
#include "st.h"
#include "tok.h"
#include "json.h"
#ifdef _OPENMP
#include <omp.h>
#else
static inline int omp_get_max_threads(void){ return 1; }
static inline int omp_get_thread_num(void){ return 0; }
#endif
#ifdef __AVX2__
#include <immintrin.h>
static inline float hsum256(__m256 v){
    __m128 lo=_mm256_castps256_ps128(v), hi=_mm256_extractf128_ps(v,1);
    lo=_mm_add_ps(lo,hi); __m128 sh=_mm_movehl_ps(lo,lo); lo=_mm_add_ps(lo,sh);
    sh=_mm_shuffle_ps(lo,lo,1); lo=_mm_add_ss(lo,sh); return _mm_cvtss_f32(lo);
}
#elif defined(__ARM_NEON)
#include <arm_neon.h>
#endif
#ifdef __APPLE__
#include <mach/mach.h>
#endif

/* ---- config ---- */
typedef struct {
    int hidden, n_layers, n_heads, n_kv_heads, head_dim;
    int n_experts, topk, moe_inter, vocab;
    int n_shared;       /* num_shared_experts (usually 1) */
    int norm_topk;      /* norm_topk_prob */
    float theta, eps;
    int stop_ids[8]; int n_stop;
} Cfg;

/* ---- per-layer dense weights ---- */
typedef struct {
    float *in_ln, *post_ln;
    float *q, *k, *v, *o;     /* attention projections (f32 resident) */
    float *qn, *kn;            /* q_norm / k_norm weights [head_dim] */
    float *gate;               /* router [n_experts, hidden] */
    float *sh_gate, *sh_up, *sh_down; /* shared expert f32 resident */
} Layer;

/* ---- quantised expert slot (int8 + scale) ---- */
typedef struct {
    int eid;
    int8_t *g, *u, *d;
    float  *gs, *us, *ds;
    uint64_t used;
} ESlot;
typedef struct { ESlot *slots; int n, cap; } LCache;

/* ---- model ---- */
typedef struct {
    Cfg c;
    shards S;
    int ebits;
    float *embed, *lm_head, *final_norm;
    Layer *L;
    LCache *cache;
    uint64_t clock, hits, miss;
    /* KV cache: [n_kv_heads, max_t, head_dim] per layer */
    float **K, **V;
    int kv_len, max_t;
    double t_load;
} Model;

/* ---- utility ---- */
static double now_s(void){
    struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t);
    return t.tv_sec + t.tv_nsec*1e-9;
}
static double rss_gb(void){
    struct rusage r; getrusage(RUSAGE_SELF,&r);
#ifdef __APPLE__
    return r.ru_maxrss/(1024.0*1024.0*1024.0);
#else
    return r.ru_maxrss/(1024.0*1024.0);
#endif
}
static float *falloc(int64_t n){
    if(n<=0||(uint64_t)n>SIZE_MAX/sizeof(float)){
        fprintf(stderr,"falloc: n=%lld\n",(long long)n); exit(1);
    }
    float *p=malloc((size_t)n*sizeof(float));
    if(!p){fprintf(stderr,"OOM\n");exit(1);}
    return p;
}

/* ---- matmul: y[S,O] = x[S,I] @ W^T, W[O,I] f32 ---- */
static void matmul(float *y, const float *x, const float *W, int S, int I, int O){
    #pragma omp parallel for schedule(static)
    for(int o=0;o<O;o++){
        const float *w=W+(int64_t)o*I;
        for(int s=0;s<S;s++){
            const float *xs=x+(int64_t)s*I; float a=0;
            for(int i=0;i<I;i++) a+=xs[i]*w[i];
            y[(int64_t)s*O+o]=a;
        }
    }
}

/* ---- matmul_q: y[O] = x[I] @ W^T with int8 per-row quant ---- */
static void matmul_q(float *y, const float *x, const int8_t *q, const float *sc, int I, int O){
    #pragma omp parallel for schedule(static)
    for(int o=0;o<O;o++){
        const int8_t *w=q+(int64_t)o*I; float a=0;
#ifdef __AVX2__
        __m256 acc=_mm256_setzero_ps(); int i=0;
        for(;i+8<=I;i+=8){
            __m256i wi=_mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*)(w+i)));
            acc=_mm256_fmadd_ps(_mm256_loadu_ps(x+i),_mm256_cvtepi32_ps(wi),acc);
        }
        a=hsum256(acc);
        for(;i<I;i++) a+=x[i]*(float)w[i];
#elif defined(__ARM_NEON)
        float32x4_t ac0=vdupq_n_f32(0),ac1=vdupq_n_f32(0); int i=0;
        for(;i+8<=I;i+=8){
            int16x8_t w16=vmovl_s8(vld1_s8(w+i));
            ac0=vfmaq_f32(ac0,vld1q_f32(x+i),  vcvtq_f32_s32(vmovl_s16(vget_low_s16(w16))));
            ac1=vfmaq_f32(ac1,vld1q_f32(x+i+4),vcvtq_f32_s32(vmovl_s16(vget_high_s16(w16))));
        }
        a=vaddvq_f32(vaddq_f32(ac0,ac1));
        for(;i<I;i++) a+=x[i]*(float)w[i];
#else
        for(int i=0;i<I;i++) a+=x[i]*(float)w[i];
#endif
        y[o]=a*sc[o];
    }
}

/* ---- row-wise int8 quantisation ---- */
static void quantize_rows(const float *w, int8_t *q, float *sc, int O, int I, int bits){
    int qmax=(1<<(bits-1))-1;
    #pragma omp parallel for schedule(static)
    for(int o=0;o<O;o++){
        const float *wr=w+(int64_t)o*I; float am=0;
        for(int i=0;i<I;i++){float a=fabsf(wr[i]); if(a>am)am=a;}
        float s=am/qmax; if(s<1e-8f)s=1e-8f; sc[o]=s;
        int8_t *qr=q+(int64_t)o*I;
        for(int i=0;i<I;i++){
            int v=(int)lrintf(wr[i]/s);
            if(v>qmax)v=qmax; if(v<-qmax-1)v=-qmax-1; qr[i]=(int8_t)v;
        }
    }
}

/* ---- RMS norm: out = x * w / rms(x), in-place safe ---- */
static void rmsnorm(float *out, const float *x, const float *w, int D, float eps){
    double ms=0; for(int i=0;i<D;i++) ms+=(double)x[i]*x[i];
    float r=1.f/sqrtf((float)(ms/D)+eps);
    for(int i=0;i<D;i++) out[i]=x[i]*r*w[i];
}

static void softmax(float *x, int n){
    float m=-1e30f; for(int i=0;i<n;i++) if(x[i]>m)m=x[i];
    float s=0; for(int i=0;i<n;i++){x[i]=expf(x[i]-m);s+=x[i];}
    for(int i=0;i<n;i++) x[i]/=s;
}

/* ---- config loading ---- */
static int gi(jval *r, const char *k){
    jval *v=json_get(r,k); return v?(int)v->num:0;
}

static void load_cfg(Cfg *c, const char *snap){
    char p[2048]; snprintf(p,sizeof(p),"%s/config.json",snap);
    FILE *f=fopen(p,"rb"); if(!f){perror(p);exit(1);}
    fseek(f,0,SEEK_END); long n=ftell(f); fseek(f,0,SEEK_SET);
    char *buf=malloc(n+1); size_t got=fread(buf,1,n,f); buf[got]=0; fclose(f);
    char *arena=NULL; jval *r=json_parse(buf,&arena);
    c->hidden       = gi(r,"hidden_size");
    c->n_layers     = gi(r,"num_hidden_layers");
    c->n_heads      = gi(r,"num_attention_heads");
    c->n_kv_heads   = gi(r,"num_key_value_heads"); if(c->n_kv_heads<1)c->n_kv_heads=c->n_heads;
    c->n_experts    = gi(r,"n_routed_experts"); if(!c->n_experts)c->n_experts=gi(r,"num_experts");
    c->topk         = gi(r,"num_experts_per_tok");
    c->moe_inter    = gi(r,"moe_intermediate_size");
    c->n_shared     = gi(r,"num_shared_experts"); if(c->n_shared<0)c->n_shared=0;
    c->vocab        = gi(r,"vocab_size");
    /* head_dim: explicit field if present, else hidden/n_heads */
    {jval *hd=json_get(r,"head_dim"); c->head_dim=hd?(int)hd->num:c->hidden/c->n_heads;}
    if(c->head_dim<1)c->head_dim=c->hidden/c->n_heads;
    jval *th=json_get(r,"rope_theta"); c->theta=th?(float)th->num:1000000.f;
    jval *ep=json_get(r,"rms_norm_eps"); c->eps=ep?(float)ep->num:1e-6f;
    jval *nt=json_get(r,"norm_topk_prob"); c->norm_topk=(nt&&nt->t==J_BOOL)?nt->boolean:0;
    /* stop tokens */
    c->n_stop=0;
    jval *eo=json_get(r,"eos_token_id");
    if(eo){
        if(eo->t==J_NUM) c->stop_ids[c->n_stop++]=(int)eo->num;
        else if(eo->t==J_ARR) for(int i=0;i<eo->len&&c->n_stop<8;i++)
            c->stop_ids[c->n_stop++]=(int)eo->kids[i]->num;
    }
    /* validation */
#define CKR(name,v,lo,hi) if((v)<(lo)||(v)>(hi)){\
    fprintf(stderr,"qwen config: %s=%d outside [%d,%d]\n",name,(int)(v),(int)(lo),(int)(hi));exit(1);}
    CKR("hidden_size",c->hidden,1,1<<20)
    CKR("num_hidden_layers",c->n_layers,1,512)
    CKR("num_attention_heads",c->n_heads,1,1024)
    CKR("num_key_value_heads",c->n_kv_heads,1,c->n_heads)
    CKR("head_dim",c->head_dim,1,1<<16)
    CKR("n_routed_experts",c->n_experts,1,4096)
    CKR("num_experts_per_tok",c->topk,1,64)
    CKR("moe_intermediate_size",c->moe_inter,1,1<<20)
    CKR("vocab_size",c->vocab,1,1<<24)
#undef CKR
    free(buf); free(arena);
}

/* ---- weight loading ---- */
static float *ld_f32(Model *m, const char *name){
    int64_t n=st_numel(&m->S,name);
    if(n<0){fprintf(stderr,"qwen: missing tensor %s\n",name);exit(1);}
    float *p=falloc(n); st_read_f32(&m->S,name,p,0); return p;
}

/* load expert weight: pre-quant container OR on-the-fly from f32/bf16 */
static void load_expert_w(Model *m, const char *name, int8_t *q, float *sc, int O, int I, float *tmp){
    char qs[320]; snprintf(qs,sizeof(qs),"%s.qs",name);
    if(st_has(&m->S,qs)){
        st_read_raw(&m->S,name,q,1);
        st_read_f32(&m->S,qs,sc,1);
        return;
    }
    st_read_f32(&m->S,name,tmp,1);
    quantize_rows(tmp,q,sc,O,I,m->ebits);
}

/* ---- expert cache ---- */
static void expert_get(Model *m, int layer, int eid, ESlot **out){
    LCache *lc=&m->cache[layer];
    for(int i=0;i<lc->n;i++) if(lc->slots[i].eid==eid){
        m->hits++; lc->slots[i].used=++m->clock; *out=&lc->slots[i]; return;
    }
    m->miss++;
    Cfg *c=&m->c;
    int64_t ng=(int64_t)c->moe_inter*c->hidden;
    int64_t nd=(int64_t)c->hidden*c->moe_inter;
    ESlot *s;
    if(lc->n<lc->cap){
        s=&lc->slots[lc->n++];
        s->g=malloc(ng); s->u=malloc(ng); s->d=malloc(nd);
        s->gs=falloc(c->moe_inter); s->us=falloc(c->moe_inter); s->ds=falloc(c->hidden);
    } else {
        int lru=0; for(int i=1;i<lc->n;i++) if(lc->slots[i].used<lc->slots[lru].used) lru=i;
        s=&lc->slots[lru];
    }
    float *tmp=falloc(ng>nd?ng:nd);
    char nm[320];
    snprintf(nm,sizeof(nm),"model.layers.%d.mlp.experts.%d.gate_proj.weight",layer,eid);
    load_expert_w(m,nm,s->g,s->gs,c->moe_inter,c->hidden,tmp);
    snprintf(nm,sizeof(nm),"model.layers.%d.mlp.experts.%d.up_proj.weight",layer,eid);
    load_expert_w(m,nm,s->u,s->us,c->moe_inter,c->hidden,tmp);
    snprintf(nm,sizeof(nm),"model.layers.%d.mlp.experts.%d.down_proj.weight",layer,eid);
    load_expert_w(m,nm,s->d,s->ds,c->hidden,c->moe_inter,tmp);
    free(tmp);
    s->eid=eid; s->used=++m->clock; *out=s;
}

/* ---- model init ---- */
static void model_init(Model *m, const char *snap, int cap, int ebits){
    memset(m,0,sizeof(*m)); m->ebits=ebits;
    load_cfg(&m->c,snap); st_init(&m->S,snap);
    Cfg *c=&m->c;
    double t0=now_s();
    m->embed      = ld_f32(m,"model.embed_tokens.weight");
    m->lm_head    = ld_f32(m,"lm_head.weight");
    m->final_norm = ld_f32(m,"model.norm.weight");
    m->L=calloc(c->n_layers,sizeof(Layer));
    char nm[320];
    for(int i=0;i<c->n_layers;i++){
        Layer *l=&m->L[i];
#define LD(field,sfx) snprintf(nm,sizeof(nm),"model.layers.%d." sfx,i); l->field=ld_f32(m,nm)
        LD(in_ln,  "input_layernorm.weight");
        LD(post_ln,"post_attention_layernorm.weight");
        LD(q, "self_attn.q_proj.weight");
        LD(k, "self_attn.k_proj.weight");
        LD(v, "self_attn.v_proj.weight");
        LD(o, "self_attn.o_proj.weight");
        LD(qn,"self_attn.q_norm.weight");
        LD(kn,"self_attn.k_norm.weight");
        LD(gate,"mlp.gate.weight");
        LD(sh_gate,"mlp.shared_expert.gate_proj.weight");
        LD(sh_up,  "mlp.shared_expert.up_proj.weight");
        LD(sh_down,"mlp.shared_expert.down_proj.weight");
#undef LD
    }
    m->cache=calloc(c->n_layers,sizeof(LCache));
    for(int i=0;i<c->n_layers;i++){
        m->cache[i].cap=cap;
        m->cache[i].slots=calloc(cap,sizeof(ESlot));
    }
    m->t_load=now_s()-t0;
}

/* ---- RoPE: standard (non-interleaved) half-half rotation ---- */
static void rope_head(float *x, int pos, int hd, float theta){
    int h=hd/2;
    for(int j=0;j<h;j++){
        float inv=powf(theta,-2.f*j/hd);
        float ang=pos*inv, cs=cosf(ang), sn=sinf(ang);
        float a=x[j], b=x[j+h];
        x[j]=a*cs-b*sn; x[j+h]=b*cs+a*sn;
    }
}

/* ---- GQA attention (S new tokens at pos_base) ---- */
static void attention(Model *m, Layer *l, int layer,
                      const float *x, int S, int pos_base, float *out){
    Cfg *c=&m->c;
    int H=c->n_heads, Hkv=c->n_kv_heads, hd=c->head_dim, D=c->hidden;
    int gqa_ratio=H/Hkv;                      /* heads per KV group */
    int Qd=H*hd, Kd=Hkv*hd;
    float *q=falloc((int64_t)S*Qd);
    float *k=falloc((int64_t)S*Kd);
    float *v=falloc((int64_t)S*Kd);
    float *ctx=falloc((int64_t)S*Qd);
    /* projections */
    matmul(q,x,l->q,S,D,Qd);
    matmul(k,x,l->k,S,D,Kd);
    matmul(v,x,l->v,S,D,Kd);
    /* per-head q_norm, k_norm, then RoPE */
    for(int s=0;s<S;s++){
        float *qs=q+(int64_t)s*Qd;
        float *ks=k+(int64_t)s*Kd;
        int pos=pos_base+s;
        for(int hh=0;hh<H;hh++){
            rmsnorm(qs+hh*hd, qs+hh*hd, l->qn, hd, c->eps);
            rope_head(qs+hh*hd, pos, hd, c->theta);
        }
        for(int hh=0;hh<Hkv;hh++){
            rmsnorm(ks+hh*hd, ks+hh*hd, l->kn, hd, c->eps);
            rope_head(ks+hh*hd, pos, hd, c->theta);
        }
    }
    /* write into KV cache */
    for(int s=0;s<S;s++){
        int t=pos_base+s;
        float *ks=k+(int64_t)s*Kd;
        float *vs=v+(int64_t)s*Kd;
        for(int hh=0;hh<Hkv;hh++){
            memcpy(m->K[layer]+((int64_t)hh*m->max_t+t)*hd, ks+hh*hd, hd*sizeof(float));
            memcpy(m->V[layer]+((int64_t)hh*m->max_t+t)*hd, vs+hh*hd, hd*sizeof(float));
        }
    }
    float scale=1.f/sqrtf((float)hd);
    /* per-thread score scratch: one buffer of max_t floats per thread */
    int nthreads=omp_get_max_threads();
    float **sc_bufs=(float**)calloc(nthreads,sizeof(float*));
    for(int i=0;i<nthreads;i++) sc_bufs[i]=(float*)falloc(m->max_t);
    /* attention kernel */
    #pragma omp parallel for collapse(2) schedule(static)
    for(int hh=0;hh<H;hh++){
        for(int s=0;s<S;s++){
            int kvh=hh/gqa_ratio;
            int qpos=pos_base+s;
            const float *qv=q+(int64_t)s*Qd+hh*hd;
            float *sc=sc_bufs[omp_get_thread_num()];
            for(int t=0;t<=qpos;t++){
                const float *kv=m->K[layer]+((int64_t)kvh*m->max_t+t)*hd;
                float a=0; for(int dd=0;dd<hd;dd++) a+=qv[dd]*kv[dd];
                sc[t]=a*scale;
            }
            softmax(sc,qpos+1);
            float *cx=ctx+(int64_t)s*Qd+hh*hd;
            memset(cx,0,hd*sizeof(float));
            for(int t=0;t<=qpos;t++){
                const float *vv=m->V[layer]+((int64_t)kvh*m->max_t+t)*hd;
                float a=sc[t]; for(int dd=0;dd<hd;dd++) cx[dd]+=a*vv[dd];
            }
        }
    }
    for(int i=0;i<nthreads;i++) free(sc_bufs[i]);
    free(sc_bufs);
    matmul(out,ctx,l->o,S,Qd,D);
    free(q); free(k); free(v); free(ctx);
}

/* ---- SiLU activation: x = silu(gate) * up ---- */
static void silu_mul(float *g, const float *u, int I){
    for(int i=0;i<I;i++){ float gv=g[i]; g[i]=(gv/(1.f+expf(-gv)))*u[i]; }
}

/* ---- MoE forward for one token at index s in the S-token batch ---- */
static void moe_token(Model *m, Layer *l, int layer, const float *xs, float *os){
    Cfg *c=&m->c;
    int D=c->hidden, E=c->n_experts, K=c->topk, I=c->moe_inter;
    /* router */
    float *logits=falloc(E);
    matmul(logits,xs,l->gate,1,D,E);
    softmax(logits,E);
    /* top-K selection */
    int idx[64]; float val[64];
    for(int kk=0;kk<K;kk++){
        int best=-1; float bv=-1e30f;
        for(int e=0;e<E;e++){
            int taken=0; for(int j=0;j<kk;j++) if(idx[j]==e){taken=1;break;}
            if(!taken&&logits[e]>bv){bv=logits[e];best=e;}
        }
        idx[kk]=best; val[kk]=bv;
    }
    if(c->norm_topk){ float sm=0; for(int kk=0;kk<K;kk++) sm+=val[kk];
        if(sm>1e-8f) for(int kk=0;kk<K;kk++) val[kk]/=sm; }
    /* routed experts */
    float *g=falloc(I), *u=falloc(I), *hh=falloc(D);
    memset(os,0,D*sizeof(float));
    for(int kk=0;kk<K;kk++){
        ESlot *e; expert_get(m,layer,idx[kk],&e);
        matmul_q(g,xs,e->g,e->gs,D,I);
        matmul_q(u,xs,e->u,e->us,D,I);
        silu_mul(g,u,I);
        matmul_q(hh,g,e->d,e->ds,I,D);
        float w=val[kk]; for(int d=0;d<D;d++) os[d]+=w*hh[d];
    }
    /* shared expert (always applied) */
    if(c->n_shared>0){
        matmul(g,xs,l->sh_gate,1,D,I);
        matmul(u,xs,l->sh_up,  1,D,I);
        silu_mul(g,u,I);
        matmul(hh,g,l->sh_down,1,I,D);
        for(int d=0;d<D;d++) os[d]+=hh[d];
    }
    free(logits); free(g); free(u); free(hh);
}

/* ---- forward pass: ids[S] at pos_base, returns logits of last token (heap) ---- */
static float *step(Model *m, const int *ids, int S, int pos_base){
    Cfg *c=&m->c; int D=c->hidden;
    float *x=falloc((int64_t)S*D);
    for(int s=0;s<S;s++) memcpy(x+(int64_t)s*D, m->embed+(int64_t)ids[s]*D, D*sizeof(float));
    float *nrm=falloc((int64_t)S*D), *tmp=falloc((int64_t)S*D);
    for(int i=0;i<c->n_layers;i++){
        Layer *l=&m->L[i];
        /* pre-attention norm */
        for(int s=0;s<S;s++) rmsnorm(nrm+(int64_t)s*D, x+(int64_t)s*D, l->in_ln, D, c->eps);
        attention(m,l,i,nrm,S,pos_base,tmp);
        for(int64_t j=0;j<(int64_t)S*D;j++) x[j]+=tmp[j];
        /* pre-MLP norm */
        for(int s=0;s<S;s++) rmsnorm(nrm+(int64_t)s*D, x+(int64_t)s*D, l->post_ln, D, c->eps);
        /* MoE per token */
        for(int s=0;s<S;s++) moe_token(m,l,i, nrm+(int64_t)s*D, tmp+(int64_t)s*D);
        for(int64_t j=0;j<(int64_t)S*D;j++) x[j]+=tmp[j];
    }
    m->kv_len=pos_base+S;
    float *last=falloc(D);
    rmsnorm(last, x+(int64_t)(S-1)*D, m->final_norm, D, c->eps);
    float *logit=falloc(c->vocab);
    matmul(logit,last,m->lm_head,1,D,c->vocab);
    free(x); free(nrm); free(tmp); free(last);
    return logit;
}

/* ---- KV cache allocation ---- */
static void kv_alloc(Model *m, int max_t){
    Cfg *c=&m->c; m->max_t=max_t;
    m->K=calloc(c->n_layers,sizeof(float*));
    m->V=calloc(c->n_layers,sizeof(float*));
    for(int i=0;i<c->n_layers;i++){
        m->K[i]=falloc((int64_t)c->n_kv_heads*max_t*c->head_dim);
        m->V[i]=falloc((int64_t)c->n_kv_heads*max_t*c->head_dim);
    }
}

static void kv_free(Model *m){
    Cfg *c=&m->c;
    if(m->K){ for(int i=0;i<c->n_layers;i++) free(m->K[i]); free(m->K); m->K=NULL; }
    if(m->V){ for(int i=0;i<c->n_layers;i++) free(m->V[i]); free(m->V); m->V=NULL; }
    m->kv_len=0; m->max_t=0;
}

/* ---- sampling ---- */
static float g_temp=0.7f, g_nuc=0.9f;

static int pick_tok(const float *logits, int vocab){
    if(g_temp<=0.f){
        int best=0; float bv=logits[0];
        for(int i=1;i<vocab;i++) if(logits[i]>bv){bv=logits[i];best=i;}
        return best;
    }
    /* temperature + nucleus */
    float *p=falloc(vocab);
    float m=-1e30f; for(int i=0;i<vocab;i++) if(logits[i]>m) m=logits[i];
    float s=0; for(int i=0;i<vocab;i++){p[i]=expf((logits[i]-m)/g_temp);s+=p[i];}
    for(int i=0;i<vocab;i++) p[i]/=s;
    /* nucleus cutoff: partial selection sort (O(n*k), k<<n) */
    int *idx=malloc(vocab*sizeof(int)); for(int i=0;i<vocab;i++) idx[i]=i;
    float cum=0; int k=0;
    while(k<vocab && cum<g_nuc){
        /* find max in remaining elements [k, vocab) */
        int best=k;
        for(int j=k+1;j<vocab;j++) if(p[idx[j]]>p[idx[best]]) best=j;
        int tmp=idx[k]; idx[k]=idx[best]; idx[best]=tmp;
        cum+=p[idx[k]]; k++;
    }
    if(k<1) k=1;
    /* sample */
    float r=(float)rand()/RAND_MAX*cum;
    float cs=0; int tok=idx[0];
    for(int i=0;i<k;i++){ cs+=p[idx[i]]; if(r<=cs){tok=idx[i];break;} }
    free(p); free(idx);
    return tok;
}

static int is_stop(Model *m, int id){
    for(int i=0;i<m->c.n_stop;i++) if(m->c.stop_ids[i]==id) return 1;
    return 0;
}

/* ---- tokenizer helpers ---- */
static int tok_stop(Tok *T, Model *m){
    /* prefer <|im_end|> if present, else first EOS from config */
    int id=tok_id_of(T,"<|im_end|>"); if(id>=0) return id;
    if(m->c.n_stop>0) return m->c.stop_ids[0];
    return -1;
}

/* ---- SCORE mode (log-prob for each request on stdin) ---- */
static void run_score(Model *m, const char *snap){
    char tkp[2048]; snprintf(tkp,sizeof(tkp),"%s/tokenizer.json",snap);
    Tok T; tok_load(&T,tkp);
    const char *rpath=getenv("SCORE"); if(!rpath) return;
    FILE *rf=fopen(rpath,"r"); if(!rf){perror(rpath);exit(1);}
    int maxctx=getenv("CTX")?atoi(getenv("CTX")):4096;
    kv_alloc(m,maxctx);
    char *line=NULL; size_t cap=0; ssize_t nr;
    while((nr=getline(&line,&cap,rf))>0){
        if(nr>0&&line[nr-1]=='\n') line[--nr]=0;
        if(nr<1) continue;
        int *ids=malloc(maxctx*sizeof(int));
        int n=tok_encode(&T,line,(int)nr,ids,maxctx);
        if(n<2){printf("0.0\n");fflush(stdout);free(ids);continue;}
        /* clear KV */
        for(int i=0;i<m->c.n_layers;i++){
            memset(m->K[i],0,(int64_t)m->c.n_kv_heads*m->max_t*m->c.head_dim*sizeof(float));
            memset(m->V[i],0,(int64_t)m->c.n_kv_heads*m->max_t*m->c.head_dim*sizeof(float));
        }
        m->kv_len=0;
        float *logit=step(m,ids,n-1,0);
        double lp=0;
        for(int t=1;t<n;t++){
            int prev_pos=t-1;
            float *lo=(t==1)?logit:NULL;
            if(t>1){ lo=step(m,ids+t-1,1,prev_pos); }
            /* log-softmax of ids[t] */
            float mx=-1e30f; for(int v=0;v<m->c.vocab;v++) if(lo[v]>mx)mx=lo[v];
            float s=0; for(int v=0;v<m->c.vocab;v++) s+=expf(lo[v]-mx);
            lp+=lo[ids[t]]-mx-logf(s);
            free(lo);
        }
        printf("%.6f\n",lp); fflush(stdout);
        free(ids);
    }
    free(line); fclose(rf);
}

/* ---- serve loop: implements the \x02PROMPT wire protocol ---- */
static void run_serve(Model *m, const char *snap){
#ifdef _WIN32
    _setmode(_fileno(stdin), _O_BINARY);
    _setmode(_fileno(stdout), _O_BINARY);
    setvbuf(stdout, NULL, _IONBF, 0);
#endif
    char tkp[2048]; snprintf(tkp,sizeof(tkp),"%s/tokenizer.json",snap);
    Tok T; tok_load(&T,tkp);
    int eos=tok_stop(&T,m);
    int ngen=getenv("NGEN")?atoi(getenv("NGEN")):256;
    int maxctx=getenv("CTX")?atoi(getenv("CTX")):4096;
    float def_temp=g_temp, def_nuc=g_nuc;
    kv_alloc(m,maxctx);
    int hist_cap=maxctx+16;
    int *hist=malloc(hist_cap*sizeof(int));
    int hist_len=0;
    /* signal ready */
    printf("\x01\x01" "READY" "\x01\x01\n");
    printf("STAT 0 0.00 0.0 %.2f\n", rss_gb());
    fflush(stdout);
    char *line=NULL; size_t lcap=0; ssize_t nr;
    while((nr=getline(&line,&lcap,stdin))>0){
        if(nr>0&&line[nr-1]=='\n') line[--nr]=0;
        /* RESET: clear KV state */
        if(!strcmp(line,"\x02RESET")){
            hist_len=0;
            for(int i=0;i<m->c.n_layers;i++){
                memset(m->K[i],0,(int64_t)m->c.n_kv_heads*m->max_t*m->c.head_dim*sizeof(float));
                memset(m->V[i],0,(int64_t)m->c.n_kv_heads*m->max_t*m->c.head_dim*sizeof(float));
            }
            m->kv_len=0;
            printf("\x01\x01" "END" "\x01\x01\n");
            printf("STAT 0 0.00 0.0 %.2f\n",rss_gb()); fflush(stdout);
            continue;
        }
        if(nr<1){
            printf("\x01\x01" "END" "\x01\x01\n");
            printf("STAT 0 0.00 0.0 %.2f\n",rss_gb()); fflush(stdout);
            continue;
        }
        /* \x02PROMPT <bytes> <max_tokens> <temperature> <top_p> [kv_slot]\n<prompt>\n */
        char *raw=NULL; int raw_mode=0, req_ngen=ngen, prompt_tokens=0;
        if(!strncmp(line,"\x02PROMPT ",8)){
            unsigned long long nb=0; double rt=0, rp=0; int slot=0;
            int nf=sscanf(line+8,"%llu %d %lf %lf %d",&nb,&req_ngen,&rt,&rp,&slot);
            if(nf<4||nb>(16u<<20)||req_ngen<1||rt<0||rt>2||rp<=0||rp>1){
                printf("\x01\x01" "END" "\x01\x01\n");
                printf("STAT 0 0.00 0.0 %.2f 0 0\n",rss_gb()); fflush(stdout); continue;
            }
            raw=malloc((size_t)nb+1); if(!raw){fprintf(stderr,"OOM\n");exit(1);}
            if(fread(raw,1,(size_t)nb,stdin)!=(size_t)nb){free(raw);break;}
            int delim=fgetc(stdin); if(delim!='\n'&&delim!=EOF) ungetc(delim,stdin);
            if(memchr(raw,0,(size_t)nb)){free(raw);
                printf("\x01\x01" "END" "\x01\x01\n");
                printf("STAT 0 0.00 0.0 %.2f 0 0\n",rss_gb()); fflush(stdout); continue;}
            raw[nb]=0; raw_mode=1;
            if(req_ngen>ngen) req_ngen=ngen;
            g_temp=(float)rt; g_nuc=(float)rp;
        }
        const char *input=raw_mode?raw:line;
        int input_n=raw_mode?(int)strlen(raw):(int)nr;
        /* tokenize and match KV prefix */
        int *tmp=malloc((maxctx+8)*sizeof(int));
        prompt_tokens=tok_encode(&T,input,input_n,tmp,maxctx-4);
        if(prompt_tokens<1){
            free(tmp); if(raw)free(raw); g_temp=def_temp; g_nuc=def_nuc;
            printf("\x01\x01" "END" "\x01\x01\n");
            printf("STAT 0 0.00 0.0 %.2f 0 0\n",rss_gb()); fflush(stdout); continue;
        }
        /* find longest common prefix with current KV */
        int prefix=0;
        while(prefix<hist_len&&prefix<prompt_tokens&&hist[prefix]==tmp[prefix]) prefix++;
        if(prefix<hist_len){
            /* KV rollback: wipe past the prefix by zeroing tail (simple but correct) */
            hist_len=prefix; m->kv_len=prefix;
        }
        int k=prompt_tokens-prefix;
        if(k>0) memcpy(hist+hist_len, tmp+prefix, k*sizeof(int));
        fprintf(stderr,"[API] prefix %d/%d token, prefill %d\n",prefix,prompt_tokens,k);
        free(tmp);
        /* prefill */
        uint64_t h0=m->hits, ms0=m->miss; double tt0=now_s();
        float *logit;
        if(k>0){ logit=step(m,hist+hist_len,k,hist_len); hist_len+=k; }
        else { logit=step(m,hist+hist_len-1,1,hist_len-1); }
        int prod=0; int cur=req_ngen;
        if(hist_len+cur+2>=maxctx) cur=maxctx-hist_len-2;
        /* decode: free+NULL each logit immediately after sampling to prevent
         * double-free — break paths skip the step() call so logit stays NULL,
         * and free(NULL) at line 700 is a safe no-op. */
        while(cur>0){
            int t=pick_tok(logit,m->c.vocab); free(logit); logit=NULL;
            if(t==eos||is_stop(m,t)) break;
            /* emit token bytes */
            char dec[64]; int dn=tok_decode(&T,&t,1,dec,63); dec[dn]=0;
            if(dn>0) fwrite(dec,1,(size_t)dn,stdout);
            fflush(stdout);
            hist[hist_len++]=t; prod++; cur--;
            if(hist_len+2>=hist_cap) break;
            logit=step(m,&t,1,hist_len-1);
        }
        free(logit); logit=NULL; /* no-op if EOS/cap break already freed it */
        double tdt=now_s()-tt0; if(tdt<1e-6) tdt=1e-6;
        double dh=(double)(m->hits-h0), dm=(double)(m->miss-ms0);
        printf("\x01\x01" "END" "\x01\x01\n");
        printf("STAT %d %.2f %.1f %.2f %d %d\n",
               prod, prod/tdt, (dh+dm)>0?100.0*dh/(dh+dm):0.0, rss_gb(),
               prompt_tokens, prod>=req_ngen);
        fflush(stdout);
        if(raw) free(raw); g_temp=def_temp; g_nuc=def_nuc;
    }
    free(line); free(hist);
}

/* ---- interactive/text mode ---- */
static void run_text(Model *m, const char *snap){
    char tkp[2048]; snprintf(tkp,sizeof(tkp),"%s/tokenizer.json",snap);
    Tok T; tok_load(&T,tkp);
    int eos=tok_stop(&T,m);
    const char *prompt=getenv("PROMPT");
    int ngen=getenv("NGEN")?atoi(getenv("NGEN")):64;
    int maxctx=getenv("CTX")?atoi(getenv("CTX")):4096;
    int np; int *pids=malloc((maxctx+8)*sizeof(int));
    np=tok_encode(&T,prompt,(int)strlen(prompt),pids,maxctx);
    if(np<1){fprintf(stderr,"empty prompt\n");return;}
    printf("prompt: %d tokens | generating up to %d\n",np,ngen);
    fputs(prompt,stdout); fflush(stdout);
    kv_alloc(m,np+ngen+4);
    float *logit=step(m,pids,np,0);
    /* free+NULL each logit immediately after sampling; free(NULL) is safe */
    for(int s=0;s<ngen;s++){
        int t=pick_tok(logit,m->c.vocab); free(logit); logit=NULL;
        if(t==eos||is_stop(m,t)) break;
        char dec[64]; int dn=tok_decode(&T,&t,1,dec,63); dec[dn]=0;
        fputs(dec,stdout); fflush(stdout);
        logit=step(m,&t,1,np+s);
    }
    free(logit); putchar('\n'); /* no-op if EOS break already freed it */
    free(pids); kv_free(m);
}

/* ---- main ---- */
int main(int argc, char **argv){
    const char *snap=getenv("SNAP");
    if(!snap){fprintf(stderr,"set SNAP=<model directory>\n");return 1;}
    int cap   = argc>1 ? atoi(argv[1]) : 16;
    int ebits = argc>2 ? atoi(argv[2]) : 8;
    if(ebits<2||ebits>8){fprintf(stderr,"expert_bits must be 2..8\n");return 1;}

    const char *te=getenv("TEMP"); if(te) g_temp=(float)atof(te);
    const char *ne=getenv("NUCLEUS"); if(ne) g_nuc=(float)atof(ne);

    fprintf(stderr,"[qwen] loading model from %s (cap=%d ebits=%d)\n",snap,cap,ebits);
    Model m; model_init(&m,snap,cap,ebits);
    fprintf(stderr,"[qwen] dense weights loaded in %.1fs | arch: %dL %dH %dKVH hd=%d | "
        "%d experts top%d | RSS %.2f GB\n",
        m.t_load, m.c.n_layers, m.c.n_heads, m.c.n_kv_heads, m.c.head_dim,
        m.c.n_experts, m.c.topk, rss_gb());

    /* SCORE mode: log-prob scoring for eval harness */
    if(getenv("SCORE")){ run_score(&m,snap); return 0; }
    /* PROMPT mode: one-shot text generation */
    if(getenv("PROMPT")){ run_text(&m,snap); return 0; }
    /* SERVE mode: OpenAI-compatible API protocol */
    if(getenv("SERVE")){ run_serve(&m,snap); return 0; }

    /* default: ref-test mode (validate against ref.json like olmoe.c) */
    const char *refpath = argc>3 ? argv[3] : "ref.json";
    FILE *f=fopen(refpath,"rb");
    if(!f){
        fprintf(stderr,"Usage: SNAP=<dir> SERVE=1 ./qwen <cap> [<ebits>]\n"
                       "       SNAP=<dir> PROMPT='hello' ./qwen <cap>\n"
                       "Ref file not found: %s (pass as argv[3] or set SERVE/PROMPT)\n",refpath);
        return 1;
    }
    fseek(f,0,SEEK_END); long n=ftell(f); fseek(f,0,SEEK_SET);
    char *buf=malloc(n+1); if(fread(buf,1,n,f)!=(size_t)n){} buf[n]=0; fclose(f);
    char *arena=NULL; jval *ref=json_parse(buf,&arena);
    jval *pa=json_get(ref,"prompt_ids"), *fa=json_get(ref,"full_ids");
    if(!pa||!fa){fprintf(stderr,"ref.json needs prompt_ids and full_ids\n");return 1;}
    int np=(int)pa->len, nfull=(int)fa->len;
    int *prompt=malloc(np*sizeof(int)), *full=malloc(nfull*sizeof(int));
    for(int i=0;i<np;i++) prompt[i]=(int)pa->kids[i]->num;
    for(int i=0;i<nfull;i++) full[i]  =(int)fa->kids[i]->num;
    int n_new=nfull-np;
    kv_alloc(&m,nfull+4);
    int *out=malloc((np+n_new)*sizeof(int));
    for(int i=0;i<np;i++) out[i]=prompt[i];
    double t=now_s();
    float *logit=step(&m,prompt,np,0);
    for(int s=0;s<n_new;s++){
        int best=0; float bv=logit[0];
        for(int i=1;i<m.c.vocab;i++) if(logit[i]>bv){bv=logit[i];best=i;}
        free(logit); out[np+s]=best;
        if(s==n_new-1) break;
        logit=step(&m,out+np+s,1,np+s);
    }
    double dt=now_s()-t;
    int match=0;
    printf("\nReference: "); for(int i=0;i<n_new;i++) printf("%d ",full[np+i]);
    printf("\nC engine : "); for(int i=0;i<n_new;i++){printf("%d ",out[np+i]);if(out[np+i]==full[np+i])match++;}
    printf("\nMatching: %d/%d | %.2f tok/s\n",match,n_new,n_new/dt);
    double tot=m.hits+m.miss;
    printf("Expert cache hit rate: %.1f%%\n",tot?100.0*m.hits/tot:0.0);
    free(buf); free(arena); free(prompt); free(full); free(out);
    return match==n_new?0:1;
}
