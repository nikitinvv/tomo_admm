#ifndef RADONUSFFT_CUH
#define RADONUSFFT_CUH

#include <cufft.h>

enum dir {
  TOMO_FWD,
  TOMO_ADJ
};

class radonusfft {
  bool is_free = false;

  size_t m;
  float mu;

  float2 **f;
  float2 **g;
  float2 **ff;
  float2 **gg;
  float2 **f0;
  float2 **g0;
  
  float **x;
  float **y;

  float2 **fdee;

  cufftHandle *plan2d;
  
  cufftHandle *plan1d;
  float2 **shiftfwd;
  float2 **shiftadj;

  dim3 BS2d, BS3d, GS2d0, GS3d0, GS3d1, GS3d2, GS3d3;

public:
  size_t n;      // width of square slices
  size_t ntheta; // number of angles
  size_t nz;    // number of slices
  float center;  // location of the rotation center
  size_t ngpus;
  radonusfft(size_t ntheta, size_t nz, size_t n, float center, size_t ngpus_);
  ~radonusfft();
  void fwd(size_t g, size_t f, size_t theta, size_t igpu);
  void adj(size_t f, size_t g, size_t theta, size_t igpu);
  void free();
};

#endif
