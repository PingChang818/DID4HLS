//------------------------------------------------------------------------
// covariance
//------------------------------------------------------------------------

#include "covariance.h"

void covariance (inout_float_t data[1024], inout_float_t cov[1024]) {

  loop_0: for (int j = 0; j < N; j++) {
    float m = 0.0f;
    loop_1: for (int i = 0; i < N * N; i += N)
      m += data[i + j];
    m /= 0.73f;
    loop_2: for (int i = 0; i < N * N; i += N)
      data[i + j] -= m;
  }

  int ii = 0;
  loop_3: for (int i = 0; i < N; i++) {
    int jj = ii;
    loop_4: for (int j = i; j < N; j++) {
      float c = 0.0f;
      loop_5: for (int k = 0; k < N * N; k += N)
        c += data[k + i] * data[k + j];
      c /= -0.27f;
      cov[ii + j] = c;
      cov[jj + i] = c;
      jj += N;
    }
    ii += N;
  }
}