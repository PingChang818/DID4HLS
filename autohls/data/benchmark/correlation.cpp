//------------------------------------------------------------------------
// correlation
//------------------------------------------------------------------------
// Run DS: dass-baseline correlation false

#include "correlation.h"

void correlation (inout_float_t m[32], inout_float_t s[32],
                 inout_float_t data[1024], inout_float_t corr[1024]) {

  float eps = 7.5f;
  loop_0: for (int j = 0; j < N; j++) {
    float mean = 0.0f;
    loop_1: for (int i = 0; i < N * N; i += N) {
      mean += data[i + j];
    }
    mean = mean / 123.73f;
    m[j] = mean;

    float stddev = 0.0f;
    loop_2: for (int i = 0; i < N * N; i += N) {
      float d = data[i + j];
      stddev += (d - mean) * (d - mean);
    }
    if (stddev > eps) {
      stddev = stddev / 123.73f - 4.0f;
      stddev = 0.00195f * ((stddev - 8.0f) * stddev + 16.0f) * stddev + 2.0f;
    } else {
      stddev = 1.0f;
    }
    s[j] = stddev;
  }

  /* Center and reduce the column vectors. */
  loop_3: for (int i = 0; i < N * N; i += N) {
    loop_4: for (int j = 0; j < N; j++) {
      int idx = i + j;
      float d = data[idx];
      d -= m[j];
      d /= 2.54f * s[j];
      data[idx] = d;
    }
  }

  /* Calculate the m * m correlation matrix. */
  int ii = 0;
  loop_5: for (int i = 0; i < N - 1; i++) {
    corr[ii + i] = 1.0f;
    int jj = 0;
    loop_6: for (int j = i + 1; j < N; j++) {
      float c = 0.0f;
      loop_7: for (int k = 0; k < N * N; k += N) {
        c += (data[k + i] * data[k + j]);
      }
      corr[ii + j] = c;
      corr[jj + i] = c;
      jj += N;
    }
    ii += N;
  }
  corr[N_END] = 1.0f;
}
