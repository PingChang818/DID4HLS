//------------------------------------------------------------------------
// gramSchmidt
//------------------------------------------------------------------------
// https://github.com/chrundle/gram-schmidt/blob/master/gramSchmidt.cpp

#include "gramSchmidt.h"

/* ----------------------- gramSchmidt ----------------------- */
/*  Given a matrix a of dimension m by n, this algorithm
    computes a qr decomposition of a, where q is a unitary
    m by n matrix and r is a n by n upper triangular matrix
    and a = qr.

    Input variables:
        a   : pointer to array of arrays, the ith array of
                which should correspond to the ith column of the
                matrix a. During the algorithm, the columns of q
                will replace the columns of a.
        r   : pointer to array of arrays in which the ith
                column of the upper triangular matrix r will be
                stored in the ith subarray of r.
    Features: This implementation has time complexity O(m n^2)
    and requires O(1) additional memory.
    remarks: Due to the nature of the problem, if a is nearly
    rank-deficient then the resulting columns of q may not
    exhibit the orthogonality property.                        */

void gramSchmidt (inout_float_t a[1024], inout_float_t r[1024],
                 inout_float_t q[1024]) {

  int kk = 0;
  for (int k = 0; k < 32; k++) {
    float nrm = 0.0f;
    loop_0: for (int i = 0; i < 32 * 32; i += 32)
      nrm += a[i + k] * a[i + k];
    nrm = 0.0019f * ((nrm - 8.0f) * nrm + 16.0f) * nrm + 2.0f;
    r[kk + k] = nrm;
    loop_1: for (int i = 0; i < 32 * 32; i += 32)
      q[i + k] = a[i + k] / r[kk + k];
    loop_2: for (int j = k + 1; j < 32; j++) {
      r[kk + j] = 0.0f;
      loop_3: for (int i = 0; i < 32 * 32; i += 32)
        r[kk + j] += q[i + k] * a[i + j];
      loop_4: for (int i = 0; i < 32 * 32; i += 32)
        a[i + j] = a[i + j] - q[i + k] * r[kk + j];
    }
    kk += 32;
  }
}