// [[Rcpp::plugins(openmp)]]
// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::depends(RcppProgress)]]
#include "largeVis.h"

using namespace Rcpp;
using namespace std;
using namespace arma;


void Update(double *source,
            double *targetorcontext,
            double *vec_error,
            double rho,
            int label,
            int D) {
  double x = 0, g;
  for (int c = 0; c != D; c++) x += source[c] * targetorcontext[c];
  if ( x > 6) g = 1;
  else if (x < -6) g = 0;
  else g = 1 / (1 + exp(-x));
  g = (label - g) * rho;
  for (int c = 0; c != D; c++) vec_error[c] += g * targetorcontext[c];
  for (int c = 0; c != D; c++) targetorcontext[c] += g * source[c];
}

// [[Rcpp::export]]
arma::mat LINE(arma::mat coords,
              arma::ivec& targets, // target edge
              const IntegerVector sources, // source edge
              const IntegerVector ps, // N+1 length vector of indices to start of each row j in vector is
              const arma::vec ws, // w{ij}
              const double rho,
              const long nBatches,
              const int M,
              bool verbose) {

  Progress progress(nBatches, verbose);

  const int D = coords.n_rows;
  if (D > 1024) stop("Low dimensional space cannot have more than 10 dimensions.");
  const int N = ps.size() - 1;
  const int E = ws.size();
  double *coordsPtr = coords.memptr();

  double* negProb = new double[N];
  int* negAlias = new int[N];
  makeAliasTable(N, pow(diff(ps), 0.75), negProb, negAlias);
  double* posProb = new double[E];
  int* posAlias = new int[E];
  makeAliasTable(E, ws, posProb, posAlias);

  const int posSampleLength = ((nBatches > 1000000) ? 1000000 : (int) nBatches);
  mat positiveSamples = randu<mat>(2, posSampleLength);
  double *posRandomPtr = positiveSamples.memptr();

#ifdef _OPENMP
#pragma omp parallel for schedule(static) shared (coords, positiveSamples)
#endif
  for (long eIdx=0; eIdx < nBatches; eIdx++) if (progress.increment()) {

    const int e_ij = searchAliasTable(E, posRandomPtr,
                                      posProb, posAlias,
                                      eIdx % posSampleLength);

    const int source_id = sources[e_ij];
    const int target_id = targets[e_ij];

    // mix weight into learning rate
    const double localRho =  rho - (rho * eIdx / nBatches);

    double *y_start = coordsPtr + (source_id * D);
    double *y_target = coordsPtr + (target_id * D);

    double vec_error[2048];
    for (int d = 0; d < D; d++) vec_error[d] = 0;

    Update(y_start, y_target, vec_error, localRho, 1, D / 2);
    Update(y_start, y_target + (D / 2), vec_error, localRho, 1, D / 2);

    mat negSamples = mat(2, M * 2);
    double *samplesPtr = negSamples.memptr();
    int sampleIdx = 0;
    ivec searchRange = targets.subvec(ps[source_id], ps[source_id + 1] - 1);
    ivec::iterator searchBegin = searchRange.begin();
    ivec::iterator searchEnd = searchRange.end();
    int m = 0;
    int k;
    while (m < M) {
      if (sampleIdx % (M * 2) == 0) negSamples.randu();
      k = searchAliasTable(N, samplesPtr, negProb, negAlias, sampleIdx++ % (M * 2));
      // Check that the draw isn't one of i's edges
      if (k == source_id ||
          k == target_id ||
          binary_search( searchBegin,
                         searchEnd,
                         k)) continue;

      double *y_k = coordsPtr + (k * D);

      Update(y_start, y_k, vec_error, localRho, 0, D / 2);
      Update(y_start, y_k + (D / 2), vec_error, localRho, 0, D / 2);

      m++;
      if (sampleIdx > M * 10) stop("Bad sampleidx");
    }
    for (int d = 0; d < D; d++) y_start[d] += vec_error[d] * localRho;

    if (eIdx > 0 &&
        eIdx % posSampleLength == 0) positiveSamples.randu();
  }
  return normalise(coords, 2, 0);
};
