#' Estimate LINE embeddings for a graph
#'
#'
#' @param wij A symmetric sparse matrix of edge weights, in C-compressed format, as created with the \code{Matrix} package.
#' @param dim The number of dimensions for the projection space; must be even.
#' @param sgd_batches The number of edges to process during SGD; defaults to 1 million.
#' @param M The number of negative edges to sample for each positive edge.
#' @param rho Initial learning rate.
#' @param coords An initialized coordinate matrix.
#' @param verbose Verbosity
#'
#' @return A dense [N,D] matrix of the LINE projection.
#' @export
#' @importFrom stats rnorm
#'
rLINE <- function(wij, # symmetric sparse matrix
                        dim = 1024, # dimension of the projection space
                        sgd_batches = 1e7,
                        M = 5,
                        rho = 1,
                        coords = NULL,
                        verbose = TRUE) {

  N <-  (length(wij@p) - 1)
  sources <- rep(0:(N - 1), diff(wij@p))
  if (any(is.na(sources))) stop("NAs in the index vector.")
  targets <- wij@i

  ##############################################
  # Initialize coordinate matrix
  ##############################################
  if (is.null(coords)) coords <- matrix(rnorm(N * dim), nrow = dim)

  #################################################
  # SGD
  #################################################
  if (verbose) cat("Estimating embeddings.\n")
  coords <- LINE(coords,
                targets = targets,
                sources = sources,
                ps = wij@p,
                ws = wij@x,
                M = M,
                rho = rho,
                nBatches = sgd_batches,
                verbose = verbose)

  return(coords)
}
