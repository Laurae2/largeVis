#' largeVis: high-quality visualizations for large, high-dimensionality datasets
#'
#' This is an implementation of the \code{largeVis} algorithm by Tang et al.
#'
#' \code{largeVis} estimates a low-dimensional embedding for high-dimensional data, where the distance between vertices
#' in the low-dimensional space is proportional to the distance between them in the high-dimensional space. The algorithm
#' works in 4 phases:
#'
#' \itemize{
#' \item  Estimate candidate nearest-neighbors for each vertex by building \code{n.trees} random projection trees.
#' \item  Estimate \code{K} nearest-neighbors for each vertex by visiting each vertex' 2d-degree neighbors (its neighbors' neighbors).
#' This is repeated \code{max.iter} times.  Note that the original paper suggested a \code{max.iter} of 1, however a larger number
#' may be appropriate for some datasets if the algorithm has trouble finding K neighbors for every vertex.
#' \item Estimate \eqn{p_{j|i}}, the conditional probability that each edge found in the previous step is actually to a
#' nearest neighbor of each of its nodes.
#' \item Using stochastic gradient descent, estimate an embedding for each vertex in the low-dimensional space.
#' }
#' @references Jian Tang, Jingzhou Liu, Ming Zhang, Qiaozhu Mei. \href{https://arxiv.org/abs/1602.00370}{Visualizing Large-scale and High-dimensional Data.}
#' @name largeVis-package
#' @docType package
#' @useDynLib largeVis
#' @importFrom Rcpp sourceCpp
"_PACKAGE"
