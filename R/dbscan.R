#' OPTICS
#'
#' An implementation of the OPTICS algorithm.
#'
#' @param data Input data, where examples are columns.
#' @param neighbors An adjacency matrix of the type produced by \code{\link{randomProjectionTreeSearch}}
#' @param edges A weighted graph of the type produced by \code{\link{buildEdgeMatrix}}.
#' @param eps See \code{\link[dbscan]{optics}}.
#' @param minPts See \code{\link[dbscan]{optics}}.
#' @param eps_cl See \code{\link[dbscan]{optics}}.
#' @param xi See \code{\link[dbscan]{optics}}.
#' @param verbose Vebosity level.
#'
#' @details This is a preliminary implementation of a variant of the OPTICS algorithm that attempts
#' to leverage the \code{largeVis} nearest-neighbor search.
#'
#' One of \code{neighbors} or \code{edges} must be specified. If \code{edges} is missing,
#' \code{data} must also be given. If \code{data} is given along with either \code{edges}
#' or \code{neighbors}, the algorithm will attempt a more thorough search.
#'
#' @note Support for dbscan and optics are preliminary, and not fully tested for
#' correctness.
#'
#' @note This is not the original OPTICS algorithm. In particular, the neighbor-search strategy in
#' OPTICS is not used, in favor of using a pre-calculated neighbor matrix produced incidentally by
#' `largeVis`.
#'
#' @return An \code{\link[dbscan]{optics}} object.
#'
#' @importFrom dbscan optics_cut opticsXi
optics <- function(data = NULL,
                   neighbors = NULL,
                   edges = NULL,
                   eps,
                   minPts = nrow(data) + 1,
                   eps_cl,
                   xi,
                   verbose = getOption("verbose", TRUE)) {
  if (! is.null(edges) && is.null(data))
    ret <- optics_e(edges = edges,
                    eps = as.double(eps), minPts = as.integer(minPts),
                    verbose = verbose)
  else if (! is.null(edges))
    ret <- optics_ed(edges = edges, data = data,
                     eps = as.double(eps), minPts = as.integer(minPts),
                     verbose = verbose)
  else
    ret <- optics_nd(neighbors = neighbors, data = data,
                     eps = as.double(eps), minPts = as.integer(minPts),
                     verbose = verbose)

  ret$minPts <- minPts
  ret$eps <- eps
  ret$eps_cl <- NA
  class(ret) <- "optics"

  if(!missing(eps_cl)) ret <-optics_cut(ret, eps_cl)
  if(!missing(xi)) ret <- opticsXi(ret, xi)

  ret
}

#' dbscan
#'
#' An implementation of the dbscan algorithm.
#'
#' @param data Input data, where examples are columns.
#' @param neighbors An adjacency matrix of the type produced by \code{\link{randomProjectionTreeSearch}}
#' @param edges A weighted graph of the type produced by \code{\link{buildEdgeMatrix}}.
#' @param eps See \code{\link[dbscan]{dbscan}}.
#' @param minPts Minimum size of a cluster.'
#' @param partition If \code{TRUE}, attempt to calculate an approximate silhouette so the object returned is also
#' of class \code{\link[cluster]{partition.object}}, for compatibility with the \code{cluster} package.
#' @param verbose Verbosity level.
#'
#' @details This is a preliminary implementation of the OPTICS algorithm that attempts
#' to leverage the \code{largeVis} nearest-neighbor search.
#'
#' One of \code{neighbors} or \code{edges} must be specified. If \code{edges} is missing,
#' \code{data} must also be given. If \code{data} is given along with either \code{edges}
#' or \code{neighbors}, the algorithm will attempt a more thorough search.
#'
#' @note Support for dbscan and optics are preliminary, and not fully tested for
#' correctness.
#'
#' @note This is not the original DBSCAN algorithm. In particular, the neighbor-search strategy in
#' DBSCAN is not used, in favor of using a pre-calculated neighbor matrix produced incidentally by
#' `largeVis`.
#'
#' @importFrom stats aggregate
#'
#' @return An \code{\link[dbscan]{dbscan}} object.
dbscan <- function(data = NULL,
                   neighbors = NULL,
                   edges = NULL,
                   eps,
                   minPts = nrow(data) + 1,
                   partition = !missing(edges),
                   verbose = getOption("verbose", TRUE)) {

  if (! is.null(edges) && is.null(data))
    ret <- dbscan_e(edges = edges,
                    eps = as.double(eps), minPts = as.integer(minPts),
                    verbose = verbose)
  else if (! is.null(edges))
    ret <- dbscan_ed(edges = edges, data = data,
                     eps = as.double(eps), minPts = as.integer(minPts),
                     verbose = verbose)
  else
    ret <- dbscan_nd(neighbors = neighbors, data = data,
                     eps = as.double(eps), minPts = as.integer(minPts),
                     verbose = verbose)

  ret <- structure(list(cluster = ret, eps = eps, minPts = minPts),
            class = c("dbscan_fast", "dbscan"))
  if (partition) {
    ret$call <- sys.call()
    sil <- silhouette.dbscan(ret$cluster, edges)
    avgs <- aggregate(sil[, 3], by = list(as.vector(sil[, 1])), FUN = "mean", na.rm = TRUE)
    ret$silinfo <- list(
      widths = sil,
      clus.avg.widths = avgs$x,
      avg.width = mean(sil[, 3], na.rm = TRUE)
    )
    ret$objective <- NA
    ret$diss <- NA
    class(ret) <- c("dbscan_fast", "dbscan", "partition")
  }
  ret
}

silhouette.dbscan <- function(clusters, edges) {
  sil <- cbind(clusters, matrix(0, nrow = length(clusters), ncol = 2))
  silhouetteDbscan(edges, sil)
  colnames(sil) <- c("cluster", "neighbor", "sil_width")
  sil[, 2] <- abs(sil[, 2])
  class(sil) <- "silhouette"
  sil
}

edgeMatrixToKNNS <- function(edges) {
  id = apply(edges,MARGIN = 1, FUN = function(x) which(x != 0))
  dist = apply(edges, MARGIN = 1, FUN = function(x) x[x != 0])
  for (i in 1:ncol(id)) {
    ord <- order(dist[,i])
    id[,i] <- id[,i][ord]
    dist[,i] <- dist[,i][ord]
  }
  k = nrow(id)
  list(dist = t(dist), id = t(id), k = k)
}

# The source code for function lof is based on code that bore this license:
#######################################################################
# dbscan - Density Based Clustering of Applications with Noise
#          and Related Algorithms
# Copyright (C) 2015 Michael Hahsler

# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.


#' @title Local Outlier Factor Score
#'
#' @description Calculate the Local Outlier Factor (LOF) score for each data point given knowledge
#' of k-Nearest Neighbors.
#'
#' @param edges An edge matrix of the type produced by \code{\link{buildEdgeMatrix}}.
#'
#' @references Based on code in the \code{\link[dbscan]{dbscan}} package.
#'
#' @return A vector of LOF values for each data point.
#' @export
lof <- function(edges) {
  kNNlist <- edgeMatrixToKNNS(edges)
  N <- nrow(kNNlist$id)
  K <- kNNlist$k

  # lrd <- rep(0, N)
  lrd <- rep(0, N)
  # for(i in 1:N) {
  #   input <- kNNlist$dist[c(i, kNNlist$id[i, ]) ,]
  #   lrd[i] <- 1 / (sum(apply(input, MARGIN = 1, max)) / K)
  # }
  for(i in 1:N) lrd[i] <- 1/(sum(apply(
  	cbind(kNNlist$dist[kNNlist$id[i,], K], kNNlist$dist[i,]),
  	1, max)) / K)

  ret <- rep(0, N)
  for (i in 1:N) ret[i] <- sum(lrd[kNNlist$id[i,]])/K / lrd[i]

  ret[is.nan(ret)] <- NA

  ret
}
