#!/usr/bin/env Rscript

library(Seurat)
library(Matrix)
library(dplyr) 
require(stringr)
require(Signac)
library(EnsDb.Hsapiens.v86)

# Function to check if a file or directory exists
check_file_exists <- function(file_path) {
  if (!file.exists(file_path)) {
    stop(paste("The file or directory", file_path, "does not exist."))
  }
}

# Function to read RNA data manually from uncompressed files
read_rna_data <- function(rna_dir, dataset) {
  check_file_exists(rna_dir)
  
  # Define file paths for matrix, features, and barcodes
  matrix_path <- file.path(rna_dir, "matrix.mtx")
  features_path <- file.path(rna_dir, "features.tsv")
  barcodes_path <- file.path(rna_dir, "barcodes.tsv")
  
  # Read features (genes) and barcodes (cells)
  rna_features <- read.table(features_path, header = FALSE, sep = "\t")
  rna_barcodes <- read.table(barcodes_path, header = FALSE, sep = "\t")
  
  # Read the matrix file using Matrix::readMM
  rna_matrix <- readMM(matrix_path)
  row.names(rna_matrix) = rna_features$V1
  colnames(rna_matrix) = rna_barcodes$V1
  
  # Create a Seurat object using the matrix, and assign feature and barcode names
  rna <- CreateSeuratObject(counts = rna_matrix, project = dataset, min.cells = 1, min.features = 1)
  
  DefaultAssay(rna) <- "RNA"
  rna <- NormalizeData(rna)
  rna <- FindVariableFeatures(rna, nfeatures = 3000)
  rna <- ScaleData(rna)
  rna <- RunPCA(rna)
  rna <- FindNeighbors(rna, dims = 1:20, reduction = "pca")
  rna <- FindClusters(rna, resolution = 0.5)
  rna <- RunUMAP(rna, reduction = "pca", dims = 1:20)
  
  return(rna)
}

# Function to read ATAC data manually from uncompressed files
read_atac_data <- function(atac_dir) {
  check_file_exists(atac_dir)
  
  # Define file paths for matrix, features, and barcodes
  matrix_path <- file.path(atac_dir, "matrix.mtx")
  features_path <- file.path(atac_dir, "features.tsv")
  barcodes_path <- file.path(atac_dir, "barcodes.tsv")
  
  # Read features (peaks) and barcodes (cells)
  atac_features <- read.table(features_path, header = FALSE, sep = "\t")
  atac_barcodes <- read.table(barcodes_path, header = FALSE, sep = "\t")
  
  # Read the matrix file using Matrix::readMM
  atac_matrix <- readMM(matrix_path)
  rownames(atac_matrix) = atac_features$V1
  colnames(atac_matrix) = atac_barcodes$V1
  
  atac <- CreateChromatinAssay(
    counts = atac_matrix,
    sep = c("-", "-")
  )
  
  return(atac)
}

# Function to read metadata from a CSV file
read_metadata <- function(metadata_file) {
  check_file_exists(metadata_file)
  
  # Read metadata into a dataframe
  metadata <- read.csv(metadata_file, row.names = 1)
  return(metadata)
}

# Function to read multiomics data and merge RNA, ATAC data with metadata
read_multiomics_data <- function(rna_dir, atac_dir, metadata_file, dataset) {
  # Ensure all paths exist
  check_file_exists(rna_dir)
  check_file_exists(atac_dir)
  check_file_exists(metadata_file)
  
  # Read RNA and ATAC data
  rna <- read_rna_data(rna_dir, dataset)
  atac <- read_atac_data(atac_dir)
  
  # Read metadata
  metadata <- read_metadata(metadata_file)
  
  # Add metadata to RNA and ATAC objects
  rna <- AddMetaData(object = rna, metadata = metadata)
  
  return(list(rna = rna, atac = atac))
}

# Main function to process multiomics data for a given dataset
main <- function() {
  args <- commandArgs(trailingOnly = TRUE)
  
  if (length(args) == 0) {
    stop("Please provide a dataset name.")
  }
  
  dataset <- args[1]
  path_to_rna_data <- paste0('/data2/zhouwg_data/project/Garfield_benchmark/datasets/MultiomicsBenchmark/Raw_data/RNA_ATAC/', dataset, '/RNA')
  path_to_atac_data <- paste0('/data2/zhouwg_data/project/Garfield_benchmark/datasets/MultiomicsBenchmark/Raw_data/RNA_ATAC/', dataset, '/ATAC')
  path_to_metadata <- paste0('/data2/zhouwg_data/project/Garfield_benchmark/datasets/MultiomicsBenchmark/Raw_data/RNA_ATAC/', dataset, '/meta_data.csv')
  res_dir <- paste0('/data2/zhouwg_data/project/Garfield_benchmark/results/Vertical/RNA_ATAC/', dataset, '/')
  
  result <- read_multiomics_data(path_to_rna_data, path_to_atac_data, path_to_metadata, dataset)
  pbmc <- result[[1]]
  chrom_assay <- result[[2]]
  
  pbmc[["ATAC"]] <- chrom_assay
  DefaultAssay(pbmc) <- "ATAC"
  pbmc <- RunTFIDF(pbmc)
  pbmc <- FindTopFeatures(pbmc, min.cutoff = 'q0')
  pbmc <- RunSVD(pbmc, n = 50)
  pbmc <- FindNeighbors(pbmc, dims = 1:20, reduction = "lsi")
  pbmc <- FindClusters(pbmc, resolution = 0.5)
  pbmc <- RunUMAP(pbmc, reduction = "lsi", dims = 1:20)
  
  pbmc <- FindMultiModalNeighbors(pbmc, reduction.list = list("pca", "lsi"), dims.list = list(1:20, 1:20))
  
  pbmc <- RunUMAP(pbmc, nn.name = "weighted.nn", reduction.name = "wnn.umap", reduction.key = "wnnUMAP_")
  pbmc <- FindClusters(pbmc, graph.name = "wsnn", algorithm = 3, verbose = FALSE)
  
  writeMM(pbmc@graphs$wsnn, paste0(res_dir, "Seurat_connectivities.mtx"))
  writeMM(pbmc@graphs$wknn, paste0(res_dir, "Seurat_distance.mtx"))
}

# Run the main function
main()