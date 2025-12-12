#!/usr/bin/env Rscript
# Unit tests for compute_mims_r.R
# Run with: Rscript tests/test_compute_mims_r.R or testthat::test_file("tests/test_compute_mims_r.R")

suppressPackageStartupMessages({
  library(testthat)
  library(readr)
  library(dplyr)
  library(here)
})

# Set project root
here::set_here("/n/home01/egraff/sample_imputation")

# Source the main script to load functions
source(here("scripts", "compute_mims_r.R"))

# Helper function to get test data directory
get_test_data_dir <- function() {
  here("tests", "data")
}

# ==============================================================================
# Test Suite: Data Loading
# ==============================================================================

test_that("read_hourly_file reads valid CSV file", {
  valid_file <- file.path(get_test_data_dir(), "synthetic_acc_valid.csv")
  df <- read_hourly_file(valid_file)
  
  expect_false(is.null(df))
  expect_true(is.data.frame(df))
  expect_true(all(c("timestamp", "x", "y", "z") %in% colnames(df)))
  expect_true(nrow(df) > 0)
  expect_true(is.numeric(df$x))
  expect_true(is.numeric(df$y))
  expect_true(is.numeric(df$z))
})

test_that("read_hourly_file handles empty CSV file", {
  empty_file <- file.path(get_test_data_dir(), "synthetic_acc_empty.csv")
  
  # Should return NULL and produce warning
  expect_warning(df <- read_hourly_file(empty_file), "Empty file")
  expect_null(df)
})

test_that("read_hourly_file handles missing columns", {
  missing_cols_file <- file.path(get_test_data_dir(), "synthetic_acc_missing_cols.csv")
  
  # Should return NULL and produce warning about missing columns
  expect_warning(df <- read_hourly_file(missing_cols_file), "Missing columns")
  expect_null(df)
})

test_that("read_hourly_file handles non-existent file", {
  non_existent_file <- file.path(get_test_data_dir(), "does_not_exist.csv")
  
  # Should return NULL and produce warning
  expect_warning(df <- read_hourly_file(non_existent_file), "Error reading")
  expect_null(df)
})

test_that("read_hourly_file extracts only required columns", {
  valid_file <- file.path(get_test_data_dir(), "synthetic_acc_valid.csv")
  df <- read_hourly_file(valid_file)
  
  expect_false(is.null(df))
  expect_equal(ncol(df), 4)  # timestamp, x, y, z
  expect_equal(colnames(df), c("timestamp", "x", "y", "z"))
})

# ==============================================================================
# Test Suite: Timestamp Conversion
# ==============================================================================

test_that("convert_timestamp converts milliseconds to POSIXct", {
  # Test with known timestamp
  timestamp_ms <- 1647870345465  # 2022-03-21 13:45:45.465 UTC
  result <- convert_timestamp(timestamp_ms)
  
  expect_true(inherits(result, "POSIXct"))
  expect_equal(attr(result, "tzone"), "UTC")
  
  # Check the conversion is correct
  expected_time <- as.POSIXct("2022-03-21 13:45:45.465", tz = "UTC")
  expect_equal(as.numeric(result), as.numeric(expected_time))
})

test_that("convert_timestamp handles multiple timestamps", {
  timestamp_ms <- c(1647870345465, 1647870345564, 1647870345664)
  result <- convert_timestamp(timestamp_ms)
  
  expect_equal(length(result), 3)
  expect_true(all(inherits(result, "POSIXct")))
  
  # Check ordering is preserved
  expect_true(all(diff(result) > 0))
})

test_that("convert_timestamp handles edge case timestamps", {
  # Epoch timestamp
  epoch_ms <- 0
  result <- convert_timestamp(epoch_ms)
  expect_equal(as.character(result), "1970-01-01 00:00:00")
  
  # Far future timestamp
  future_ms <- 2000000000000  # Year 2033
  result <- convert_timestamp(future_ms)
  expect_true(inherits(result, "POSIXct"))
})

# ==============================================================================
# Test Suite: Duplicate Timestamp Handling
# ==============================================================================

test_that("prepare_for_mimsunit deduplicates timestamps", {
  # Create data with duplicate timestamps
  df <- data.frame(
    timestamp = as.POSIXct(c("2022-01-01 00:00:00", "2022-01-01 00:00:01", 
                              "2022-01-01 00:00:01", "2022-01-01 00:00:02"), tz = "UTC"),
    x = c(0.1, 0.2, 0.3, 0.4),
    y = c(0.1, 0.2, 0.3, 0.4),
    z = c(0.1, 0.2, 0.3, 0.4)
  )
  
  result <- prepare_for_mimsunit(df)
  
  # Should have only 3 rows (one duplicate removed)
  expect_equal(nrow(result), 3)
  
  # Should keep first occurrence of duplicate
  expect_equal(result$X[2], 0.2)
})

test_that("prepare_for_mimsunit sorts by timestamp", {
  # Create unsorted data
  df <- data.frame(
    timestamp = as.POSIXct(c("2022-01-01 00:00:02", "2022-01-01 00:00:00", 
                              "2022-01-01 00:00:01"), tz = "UTC"),
    x = c(0.3, 0.1, 0.2),
    y = c(0.3, 0.1, 0.2),
    z = c(0.3, 0.1, 0.2)
  )
  
  result <- prepare_for_mimsunit(df)
  
  # Should be sorted
  expect_true(all(diff(result$HEADER_TIME_STAMP) > 0))
  expect_equal(result$X[1], 0.1)  # First value after sorting
})

test_that("prepare_for_mimsunit renames columns correctly", {
  df <- data.frame(
    timestamp = as.POSIXct(c("2022-01-01 00:00:00", "2022-01-01 00:00:01"), tz = "UTC"),
    x = c(0.1, 0.2),
    y = c(0.1, 0.2),
    z = c(0.1, 0.2)
  )
  
  result <- prepare_for_mimsunit(df)
  
  expect_equal(colnames(result), c("HEADER_TIME_STAMP", "X", "Y", "Z"))
})

# ==============================================================================
# Test Suite: Dynamic Range Computation
# ==============================================================================

test_that("compute_dynamic_range computes correct range", {
  df1 <- data.frame(x = c(0.1, 0.2), y = c(-0.5, 0.3), z = c(0.0, 0.1))
  df2 <- data.frame(x = c(-0.3, 0.5), y = c(0.2, 0.4), z = c(-0.1, 0.2))
  
  result <- compute_dynamic_range(list(df1, df2))
  
  expect_equal(length(result), 2)
  expect_equal(result[1], -0.5)  # min value
  expect_equal(result[2], 0.5)   # max value
})

test_that("compute_dynamic_range handles all zeros", {
  df <- data.frame(x = c(0, 0, 0), y = c(0, 0, 0), z = c(0, 0, 0))
  
  # Should return default range with warning
  expect_warning(result <- compute_dynamic_range(list(df)), "Very small dynamic range")
  expect_equal(result, c(-3, 3))
})

test_that("compute_dynamic_range handles extreme outliers", {
  df <- data.frame(x = c(-100, 100), y = c(-50, 50), z = c(-75, 75))
  
  result <- compute_dynamic_range(list(df))
  
  expect_equal(result[1], -100)
  expect_equal(result[2], 100)
})

test_that("compute_dynamic_range handles missing values", {
  df <- data.frame(x = c(0.1, NA, 0.2), y = c(NA, 0.3, 0.4), z = c(0.5, 0.6, NA))
  
  result <- compute_dynamic_range(list(df))
  
  # Should ignore NA values
  expect_equal(length(result), 2)
  expect_false(anyNA(result))
  expect_true(is.finite(result[1]))
  expect_true(is.finite(result[2]))
})

test_that("compute_dynamic_range fails with empty data", {
  # Empty list
  expect_error(compute_dynamic_range(list()), "No valid acceleration values")
  
  # List with NULL
  expect_error(compute_dynamic_range(list(NULL)), "No valid acceleration values")
  
  # List with empty data frame
  df_empty <- data.frame(x = numeric(0), y = numeric(0), z = numeric(0))
  expect_error(compute_dynamic_range(list(df_empty)), "No valid acceleration values")
})

test_that("compute_dynamic_range ignores NULL elements", {
  df1 <- data.frame(x = c(0.1, 0.2), y = c(0.3, 0.4), z = c(0.5, 0.6))
  df2 <- NULL
  df3 <- data.frame(x = c(-0.1, -0.2), y = c(-0.3, -0.4), z = c(-0.5, -0.6))
  
  result <- compute_dynamic_range(list(df1, df2, df3))
  
  expect_equal(result[1], -0.6)
  expect_equal(result[2], 0.6)
})

# ==============================================================================
# Test Suite: Output Formatting
# ==============================================================================

test_that("format_timestamp_iso formats timestamps correctly", {
  timestamp <- as.POSIXct("2022-03-21 13:46:00", tz = "UTC")
  result <- format_timestamp_iso(timestamp)
  
  # Should be in format: "2022-03-21T13:46:00.000000+0000"
  expect_match(result, "^\\d{4}-\\d{2}-\\d{2}T\\d{2}:\\d{2}:\\d{2}\\.\\d{6}\\+0000$")
  expect_true(grepl("2022-03-21T13:46:00", result))
})

test_that("format_timestamp_iso handles multiple timestamps", {
  timestamps <- as.POSIXct(c("2022-03-21 13:46:00", "2022-03-21 13:47:00"), tz = "UTC")
  result <- format_timestamp_iso(timestamps)
  
  expect_equal(length(result), 2)
  expect_true(all(grepl("^\\d{4}-\\d{2}-\\d{2}T\\d{2}:\\d{2}:\\d{2}\\.\\d{6}\\+0000$", result)))
})

test_that("format_timestamp_iso preserves timezone as UTC", {
  timestamp <- as.POSIXct("2022-03-21 13:46:00", tz = "UTC")
  result <- format_timestamp_iso(timestamp)
  
  # Should end with +0000 indicating UTC
  expect_true(grepl("\\+0000$", result))
})

test_that("format_timestamp_iso includes microseconds", {
  timestamp <- as.POSIXct("2022-03-21 13:46:00.123456", tz = "UTC")
  result <- format_timestamp_iso(timestamp)
  
  # Should have 6 digits for microseconds
  expect_match(result, "\\.\\d{6}\\+0000$")
})

# ==============================================================================
# Test Suite: Helper Functions
# ==============================================================================

test_that("list_participants excludes __MACOSX", {
  # Create temporary directory structure
  temp_dir <- tempdir()
  dir.create(file.path(temp_dir, "test_raw"), showWarnings = FALSE)
  dir.create(file.path(temp_dir, "test_raw", "participant1"), showWarnings = FALSE)
  dir.create(file.path(temp_dir, "test_raw", "participant2"), showWarnings = FALSE)
  dir.create(file.path(temp_dir, "test_raw", "__MACOSX"), showWarnings = FALSE)
  
  participants <- list_participants(file.path(temp_dir, "test_raw"))
  
  expect_false("__MACOSX" %in% participants)
  expect_true(all(c("participant1", "participant2") %in% participants))
  
  # Cleanup
  unlink(file.path(temp_dir, "test_raw"), recursive = TRUE)
})

test_that("get_hourly_files returns sorted files", {
  # Create temporary directory structure
  temp_dir <- tempdir()
  accel_dir <- file.path(temp_dir, "test_participant", "accelerometer")
  dir.create(accel_dir, recursive = TRUE, showWarnings = FALSE)
  
  # Create test files
  file.create(file.path(accel_dir, "2022-03-21 13_00_00+00_00.csv"))
  file.create(file.path(accel_dir, "2022-03-21 14_00_00+00_00.csv"))
  file.create(file.path(accel_dir, "2022-03-21 12_00_00+00_00.csv"))
  
  files <- get_hourly_files("test_participant", temp_dir)
  
  expect_equal(length(files), 3)
  # Check sorted
  basenames <- basename(files)
  expect_equal(basenames[1], "2022-03-21 12_00_00+00_00.csv")
  expect_equal(basenames[3], "2022-03-21 14_00_00+00_00.csv")
  
  # Cleanup
  unlink(file.path(temp_dir, "test_participant"), recursive = TRUE)
})

test_that("get_hourly_files warns on missing directory", {
  temp_dir <- tempdir()
  
  expect_warning(files <- get_hourly_files("nonexistent_participant", temp_dir), 
                 "Accelerometer directory not found")
  expect_equal(length(files), 0)
})

test_that("get_hourly_files warns on empty directory", {
  # Create temporary directory structure
  temp_dir <- tempdir()
  accel_dir <- file.path(temp_dir, "empty_participant", "accelerometer")
  dir.create(accel_dir, recursive = TRUE, showWarnings = FALSE)
  
  expect_warning(files <- get_hourly_files("empty_participant", temp_dir), 
                 "No CSV files found")
  expect_equal(length(files), 0)
  
  # Cleanup
  unlink(file.path(temp_dir, "empty_participant"), recursive = TRUE)
})

# ==============================================================================
# Test Suite: Edge Cases
# ==============================================================================

test_that("read_hourly_file handles single row", {
  single_row_file <- file.path(get_test_data_dir(), "synthetic_acc_single_row.csv")
  
  # Check if file exists; if not, create it
  if (!file.exists(single_row_file)) {
    writeLines(
      c("timestamp,UTC time,accuracy,x,y,z",
        "1647870345465,2022-03-21T13:45:45.465,unknown,0.5,0.5,0.5"),
      single_row_file
    )
  }
  
  df <- read_hourly_file(single_row_file)
  
  expect_false(is.null(df))
  expect_equal(nrow(df), 1)
})

test_that("convert_timestamp handles vector of length 0", {
  result <- convert_timestamp(numeric(0))
  expect_equal(length(result), 0)
  expect_true(inherits(result, "POSIXct"))
})

test_that("prepare_for_mimsunit handles single row", {
  df <- data.frame(
    timestamp = as.POSIXct("2022-01-01 00:00:00", tz = "UTC"),
    x = 0.1,
    y = 0.2,
    z = 0.3
  )
  
  result <- prepare_for_mimsunit(df)
  
  expect_equal(nrow(result), 1)
  expect_equal(colnames(result), c("HEADER_TIME_STAMP", "X", "Y", "Z"))
})

test_that("dynamic range handles negative and positive values", {
  df <- data.frame(x = c(-2.5, 3.5), y = c(-1.5, 2.0), z = c(-3.0, 4.0))
  
  result <- compute_dynamic_range(list(df))
  
  expect_equal(result[1], -3.0)
  expect_equal(result[2], 4.0)
})

# ==============================================================================
# Run Tests
# ==============================================================================

cat("\n")
cat(paste(rep("=", 70), collapse = ""), "\n")
cat("Running Unit Tests for compute_mims_r.R\n")
cat(paste(rep("=", 70), collapse = ""), "\n\n")

test_results <- test_dir(here("tests"), filter = "test_compute_mims_r", reporter = "progress")

cat("\n")
cat(paste(rep("=", 70), collapse = ""), "\n")
cat("Test Summary\n")
cat(paste(rep("=", 70), collapse = ""), "\n")

# Print summary
if (inherits(test_results, "testthat_results")) {
  n_tests <- length(test_results)
  n_failed <- sum(sapply(test_results, function(x) inherits(x, "expectation_failure")))
  n_passed <- n_tests - n_failed
  
  cat(sprintf("Total tests: %d\n", n_tests))
  cat(sprintf("Passed: %d\n", n_passed))
  cat(sprintf("Failed: %d\n", n_failed))
  
  if (n_failed > 0) {
    cat("\nTests FAILED!\n")
    quit(status = 1)
  } else {
    cat("\nAll tests PASSED!\n")
  }
} else {
  cat("Tests completed.\n")
}
