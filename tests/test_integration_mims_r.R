#!/usr/bin/env Rscript
# Integration tests for compute_mims_r.R
# Tests end-to-end workflow with synthetic participant data

suppressPackageStartupMessages({
  library(testthat)
  library(readr)
  library(dplyr)
  library(here)
  library(MIMSunit)
})

# Set project root
here::set_here("/n/home01/egraff/sample_imputation")

# Source the main script to load functions
source(here("scripts", "compute_mims_r.R"))

# ==============================================================================
# Setup Test Environment
# ==============================================================================

# Create temporary test directory structure
setup_test_environment <- function() {
  temp_base <- file.path(tempdir(), "mims_integration_test")
  
  # Clean up any existing test directory
  if (dir.exists(temp_base)) {
    unlink(temp_base, recursive = TRUE)
  }
  
  dir.create(temp_base, showWarnings = FALSE)
  
  raw_dir <- file.path(temp_base, "raw")
  interim_dir <- file.path(temp_base, "interim")
  
  dir.create(raw_dir, showWarnings = FALSE)
  dir.create(interim_dir, showWarnings = FALSE)
  
  return(list(
    base = temp_base,
    raw = raw_dir,
    interim = interim_dir
  ))
}

# Create synthetic participant data
create_synthetic_participant <- function(participant_id, raw_dir, n_hours = 2, samples_per_hour = 600) {
  accel_dir <- file.path(raw_dir, participant_id, "accelerometer")
  dir.create(accel_dir, recursive = TRUE, showWarnings = FALSE)
  
  base_timestamp <- as.POSIXct("2022-03-21 13:00:00", tz = "UTC")
  
  for (hour in 0:(n_hours - 1)) {
    hour_timestamp <- base_timestamp + (hour * 3600)
    filename <- format(hour_timestamp, "%Y-%m-%d %H_00_00+00_00.csv", tz = "UTC")
    filepath <- file.path(accel_dir, filename)
    
    # Generate synthetic data
    start_ms <- as.numeric(hour_timestamp) * 1000
    timestamps <- seq(start_ms, start_ms + (samples_per_hour - 1) * 100, by = 100)
    
    # Generate sinusoidal acceleration values
    t <- seq(0, 1, length.out = samples_per_hour)
    x_vals <- 0.5 * sin(2 * pi * t)
    y_vals <- 0.3 * cos(2 * pi * t)
    z_vals <- 0.4 * sin(2 * pi * t + pi/4)
    
    df <- data.frame(
      timestamp = as.integer(timestamps),
      `UTC time` = format(as.POSIXct(timestamps/1000, origin = "1970-01-01", tz = "UTC"), 
                          "%Y-%m-%dT%H:%M:%OS3", tz = "UTC"),
      accuracy = "unknown",
      x = x_vals,
      y = y_vals,
      z = z_vals,
      check.names = FALSE
    )
    
    write_csv(df, filepath)
  }
  
  return(TRUE)
}

# Cleanup test environment
cleanup_test_environment <- function(test_env) {
  if (dir.exists(test_env$base)) {
    unlink(test_env$base, recursive = TRUE)
  }
}

# ==============================================================================
# Integration Tests
# ==============================================================================

test_that("End-to-end: process single participant with multiple hourly files", {
  test_env <- setup_test_environment()
  
  # Create synthetic participant
  participant_id <- "test_participant_001"
  create_synthetic_participant(participant_id, test_env$raw, n_hours = 3)
  
  # Process the participant
  success <- process_participant(participant_id, test_env$raw, test_env$interim)
  
  expect_true(success)
  
  # Check output file exists
  output_file <- file.path(test_env$interim, paste0(participant_id, "_R_mims.csv"))
  expect_true(file.exists(output_file))
  
  # Read and validate output
  output_df <- read_csv(output_file, show_col_types = FALSE)
  
  expect_equal(ncol(output_df), 2)
  expect_equal(colnames(output_df), c("timestamp", "mims"))
  expect_true(nrow(output_df) > 0)
  
  # Check timestamp format
  expect_true(all(grepl("^\\d{4}-\\d{2}-\\d{2}T\\d{2}:\\d{2}:\\d{2}\\.\\d{6}\\+0000$", 
                        output_df$timestamp)))
  
  # Check MIMS values are numeric
  expect_true(is.numeric(output_df$mims))
  
  # Cleanup
  cleanup_test_environment(test_env)
})

test_that("End-to-end: process multiple participants", {
  test_env <- setup_test_environment()
  
  # Create multiple synthetic participants
  participants <- c("test_participant_001", "test_participant_002", "test_participant_003")
  for (pid in participants) {
    create_synthetic_participant(pid, test_env$raw, n_hours = 2)
  }
  
  # Process all participants
  results <- list()
  for (pid in participants) {
    results[[pid]] <- process_participant(pid, test_env$raw, test_env$interim)
  }
  
  # Check all succeeded
  expect_true(all(unlist(results)))
  
  # Check all output files exist
  for (pid in participants) {
    output_file <- file.path(test_env$interim, paste0(pid, "_R_mims.csv"))
    expect_true(file.exists(output_file))
  }
  
  # Cleanup
  cleanup_test_environment(test_env)
})

test_that("End-to-end: handle participant with gaps in hourly files", {
  test_env <- setup_test_environment()
  
  participant_id <- "test_participant_gaps"
  accel_dir <- file.path(test_env$raw, participant_id, "accelerometer")
  dir.create(accel_dir, recursive = TRUE, showWarnings = FALSE)
  
  # Create files with gaps (hour 0, skip hour 1, hour 2)
  base_timestamp <- as.POSIXct("2022-03-21 13:00:00", tz = "UTC")
  
  for (hour in c(0, 2, 3)) {  # Gap at hour 1
    hour_timestamp <- base_timestamp + (hour * 3600)
    filename <- format(hour_timestamp, "%Y-%m-%d %H_00_00+00_00.csv", tz = "UTC")
    filepath <- file.path(accel_dir, filename)
    
    # Generate simple data
    start_ms <- as.numeric(hour_timestamp) * 1000
    timestamps <- seq(start_ms, start_ms + 599 * 100, by = 100)
    
    df <- data.frame(
      timestamp = as.integer(timestamps),
      `UTC time` = format(as.POSIXct(timestamps/1000, origin = "1970-01-01", tz = "UTC"), 
                          "%Y-%m-%dT%H:%M:%OS3", tz = "UTC"),
      accuracy = "unknown",
      x = rep(0.1, 600),
      y = rep(0.2, 600),
      z = rep(0.3, 600),
      check.names = FALSE
    )
    
    write_csv(df, filepath)
  }
  
  # Process the participant
  success <- process_participant(participant_id, test_env$raw, test_env$interim)
  
  # Should still succeed despite gaps
  expect_true(success)
  
  # Check output exists
  output_file <- file.path(test_env$interim, paste0(participant_id, "_R_mims.csv"))
  expect_true(file.exists(output_file))
  
  # Cleanup
  cleanup_test_environment(test_env)
})

test_that("End-to-end: handle participant with some invalid files", {
  test_env <- setup_test_environment()
  
  participant_id <- "test_participant_mixed"
  accel_dir <- file.path(test_env$raw, participant_id, "accelerometer")
  dir.create(accel_dir, recursive = TRUE, showWarnings = FALSE)
  
  # Create one valid file
  create_synthetic_participant(participant_id, test_env$raw, n_hours = 1)
  
  # Create an empty file
  empty_file <- file.path(accel_dir, "2022-03-21 14_00_00+00_00.csv")
  writeLines("timestamp,UTC time,accuracy,x,y,z", empty_file)
  
  # Create a file with missing columns
  invalid_file <- file.path(accel_dir, "2022-03-21 15_00_00+00_00.csv")
  writeLines(c("timestamp,UTC time,accuracy,x,y", "1647878400000,2022-03-21T15:00:00.000,unknown,0.1,0.2"), 
             invalid_file)
  
  # Process the participant (should succeed with warnings)
  expect_warning(success <- process_participant(participant_id, test_env$raw, test_env$interim))
  
  # Should still succeed with at least one valid file
  expect_true(success)
  
  # Cleanup
  cleanup_test_environment(test_env)
})

test_that("End-to-end: fail gracefully with no valid data", {
  test_env <- setup_test_environment()
  
  participant_id <- "test_participant_no_data"
  accel_dir <- file.path(test_env$raw, participant_id, "accelerometer")
  dir.create(accel_dir, recursive = TRUE, showWarnings = FALSE)
  
  # Create only empty files
  for (hour in 0:1) {
    filename <- sprintf("2022-03-21 %02d_00_00+00_00.csv", 13 + hour)
    filepath <- file.path(accel_dir, filename)
    writeLines("timestamp,UTC time,accuracy,x,y,z", filepath)
  }
  
  # Process the participant (should fail gracefully)
  expect_warning(success <- process_participant(participant_id, test_env$raw, test_env$interim))
  expect_false(success)
  
  # Output file should not exist
  output_file <- file.path(test_env$interim, paste0(participant_id, "_R_mims.csv"))
  expect_false(file.exists(output_file))
  
  # Cleanup
  cleanup_test_environment(test_env)
})

test_that("End-to-end: compute MIMS with expected output structure", {
  test_env <- setup_test_environment()
  
  participant_id <- "test_participant_structure"
  create_synthetic_participant(participant_id, test_env$raw, n_hours = 2, samples_per_hour = 6000)
  
  # Process the participant
  success <- process_participant(participant_id, test_env$raw, test_env$interim)
  expect_true(success)
  
  # Read output
  output_file <- file.path(test_env$interim, paste0(participant_id, "_R_mims.csv"))
  output_df <- read_csv(output_file, show_col_types = FALSE)
  
  # Should have approximately 2 hours * 60 minutes = 120 epochs (give or take)
  expect_true(nrow(output_df) >= 110 && nrow(output_df) <= 130)
  
  # Parse timestamps and check they're sorted
  timestamps <- as.POSIXct(output_df$timestamp, format = "%Y-%m-%dT%H:%M:%S", tz = "UTC")
  expect_true(all(diff(timestamps) > 0))
  
  # Check MIMS values are reasonable (non-negative for valid data)
  # Note: -0.01 indicates abnormal epochs in MIMSunit
  valid_mims <- output_df$mims[output_df$mims >= 0]
  expect_true(length(valid_mims) > 0)
  expect_true(all(is.finite(valid_mims)))
  
  # Cleanup
  cleanup_test_environment(test_env)
})

test_that("Integration: list_participants and process workflow", {
  test_env <- setup_test_environment()
  
  # Create multiple participants including one to exclude
  participants <- c("participant_a", "participant_b", "__MACOSX")
  for (pid in participants) {
    if (pid != "__MACOSX") {
      create_synthetic_participant(pid, test_env$raw, n_hours = 1)
    } else {
      # Create __MACOSX directory (should be excluded)
      dir.create(file.path(test_env$raw, pid), showWarnings = FALSE)
    }
  }
  
  # List participants (should exclude __MACOSX)
  participant_list <- list_participants(test_env$raw)
  
  expect_false("__MACOSX" %in% participant_list)
  expect_true("participant_a" %in% participant_list)
  expect_true("participant_b" %in% participant_list)
  expect_equal(length(participant_list), 2)
  
  # Cleanup
  cleanup_test_environment(test_env)
})

test_that("Integration: dynamic range computation across multiple files", {
  test_env <- setup_test_environment()
  
  participant_id <- "test_participant_range"
  accel_dir <- file.path(test_env$raw, participant_id, "accelerometer")
  dir.create(accel_dir, recursive = TRUE, showWarnings = FALSE)
  
  # Create files with different value ranges
  base_timestamp <- as.POSIXct("2022-03-21 13:00:00", tz = "UTC")
  
  # File 1: small values
  filepath1 <- file.path(accel_dir, "2022-03-21 13_00_00+00_00.csv")
  df1 <- data.frame(
    timestamp = as.integer(seq(as.numeric(base_timestamp) * 1000, 
                                as.numeric(base_timestamp + 60) * 1000, by = 100)),
    `UTC time` = "2022-03-21T13:00:00.000",
    accuracy = "unknown",
    x = rep(0.1, 601),
    y = rep(0.2, 601),
    z = rep(0.3, 601),
    check.names = FALSE
  )
  write_csv(df1, filepath1)
  
  # File 2: larger values
  filepath2 <- file.path(accel_dir, "2022-03-21 14_00_00+00_00.csv")
  df2 <- data.frame(
    timestamp = as.integer(seq(as.numeric(base_timestamp + 3600) * 1000, 
                                as.numeric(base_timestamp + 3660) * 1000, by = 100)),
    `UTC time` = "2022-03-21T14:00:00.000",
    accuracy = "unknown",
    x = rep(-2.5, 601),
    y = rep(3.0, 601),
    z = rep(-1.8, 601),
    check.names = FALSE
  )
  write_csv(df2, filepath2)
  
  # Get files and compute dynamic range
  hourly_files <- get_hourly_files(participant_id, test_env$raw)
  hourly_data <- lapply(hourly_files, read_hourly_file)
  hourly_data <- Filter(Negate(is.null), hourly_data)
  
  dynamic_range <- compute_dynamic_range(hourly_data)
  
  # Should capture the full range across both files
  expect_equal(dynamic_range[1], -2.5)
  expect_equal(dynamic_range[2], 3.0)
  
  # Cleanup
  cleanup_test_environment(test_env)
})

# ==============================================================================
# Run Integration Tests
# ==============================================================================

cat("\n")
cat(paste(rep("=", 70), collapse = ""), "\n")
cat("Running Integration Tests for compute_mims_r.R\n")
cat(paste(rep("=", 70), collapse = ""), "\n\n")

test_results <- test_file(here("tests", "test_integration_mims_r.R"), reporter = "progress")

cat("\n")
cat(paste(rep("=", 70), collapse = ""), "\n")
cat("Integration Test Summary\n")
cat(paste(rep("=", 70), collapse = ""), "\n")

# Print summary
if (!is.null(test_results)) {
  cat("Integration tests completed.\n")
} else {
  cat("Integration tests completed.\n")
}



