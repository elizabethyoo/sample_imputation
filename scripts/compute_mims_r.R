#!/usr/bin/env Rscript
# Compute minutely MIMS units using the MIMSunit package
#
# This script processes raw accelerometer data for all participants and computes
# minutely MIMS units using the R MIMSunit package. Output files are saved to
# data/interim/ with the format {participant_id}_R_mims.csv

# Load required packages
suppressPackageStartupMessages({
  library(MIMSunit)
  library(readr)
  library(dplyr)
  library(here)
})

# Set project root directory
# Find the script's location and set project root to parent directory (sample_imputation/)
# This works when script is run via Rscript or source()
get_script_path <- function() {
  # Try method 1: commandArgs (works with Rscript)
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", args, value = TRUE)
  if (length(file_arg) > 0) {
    script_path <- sub("^--file=", "", file_arg)
    return(normalizePath(script_path))
  }
  # Try method 2: sys.frame (works when sourced)
  for (i in 1:sys.nframe()) {
    if (!is.null(sys.frame(i)$ofile)) {
      return(normalizePath(sys.frame(i)$ofile))
    }
  }
  # Fallback: current working directory
  return(getwd())
}

script_path <- get_script_path()
script_dir <- dirname(script_path)
project_root <- dirname(script_dir)  # Go up from scripts/ to project root
here::set_here(project_root)

# Configuration
# Use here() to define paths relative to project root
RAW_DATA_DIR <- here("data", "raw")
INTERIM_DATA_DIR <- here("data", "interim")
EPOCH <- "1 min"

# ==============================================================================
# Helper Functions
# ==============================================================================

#' List all participant directories
#' 
#' @param raw_dir Path to raw data directory
#' @return Character vector of participant IDs (excluding __MACOSX)
list_participants <- function(raw_dir) {
  dirs <- list.dirs(raw_dir, full.names = FALSE, recursive = FALSE)
  participants <- dirs[dirs != "__MACOSX" & dirs != ""]
  return(sort(participants))
}

#' Get all hourly CSV files for a participant
#' 
#' @param participant_id Participant identifier
#' @param raw_dir Path to raw data directory
#' @return Character vector of file paths
get_hourly_files <- function(participant_id, raw_dir) {
  accel_dir <- file.path(raw_dir, participant_id, "accelerometer")
  if (!dir.exists(accel_dir)) {
    warning(sprintf("Accelerometer directory not found for %s: %s", participant_id, accel_dir))
    return(character(0))
  }
  
  files <- list.files(accel_dir, pattern = "\\.csv$", full.names = TRUE)
  if (length(files) == 0) {
    warning(sprintf("No CSV files found for %s in %s", participant_id, accel_dir))
    return(character(0))
  }
  
  return(sort(files))
}

#' Read and validate a single hourly CSV file
#' 
#' @param file_path Path to CSV file
#' @return Data frame with columns timestamp, x, y, z, or NULL if invalid
read_hourly_file <- function(file_path) {
  tryCatch({
    # Read CSV file
    df <- read_csv(
      file_path,
      col_types = cols(
        timestamp = col_integer(),
        `UTC time` = col_character(),
        accuracy = col_character(),
        x = col_double(),
        y = col_double(),
        z = col_double()
      ),
      show_col_types = FALSE
    )
    
    # Check if file is empty
    if (nrow(df) == 0) {
      warning(sprintf("Empty file: %s", basename(file_path)))
      return(NULL)
    }
    
    # Check required columns
    required_cols <- c("timestamp", "x", "y", "z")
    missing_cols <- setdiff(required_cols, colnames(df))
    if (length(missing_cols) > 0) {
      warning(sprintf("Missing columns in %s: %s", basename(file_path), 
                      paste(missing_cols, collapse = ", ")))
      return(NULL)
    }
    
    # Check for non-numeric values
    numeric_cols <- c("x", "y", "z")
    for (col in numeric_cols) {
      if (!is.numeric(df[[col]])) {
        warning(sprintf("Non-numeric values in column %s of %s", col, basename(file_path)))
        return(NULL)
      }
    }
    
    # Select only required columns
    df <- df[, c("timestamp", "x", "y", "z")]
    
    return(df)
  }, error = function(e) {
    warning(sprintf("Error reading %s: %s", basename(file_path), e$message))
    return(NULL)
  })
}

#' Convert timestamp from milliseconds to POSIXct
#' 
#' @param timestamp_ms Timestamp in milliseconds (integer)
#' @return POSIXct object in UTC timezone
convert_timestamp <- function(timestamp_ms) {
  # Convert milliseconds to seconds and create POSIXct
  timestamp_sec <- as.numeric(timestamp_ms) / 1000
  posix_time <- as.POSIXct(timestamp_sec, origin = "1970-01-01", tz = "UTC")
  return(posix_time)
}

#' Prepare data frame for MIMSunit
#' 
#' Converts data to format expected by MIMSunit: HEADER_TIME_STAMP, X, Y, Z
#' 
#' @param df Data frame with columns timestamp (POSIXct), x, y, z
#' @return Data frame with columns HEADER_TIME_STAMP, X, Y, Z
prepare_for_mimsunit <- function(df) {
  # Sort by timestamp
  df <- df[order(df$timestamp), ]
  
  # Deduplicate timestamps (keep first occurrence)
  df <- df[!duplicated(df$timestamp), ]
  
  # Rename columns to match MIMSunit format
  mims_df <- data.frame(
    HEADER_TIME_STAMP = df$timestamp,
    X = df$x,
    Y = df$y,
    Z = df$z,
    stringsAsFactors = FALSE
  )
  
  return(mims_df)
}

#' Compute dynamic range from accelerometer data
#' 
#' @param df_list List of data frames with x, y, z columns
#' @return Numeric vector of length 2: c(low, high)
compute_dynamic_range <- function(df_list) {
  all_values <- numeric(0)
  
  for (df in df_list) {
    if (!is.null(df) && nrow(df) > 0) {
      values <- c(df$x, df$y, df$z)
      all_values <- c(all_values, values[is.finite(values)])
    }
  }
  
  if (length(all_values) == 0) {
    stop("No valid acceleration values found for dynamic range computation")
  }
  
  low <- min(all_values, na.rm = TRUE)
  high <- max(all_values, na.rm = TRUE)
  
  # Handle edge case: all zeros or constant values
  if (!is.finite(low) || !is.finite(high)) {
    stop("Failed to compute finite dynamic range")
  }
  
  # Use symmetric range centered on zero if data is very small
  if (abs(low) < 0.01 && abs(high) < 0.01) {
    warning("Very small dynamic range detected, using default range (-3, 3)")
    return(c(-3, 3))
  }
  
  return(c(low, high))
}

#' Format timestamp to ISO format with timezone
#' 
#' @param posix_time POSIXct object
#' @return Character vector with ISO format timestamps
format_timestamp_iso <- function(posix_time) {
  # Format as ISO 8601 with timezone
  # Example: "2022-03-21T13:46:00.000000+0000"
  formatted <- format(posix_time, format = "%Y-%m-%dT%H:%M:%S", tz = "UTC")
  
  # Add microseconds and timezone offset
  # Extract seconds since epoch to get fractional part
  secs <- as.numeric(posix_time)
  fractional <- secs - floor(secs)
  microseconds <- sprintf("%06d", round(fractional * 1e6))
  
  # Combine
  iso_string <- paste0(formatted, ".", microseconds, "+0000")
  return(iso_string)
}

#' Process a single participant
#' 
#' @param participant_id Participant identifier
#' @param raw_dir Path to raw data directory
#' @param interim_dir Path to interim data directory
#' @return Logical indicating success
process_participant <- function(participant_id, raw_dir, interim_dir) {
  cat(sprintf("\nProcessing participant: %s\n", participant_id))
  cat(paste(rep("=", 60), collapse = ""), "\n")
  
  # Get hourly files
  hourly_files <- get_hourly_files(participant_id, raw_dir)
  if (length(hourly_files) == 0) {
    warning(sprintf("No hourly files found for %s", participant_id))
    return(FALSE)
  }
  
  cat(sprintf("Found %d hourly CSV files\n", length(hourly_files)))
  
  # Read all hourly files
  hourly_data <- list()
  for (file_path in hourly_files) {
    df <- read_hourly_file(file_path)
    if (!is.null(df)) {
      hourly_data[[length(hourly_data) + 1]] <- df
    }
  }
  
  if (length(hourly_data) == 0) {
    warning(sprintf("No valid data found for %s", participant_id))
    return(FALSE)
  }
  
  cat(sprintf("Successfully read %d valid hourly files\n", length(hourly_data)))
  
  # Combine all data
  combined_df <- bind_rows(hourly_data)
  
  # Convert timestamps
  combined_df$timestamp <- convert_timestamp(combined_df$timestamp)
  
  # Remove any rows with invalid timestamps
  valid_timestamps <- !is.na(combined_df$timestamp)
  combined_df <- combined_df[valid_timestamps, ]
  
  if (nrow(combined_df) == 0) {
    warning(sprintf("No valid timestamps found for %s", participant_id))
    return(FALSE)
  }
  
  cat(sprintf("Total samples: %d\n", nrow(combined_df)))
  cat(sprintf("Time range: %s to %s\n", 
              min(combined_df$timestamp), 
              max(combined_df$timestamp)))
  
  # Compute dynamic range
  dynamic_range <- compute_dynamic_range(hourly_data)
  cat(sprintf("Dynamic range: %.4f to %.4f\n", dynamic_range[1], dynamic_range[2]))
  
  # Prepare data for MIMSunit
  mims_input <- prepare_for_mimsunit(combined_df)
  
  # Compute MIMS units
  tryCatch({
      mims_result <- mims_unit(
        mims_input,
        dynamic_range = dynamic_range,
        epoch = EPOCH
      )
      
      if (is.null(mims_result) || nrow(mims_result) == 0) {
        warning(sprintf("MIMSunit returned empty result for %s", participant_id))
        return(FALSE)
      }
      
      # Format output
      # MIMSunit returns a data frame - check column names
      # The first column should be the timestamp, and there should be a MIMS column
      col_names <- colnames(mims_result)
      
      # Find timestamp column (usually first column or contains "TIME" or "TIMESTAMP")
      timestamp_col <- NULL
      if (length(col_names) > 0) {
        # Check if first column is timestamp-like
        if (inherits(mims_result[[1]], "POSIXct") || 
            any(grepl("TIME", toupper(col_names[1])))) {
          timestamp_col <- col_names[1]
        } else {
          # Look for timestamp column
          ts_idx <- grep("TIME", toupper(col_names))
          if (length(ts_idx) > 0) {
            timestamp_col <- col_names[ts_idx[1]]
          } else {
            timestamp_col <- col_names[1]  # Default to first column
          }
        }
      }
      
      # Find MIMS column (usually contains "MIMS" or is the last numeric column)
      mims_col <- NULL
      mims_idx <- grep("MIMS", toupper(col_names))
      if (length(mims_idx) > 0) {
        mims_col <- col_names[mims_idx[1]]
      } else {
        # Look for numeric columns (excluding timestamp)
        numeric_cols <- sapply(mims_result, is.numeric)
        numeric_cols[col_names == timestamp_col] <- FALSE
        if (any(numeric_cols)) {
          mims_col <- col_names[which(numeric_cols)[1]]
        } else {
          stop("Could not find MIMS column in MIMSunit output")
        }
      }
      
      # Extract timestamp and convert to POSIXct if needed
      if (inherits(mims_result[[timestamp_col]], "POSIXct")) {
        timestamp_posix <- mims_result[[timestamp_col]]
      } else {
        timestamp_posix <- as.POSIXct(mims_result[[timestamp_col]], tz = "UTC")
      }
      
      output_df <- data.frame(
        timestamp = format_timestamp_iso(timestamp_posix),
        mims = as.numeric(mims_result[[mims_col]]),
        stringsAsFactors = FALSE
      )
    
    # Save to CSV
    output_file <- file.path(interim_dir, sprintf("%s_R_mims.csv", participant_id))
    write_csv(output_df, output_file)
    
    cat(sprintf("Generated %d minutely MIMS epochs\n", nrow(output_df)))
    cat(sprintf("Saved to: %s\n", output_file))
    
    return(TRUE)
    
  }, error = function(e) {
    warning(sprintf("Error computing MIMS for %s: %s", participant_id, e$message))
    return(FALSE)
  })
}

# ==============================================================================
# Main Execution
# ==============================================================================

main <- function() {
  # Check if directories exist
  if (!dir.exists(RAW_DATA_DIR)) {
    stop(sprintf("Raw data directory not found: %s", RAW_DATA_DIR))
  }
  
  if (!dir.exists(INTERIM_DATA_DIR)) {
    dir.create(INTERIM_DATA_DIR, recursive = TRUE)
    cat(sprintf("Created interim directory: %s\n", INTERIM_DATA_DIR))
  }
  
  # Get all participants
  participants <- list_participants(RAW_DATA_DIR)
  
  if (length(participants) == 0) {
    stop(sprintf("No participant directories found in %s", RAW_DATA_DIR))
  }
  
  cat(sprintf("Found %d participants: %s\n", 
              length(participants), 
              paste(participants, collapse = ", ")))
  
  # Process each participant
  results <- list()
  for (participant_id in participants) {
    success <- process_participant(participant_id, RAW_DATA_DIR, INTERIM_DATA_DIR)
    results[[participant_id]] <- success
  }
  
  # Summary
  cat("\n", paste(rep("=", 60), collapse = ""), "\n")
  cat("Processing Summary\n")
  cat(paste(rep("=", 60), collapse = ""), "\n")
  
  successful <- sum(unlist(results))
  failed <- length(results) - successful
  
  cat(sprintf("Successful: %d\n", successful))
  cat(sprintf("Failed: %d\n", failed))
  
  if (failed > 0) {
    failed_participants <- names(results)[!unlist(results)]
    cat(sprintf("Failed participants: %s\n", paste(failed_participants, collapse = ", ")))
  }
}

# Run main function if script is executed directly
if (!interactive()) {
  main()
}

