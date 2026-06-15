#!/usr/bin/env Rscript
# Integration tests for xpublish-opendap using R's ncdf4 package.
#
# Usage:
#   Rscript --vanilla test_r_client.R <opendap_url> --test <test_name>
#
# Test names: open, dimensions, variables, variable_shape, read_lat,
#             read_data_slice, attributes

library(ncdf4)

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

args <- commandArgs(trailingOnly = TRUE)

if (length(args) < 3 || args[2] != "--test") {
  cat("Usage: Rscript --vanilla test_r_client.R <url> --test <name>\n")
  quit(status = 1)
}

url <- args[1]
test_name <- args[3]

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

fail <- function(msg) {
  cat("FAIL:", msg, "\n")
  quit(status = 1)
}

pass <- function(msg) {
  cat("PASS:", msg, "\n")
  quit(status = 0)
}

assert_equal <- function(actual, expected, label) {
  if (!identical(actual, expected)) {
    fail(paste0(label, ": expected ", deparse(expected), " got ", deparse(actual)))
  }
}

assert_near <- function(actual, expected, tolerance, label) {
  result <- all.equal(actual, expected, tolerance = tolerance)
  if (!isTRUE(result)) {
    fail(paste0(label, ": ", result))
  }
}

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

test_open <- function() {
  nc <- nc_open(url)
  nc_close(nc)
  pass("opened and closed dataset")
}

test_dimensions <- function() {
  nc <- nc_open(url)
  assert_equal(nc$dim$time$len, 2920L, "time length")
  assert_equal(nc$dim$lat$len, 25L, "lat length")
  assert_equal(nc$dim$lon$len, 53L, "lon length")
  nc_close(nc)
  pass("dimensions match")
}

test_variables <- function() {
  nc <- nc_open(url)
  var_names <- names(nc$var)
  if (!("air" %in% var_names)) fail("'air' not in variables")
  # Coordinate variables live in nc$dim, not nc$var in ncdf4
  dim_names <- names(nc$dim)
  if (!("lat" %in% dim_names)) fail("'lat' not in dimensions")
  if (!("lon" %in% dim_names)) fail("'lon' not in dimensions")
  if (!("time" %in% dim_names)) fail("'time' not in dimensions")
  nc_close(nc)
  pass("variables present")
}

test_variable_shape <- function() {
  nc <- nc_open(url)
  # ncdf4 reports dimensions in Fortran (column-major) order: lon, lat, time
  air_size <- nc$var$air$size
  assert_equal(air_size, c(53L, 25L, 2920L), "air shape (column-major)")
  nc_close(nc)
  pass("variable shape matches")
}

test_read_lat <- function() {
  nc <- nc_open(url)
  lat_vals <- nc$dim$lat$vals
  assert_equal(length(lat_vals), 25L, "lat length")
  assert_near(lat_vals[1], 75.0, 1e-4, "lat[1]")
  assert_near(lat_vals[25], 15.0, 1e-4, "lat[25]")
  nc_close(nc)
  pass("latitude values match")
}

test_read_data_slice <- function() {
  nc <- nc_open(url)
  # 1-based indexing: start=c(1,1,1), count=c(1,1,1) → air[0,0,0] in Python
  val <- ncvar_get(nc, "air", start = c(1, 1, 1), count = c(1, 1, 1))
  assert_near(as.numeric(val), 241.2, 0.1, "air[1,1,1]")
  nc_close(nc)
  pass("data slice matches")
}

test_attributes <- function() {
  nc <- nc_open(url)
  long_name <- ncatt_get(nc, "air", "long_name")
  if (!long_name$hasatt) fail("air missing long_name attribute")
  expected <- "4xDaily Air temperature at sigma level 995"
  assert_equal(long_name$value, expected, "air long_name")
  nc_close(nc)
  pass("attributes match")
}

# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

tests <- list(
  open = test_open,
  dimensions = test_dimensions,
  variables = test_variables,
  variable_shape = test_variable_shape,
  read_lat = test_read_lat,
  read_data_slice = test_read_data_slice,
  attributes = test_attributes
)

if (!(test_name %in% names(tests))) {
  fail(paste0("unknown test: ", test_name, ". Available: ", paste(names(tests), collapse = ", ")))
}

tests[[test_name]]()
