#!/usr/bin/env Rscript

# Fit the GAMMs used for the climate-by-urban-form response surfaces.
# The script exports fitted surfaces and marginal effects; it does not detect
# or report thresholds.

suppressPackageStartupMessages({
  library(mgcv)
  library(data.table)
})


parse_args <- function() {
  raw <- commandArgs(trailingOnly = TRUE)
  args <- list(draws = 1000L, seed = 20260825L)
  for (item in raw) {
    pair <- strsplit(sub("^--", "", item), "=", fixed = TRUE)[[1]]
    if (length(pair) == 2L) args[[pair[1]]] <- pair[2]
  }
  if (is.null(args$metrics) || is.null(args$`output-dir`)) {
    stop("Usage: Rscript 04_gamm_response_surfaces.R --metrics=FILE --output-dir=DIR")
  }
  args$draws <- as.integer(args$draws)
  args$seed <- as.integer(args$seed)
  args
}


read_table <- function(path) {
  extension <- tolower(tools::file_ext(path))
  if (extension %in% c("parquet", "pq")) {
    if (!requireNamespace("arrow", quietly = TRUE)) {
      stop("The arrow package is required to read Parquet input.")
    }
    return(as.data.table(arrow::read_parquet(path)))
  }
  fread(path)
}


require_columns <- function(data, columns) {
  missing <- setdiff(columns, names(data))
  if (length(missing)) stop("Missing required columns: ", paste(missing, collapse = ", "))
}


balanced_weights <- function(city_id, event_type) {
  group <- interaction(city_id, event_type, drop = TRUE)
  1 / as.numeric(table(group)[group])
}


model_configurations <- function() {
  list(
    list(
      name = "gpp_loss",
      variable = "gpp_primary",
      response = "loss_magnitude_lag_0_4",
      x = "longterm_aridity_index",
      z = "vegetation_fraction",
      family = gaussian(),
      positive_only = FALSE
    ),
    list(
      name = "evi_recovery_time",
      variable = "evi_vegetation",
      response = "recovery_time_days_after_nadir",
      x = "longterm_aridity_index",
      z = "vegetation_fraction",
      family = Gamma(link = "log"),
      positive_only = TRUE
    ),
    list(
      name = "peak_thermal_excess",
      variable = "thermal_excess_day_all",
      response = "thermal_amplification_lag_0_4",
      x = "vpd_mean_drought",
      z = "built_up_fraction",
      family = gaussian(),
      positive_only = FALSE
    )
  )
}


prepare_model_data <- function(metrics, config) {
  columns <- c(
    "city_id", "event_type", "variable", config$response, config$x, config$z
  )
  if (config$name == "evi_recovery_time") {
    columns <- c(columns, "recovery_observed")
  }
  require_columns(metrics, columns)
  data <- metrics[
    variable == config$variable & event_type %in% c("FD_only", "FD_P"),
    ..columns
  ]
  setnames(data, c(config$response, config$x, config$z), c("response", "x", "z"))

  if (config$name == "evi_recovery_time") {
    data <- data[as.logical(recovery_observed)]
  }
  data <- data[complete.cases(data)]
  if (config$positive_only) data <- data[response > 0]
  data[, event_type := factor(event_type, levels = c("FD_only", "FD_P"))]
  data[, city_id := factor(city_id)]
  data[, analysis_weight := balanced_weights(city_id, event_type)]
  data
}


fit_model <- function(data, config) {
  formula <- response ~ event_type +
    s(x, by = event_type, bs = "tp", k = 6) +
    s(z, by = event_type, bs = "tp", k = 6) +
    ti(x, z, by = event_type, bs = c("tp", "tp"), k = c(5, 5)) +
    s(city_id, bs = "re")

  bam(
    formula,
    data = data,
    family = config$family,
    weights = analysis_weight,
    method = "fREML",
    discrete = TRUE,
    select = TRUE,
    nthreads = 1
  )
}


new_data <- function(data, event_type, x, z) {
  data.table(
    event_type = factor(event_type, levels = levels(data$event_type)),
    city_id = factor(levels(data$city_id)[1], levels = levels(data$city_id)),
    x = x,
    z = z
  )
}


support_mask <- function(grid, observed, x_limits, z_limits, distance = 0.10) {
  gx <- (grid$x - x_limits[1]) / diff(x_limits)
  gz <- (grid$z - z_limits[1]) / diff(z_limits)
  ox <- (observed$x - x_limits[1]) / diff(x_limits)
  oz <- (observed$z - z_limits[1]) / diff(z_limits)
  !exclude.too.far(distance, gx, gz, ox, oz)
}


response_surface <- function(model, data, config) {
  x_limits <- quantile(data$x, c(0.05, 0.95), na.rm = TRUE, names = FALSE)
  z_limits <- quantile(data$z, c(0.05, 0.95), na.rm = TRUE, names = FALSE)
  x_values <- seq(x_limits[1], x_limits[2], length.out = 61)
  z_values <- seq(z_limits[1], z_limits[2], length.out = 61)
  grid <- CJ(x = x_values, z = z_values)

  result <- rbindlist(lapply(levels(data$event_type), function(type) {
    prediction_data <- new_data(data, type, grid$x, grid$z)
    prediction <- predict(
      model,
      newdata = prediction_data,
      type = "response",
      se.fit = TRUE,
      exclude = "s(city_id)"
    )
    supported <- support_mask(
      grid,
      data[event_type == type],
      x_limits,
      z_limits
    )
    data.table(
      model = config$name,
      event_type = type,
      x_variable = config$x,
      z_variable = config$z,
      x = grid$x,
      z = grid$z,
      predicted_response = as.numeric(prediction$fit),
      se = as.numeric(prediction$se.fit),
      ci_low = as.numeric(prediction$fit - 1.96 * prediction$se.fit),
      ci_high = as.numeric(prediction$fit + 1.96 * prediction$se.fit),
      supported = supported
    )
  }))
  result
}


marginal_effects <- function(model, data, config, draws, seed) {
  set.seed(seed)
  x_limits <- quantile(data$x, c(0.05, 0.95), na.rm = TRUE, names = FALSE)
  z_limits <- quantile(data$z, c(0.05, 0.95), na.rm = TRUE, names = FALSE)
  x_values <- seq(x_limits[1], x_limits[2], length.out = 61)
  z0 <- median(data$z, na.rm = TRUE)
  z1 <- min(z0 + 0.10, z_limits[2])
  coefficient_draws <- rmvn(draws, coef(model), vcov(model))

  rbindlist(lapply(levels(data$event_type), function(type) {
    low <- new_data(data, type, x_values, rep(z0, length(x_values)))
    high <- new_data(data, type, x_values, rep(z1, length(x_values)))
    matrix_low <- predict(model, low, type = "lpmatrix", exclude = "s(city_id)")
    matrix_high <- predict(model, high, type = "lpmatrix", exclude = "s(city_id)")

    eta_low <- matrix_low %*% t(coefficient_draws)
    eta_high <- matrix_high %*% t(coefficient_draws)
    response_low <- model$family$linkinv(eta_low)
    response_high <- model$family$linkinv(eta_high)
    effects <- response_high - response_low

    fitted_low <- model$family$linkinv(as.numeric(matrix_low %*% coef(model)))
    fitted_high <- model$family$linkinv(as.numeric(matrix_high %*% coef(model)))
    data.table(
      model = config$name,
      event_type = type,
      climate_variable = config$x,
      urban_variable = config$z,
      climate_value = x_values,
      urban_increment = z1 - z0,
      marginal_effect = fitted_high - fitted_low,
      ci_low = apply(effects, 1, quantile, 0.025),
      ci_high = apply(effects, 1, quantile, 0.975)
    )
  }))
}


main <- function() {
  args <- parse_args()
  output_dir <- args$`output-dir`
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  metrics <- read_table(args$metrics)
  if (!"built_up_fraction" %in% names(metrics) && "built_fraction" %in% names(metrics)) {
    metrics[, built_up_fraction := built_fraction]
  }

  surfaces <- list()
  effects <- list()
  summaries <- list()
  for (config in model_configurations()) {
    data <- prepare_model_data(metrics, config)
    if (nrow(data) == 0L || length(unique(data$event_type)) < 2L) {
      stop("Insufficient data for model: ", config$name)
    }
    model <- fit_model(data, config)
    saveRDS(model, file.path(output_dir, paste0(config$name, "_gamm.rds")))
    capture.output(
      summary(model),
      file = file.path(output_dir, paste0(config$name, "_summary.txt"))
    )
    capture.output(
      k.check(model),
      file = file.path(output_dir, paste0(config$name, "_k_check.txt"))
    )
    capture.output(
      concurvity(model, full = TRUE),
      file = file.path(output_dir, paste0(config$name, "_concurvity.txt"))
    )

    model_summary <- summary(model)
    summaries[[config$name]] <- data.table(
      model = config$name,
      n = nrow(data),
      n_cities = uniqueN(data$city_id),
      family = model$family$family,
      link = model$family$link,
      deviance_explained = model_summary$dev.expl,
      adjusted_r_squared = model_summary$r.sq,
      aic = AIC(model)
    )
    surfaces[[config$name]] <- response_surface(model, data, config)
    effects[[config$name]] <- marginal_effects(
      model, data, config, args$draws, args$seed
    )
  }

  fwrite(rbindlist(surfaces), file.path(output_dir, "gamm_response_surfaces.csv"))
  fwrite(rbindlist(effects), file.path(output_dir, "gamm_marginal_effects.csv"))
  fwrite(rbindlist(summaries), file.path(output_dir, "gamm_model_summary.csv"))
}


main()
