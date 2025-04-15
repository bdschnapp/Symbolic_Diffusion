from scipy.optimize import minimize
import numpy as np
import time
import re
import os
import concurrent.futures
import config
import matplotlib.pyplot as plt


def objective_for_optimization_old(param_vector, rpn_expr, const_names, x_data, y_data, epsilon=1e-8):
    """
    Compute the Normalized Root Mean Squared Error (NRMSE) using the standard deviation of the target data.
    NRMSE = RMSE / (std(y_data) + epsilon)
    This provides a loss value less sensitive to the scale of y_data.
    Returns a large value if predictions are invalid or standard deviation of y is near zero.
    """
    params = {name: value for name, value in zip(const_names, param_vector)}

    try:
        predictions = model_predictions(rpn_expr, x_data, params)

        # Check for NaN/Inf in predictions which can occur from invalid params/expressions
        if not np.all(np.isfinite(predictions)):
            return 1e10  # Return a large error if predictions are invalid

    except Exception:
        # Catch errors during model prediction (e.g., math errors in RPN evaluation)
        return 1e10  # Return a very large error

    # Ensure predictions and y_data have compatible shapes if necessary
    # (Assuming model_predictions handles broadcasting or returns correct shape)
    if predictions.shape != y_data.shape:
        # Handle shape mismatch - returning large error is safest
        return 1e10

    # Compute the residuals
    residuals = predictions - y_data

    # Compute the Mean Squared Error (MSE)
    mse = np.mean(residuals ** 2)

    # Compute the Root Mean Squared Error (RMSE)
    rmse = np.sqrt(mse)

    # Compute the standard deviation of the ground truth data
    std_y = np.std(y_data)

    # Compute the Normalized RMSE (NRMSE)
    # Add epsilon to prevent division by zero or instability if std_y is very small
    if std_y < epsilon:
        # If standard deviation is effectively zero (y_data is constant),
        # return RMSE directly, or a very large number if RMSE is also non-zero.
        # Returning RMSE makes sense as any deviation from the constant value is an error.
        # However, for optimization stability, returning a large value might be better if rmse is not zero.
        # If rmse is also near zero, the fit is good, return small value.
        return rmse if rmse < epsilon else 1e13  # Return large error if std_dev is zero but rmse isn't

    nrmse = rmse / std_y

    # Final check for sanity of the result (e.g. if rmse was Inf)
    if not np.isfinite(nrmse):
        return 1e10  # Return a large error

    return nrmse


# --- Top-Level Objective Function ---
def objective_for_optimization(param_vector, rpn_expr, const_names, x_data, y_data, epsilon=1e-6):
    """
    Compute the normalized Mean Squared Error (MSE) between predictions and ground truth.
    Normalization is done using ||y||_2 + epsilon to avoid division by zero.
    """
    params = {name: value for name, value in zip(const_names, param_vector)}
    predictions = model_predictions(rpn_expr, x_data, params)

    if np.isnan(predictions).any():
        return 1e10

    # Compute the residuals
    residuals = predictions - y_data

    # Compute the normalization factor ||y||_2 + epsilon
    normalization_factor = np.linalg.norm(y_data) + epsilon

    # Compute the normalized MSE
    normalized_mse = np.mean((residuals ** 2) / normalization_factor)

    return normalized_mse if not (np.isinf(normalized_mse) or np.isnan(normalized_mse)) else 1e12


# --- Top-Level Worker Function for a Single L-BFGS-B Run ---
def run_single_lbfgsb(initial_guess, objective_func, obj_args, bounds):
    """
    Performs a single run of L-BFGS-B minimization.
    Designed to be called by the ProcessPoolExecutor.
    Returns the OptimizeResult object.
    """
    try:
        result = minimize(
            objective_func,
            initial_guess,
            args=obj_args,
            method='L-BFGS-B',
            bounds=bounds,
            # Optional: Add options here if needed for individual runs
            # options={'ftol': 1e-7, 'gtol': 1e-5, 'maxiter': 100}
        )
        return result
    except Exception as e:
        # In case minimize itself raises an unexpected error
        print(f" Error within L-BFGS-B worker: {e}")
        # Return a dummy result indicating failure
        from scipy.optimize import OptimizeResult
        return OptimizeResult(x=initial_guess, success=False, status=-1, message=f"Worker Error: {e}", fun=np.inf)


# --- Utility: Uniquify Constant Tokens ---
def uniquify_rpn_constants(rpn_expr):
    tokens = rpn_expr.split()
    new_tokens = []
    count = 0
    constant_names = []
    for token in tokens:
        if token == "C":
            const_name = f"C_{count}"
            new_tokens.append(const_name)
            constant_names.append(const_name)
            count += 1
        else:
            new_tokens.append(token)
    new_expr = " ".join(new_tokens)
    return new_expr, constant_names


# --- Model Predictions ---
def model_predictions(rpn_expr, x_data, params):
    preds = [evaluate_rpn_with_params(rpn_expr, float(x_val), params) if isinstance(x_val, (int, float, np.number)) else np.nan for x_val in x_data]
    return np.array(preds, dtype=float)


# --- Tokenization and Robust RPN Evaluation ---
def evaluate_rpn_with_params(rpn_expr, x_value, params):
    # ... (implementation from previous step remains the same) ...
    tokens = rpn_expr.split()
    stack = []
    EVAL_FUNCTIONS = {
        "sin": np.sin, "cos": np.cos,
        "exp": lambda a: np.exp(a) if a < config.EXP_ARG_MAX else np.inf,
        "log": lambda a: np.log(np.abs(a)) if np.abs(a) > 1e-100 else np.nan,
    }
    EVAL_OPERATORS = {
        '+': np.add, '-': np.subtract, '*': np.multiply,
        '/': lambda a, b: np.divide(a, b) if b != 0 else np.nan,
        '**': lambda a, b: np.power(a, b) if np.isreal(np.power(a+0j, b+0j)) else np.nan,
    }
    try: current_x = float(x_value); assert not (np.isnan(current_x) or np.isinf(current_x))
    except: return np.nan
    variable_map = {'x1': current_x}
    try:
        safe_constants = {k: float(v) for k, v in params.items()}
        assert not any(np.isnan(v) or np.isinf(v) for v in safe_constants.values())
        variable_map.update(safe_constants)
    except: return np.nan
    for token in tokens:
        try:
            if re.match(r'^-?\d+(\.\d+)?([eE][-+]?\d+)?$', token):
                value = float(token); assert not (np.isnan(value) or np.isinf(value))
                stack.append(value)
            elif token in variable_map: stack.append(variable_map[token])
            elif token in EVAL_OPERATORS:
                assert len(stack) >= 2
                op2, op1 = stack.pop(), stack.pop(); assert not any(np.isnan([op1, op2]) | np.isinf([op1, op2]))
                res = EVAL_OPERATORS[token](op1, op2); stack.append(res if not (np.isnan(res) or np.isinf(res)) else np.nan)
            elif token in EVAL_FUNCTIONS:
                assert len(stack) >= 1
                op1 = stack.pop(); assert not (np.isnan(op1) or np.isinf(op1))
                res = EVAL_FUNCTIONS[token](op1); stack.append(res if not (np.isnan(res) or np.isinf(res)) else np.nan)
            else: return np.nan
        except: return np.nan
    assert len(stack) == 1
    final_result = stack[0]
    return float(final_result) if not (np.isnan(final_result) or np.isinf(final_result)) else np.nan


# --- Constant Fitting Function (MODIFIED for parallel multi-start L-BFGS-B) ---
def fit_constants(original_rpn_expr, X_data, Y_data, num_starts=64, max_workers=None, verbose=False):
    """
    Fits constants using multiple runs of L-BFGS-B in parallel.
    """
    start_time = time.time()
    try:
        X_data_flat = np.array(X_data, dtype=float).flatten()
        Y_data_flat = np.array(Y_data, dtype=float).flatten()
    except Exception as e: return original_rpn_expr, {}, f"Data Conv. Fail: {e}"

    if X_data_flat.shape[0] != Y_data_flat.shape[0] or X_data_flat.size == 0:
        return original_rpn_expr, {}, "Bad Data Shape/Empty"

    unique_rpn_expr, constant_names = uniquify_rpn_constants(original_rpn_expr)
    n_constants = len(constant_names)

    if n_constants == 0: return unique_rpn_expr, {}, "Success (no constants)"

    bounds = [(-20, 20)] * n_constants

    # Prepare data for fitting (filter NaNs from Y)
    valid_indices = ~np.isnan(Y_data_flat)
    if not np.all(valid_indices):
        X_fit = X_data_flat[valid_indices]
        Y_fit = Y_data_flat[valid_indices]
        if X_fit.size == 0: return unique_rpn_expr, {}, "All Y were NaN"
    else:
        X_fit = X_data_flat
        Y_fit = Y_data_flat

    # Arguments for the objective function (same for all runs)
    obj_args = (unique_rpn_expr, constant_names, X_fit, Y_fit)

    # --- Parallel Multi-Start L-BFGS-B ---
    best_result = None
    best_error = np.inf
    results_list = [] # To store results from futures

    # Determine max_workers
    if max_workers is None:
        try:
            # Default to using 90% of available cores
            max_workers = int(os.cpu_count() * 9 / 10)
        except NotImplementedError:
            max_workers = 1
    elif max_workers <= 0:
         max_workers = 1 # Ensure at least one worker

    if verbose:
        print(f"Starting {num_starts} L-BFGS-B runs using up to {max_workers} workers...")

    # Generate all initial guesses first
    initial_guesses = [np.random.uniform(low=-2.1, high=2.1, size=n_constants) for _ in range(num_starts)]

    futures = []
    # Use ProcessPoolExecutor for parallel execution
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        for i in range(num_starts):
            future = executor.submit(
                run_single_lbfgsb,             # Worker function
                initial_guesses[i],            # Unique initial guess for this job
                objective_for_optimization,    # Objective function reference
                obj_args,                      # Shared arguments for objective
                bounds                         # Shared bounds
            )
            futures.append(future)

        # Collect results as they complete
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result() # Get the OptimizeResult object
                results_list.append(result)
            except Exception as exc:
                # Catch potential errors during future.result() (e.g., pickling errors in rare cases)
                print(f' L-BFGS-B run generated an exception: {exc}')
                # Optionally append a dummy failure result
                from scipy.optimize import OptimizeResult
                results_list.append(OptimizeResult(x=None, success=False, status=-1, message=f"Future Error: {exc}", fun=np.inf))


    # --- Process Collected Results ---
    successful_runs = 0
    for result in results_list:
        if result is not None and result.success:
            successful_runs += 1
            if result.fun < best_error:
                best_error = result.fun
                best_result = result

    fitted_params = {}
    status_message = "Optimization failed (no successful L-BFGS-B runs)"

    if best_result is not None:
        final_params = best_result.x
        fitted_params = {name: value for name, value in zip(constant_names, final_params)}
        status_message = f"Success ({successful_runs}/{num_starts} LBFGS runs OK): Best Err={best_error:.4e}"
    elif successful_runs > 0:
         status_message = f"Optimization Warning: {successful_runs} LBFGS runs succeeded but no best result found?"

    if verbose:
        elapsed_time = time.time() - start_time
        print(f"Parallel L-BFGS-B finished in {elapsed_time:.2f} seconds.")

    return unique_rpn_expr, fitted_params, status_message


def process_and_plot(record, lbfgs_starts=64, max_workers=None, verbose=False):
    if not all(k in record for k in ["X", "Y", "RPN"]):
        print("Skipping record: Missing keys.")
        return
    try:
        X_sample = np.array(record["X"], dtype=float).flatten()
        Y_sample = np.array(record["Y"], dtype=float).flatten()
    except Exception as e: print(f"Skipping record: Bad X/Y data. Error: {e}"); return
    if X_sample.size == 0 or Y_sample.size == 0 or X_sample.shape != Y_sample.shape:
        print(f"Skipping record: Empty/mismatched X/Y (X:{X_sample.shape}, Y:{Y_sample.shape})."); return
    valid_indices_plot = ~np.isnan(Y_sample)
    X_plot = X_sample[valid_indices_plot]
    Y_plot = Y_sample[valid_indices_plot]
    if X_plot.size == 0: print("Skipping record: All Y values were NaN."); return
    original_rpn_expr = record["RPN"]

    # Pass num_starts and max_workers
    unique_rpn_expr, fitted_params, fit_status = fit_constants(
        original_rpn_expr, X_plot, Y_plot,
        num_starts=lbfgs_starts, max_workers=max_workers
    )

    if verbose:
        print(f"Fit Status: {fit_status}")

    if not fitted_params and "Success" not in fit_status:
        print("Fitting failed.")
        return

    mse = float((fit_status.split(': ')[-1]).split('=')[-1].strip())

    if verbose:
        x_min, x_max = X_plot.min(), X_plot.max()
        if np.isclose(x_min, x_max): x_eval = np.array([x_min])
        else: x_eval = np.linspace(x_min, x_max, 100)
        y_eval = model_predictions(unique_rpn_expr, x_eval, fitted_params)
        plt.figure(figsize=(8, 6))
        y_eval_valid = y_eval[~np.isnan(y_eval)]
        y_min_plot = min(Y_plot.min(), y_eval_valid.min()) if y_eval_valid.size > 0 else Y_plot.min()
        y_max_plot = max(Y_plot.max(), y_eval_valid.max()) if y_eval_valid.size > 0 else Y_plot.max()
        y_range = y_max_plot - y_min_plot
        if np.isclose(y_range, 0): y_range = max(abs(y_min_plot) * 0.2, 1.0)
        plt.plot(x_eval, y_eval, label="Fitted Model", color='blue', linewidth=2)
        plt.scatter(X_plot, Y_plot, color="red", label="Sample Data (Valid)", s=10, alpha=0.7)
        title = "Fitted Model vs Sample Data"
        if "Err=" in fit_status: title += f"\n({fit_status.split(': ')[-1]})"
        plt.title(title)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.ylim(y_min_plot - y_range * 0.1, y_max_plot + y_range * 0.1)
        plt.tight_layout()
        plt.show()

    # return the error
    print(f"MSE: {mse}")
    return mse


def mse_log_histogram(mse_list):
    """
    Plot a histogram of MSE values with logarithmic bin boundaries.
    """
    # Define logarithmic bin boundaries
    bins = np.logspace(np.log10(min(mse_list) + 1e-10), np.log10(max(mse_list) + 1e-10), 30)

    plt.figure(figsize=(10, 5))
    plt.hist(mse_list, bins=bins, edgecolor='black', alpha=0.7)
    plt.xscale('log')
    plt.title("Histogram of $MSE_N$ Values with Logarithmic Bins")
    plt.xlabel("$MSE_N$ (log scale)")
    plt.ylabel("Frequency")
    plt.show()


def mse_bar_chart():
    """
    Plot a bar chart of MSE values.
    """
    counts = [75, 49, 14, 11, 1]
    y_pos = np.arange(len(counts))
    labels = ["Type 1", "Type 2", "Type 3", "Type 4", "Type 5"]

    # Plot the bar chart using bin widths
    plt.bar(y_pos, counts, align='center', edgecolor='black', alpha=0.7)
    plt.xticks(y_pos, labels)
    plt.xlabel("Generated Expression Types")
    plt.ylabel("Count")
    plt.title("Expression Type Bar Chart")
    plt.show()
