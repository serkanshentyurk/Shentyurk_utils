import jax
import numpy as np
from scipy import stats
from typing import Optional, Literal, Tuple

def apply_paired_t_test(group_1: np.ndarray, 
                        group_2: np.ndarray, 
                        side: Optional[Literal['two_sided', 'greater', 'lesser']] = 'two_sided', 
                        verbose: bool = True
                        ) -> Tuple[Optional[float], Optional[float]]:
	'''
	Apply a paired t-test to two groups.
	Args:
		group_1 (array-like): Group 1 data
		group_2 (array-like): Group 2 data
		side (str, optional): Type of t-test ('two_sided', 'greater', 'lesser'). Defaults to 'two_sided'.
		verbose (bool, optional): If True, prints the results. Defaults to True.
	Returns:
		t_statistic (float): The t-statistic from the paired t-test.
		p_value (float): The p-value from the paired t-test.
	'''
	if side not in ['two_sided', 'greater', 'lesser']:
		raise ValueError("Invalid value for 'side'. Choose from 'two_sided', 'greater', or 'lesser'.")
	valid_indices = ~np.isnan(group_1) & ~np.isnan(group_2)
	l1_paired_clean = np.array(group_1)[valid_indices]
	l2_paired_clean = np.array(group_2)[valid_indices]

	if len(l1_paired_clean) > 1 and len(l2_paired_clean) > 1 and len(l1_paired_clean) == len(l2_paired_clean):
		t_statistic, p_value = stats.ttest_rel(l1_paired_clean, l2_paired_clean, alternative = side)
		if verbose:
			print(f"Paired t-test between Group 1 and Group 2 likelihoods:")
			print(f"T-statistic: {t_statistic:.4f}")
			print(f"P-value: {p_value:.4f}")

		if p_value < 0.05:
			if verbose:
				print("The difference between Group 1 and Group 2 likelihoods is statistically significant.")
		else:
			if verbose:
				print("The difference between Group 1 and Group 2 likelihoods is not statistically significant.")
	else:
		if verbose:
			print("Not enough data points or unequal sample sizes after NaN removal for a paired t-test.")
		return None, None
	return t_statistic, p_value


def compute_probability_from_logit(model_outputs: np.ndarray, 
								   log_probability: bool = True
								   ) -> np.ndarray:
	"""
	Function to compute probabilities from logits.
	Args:
		model_outputs (np.ndarray): Model outputs (logits).
		log_probability (bool): If True, compute log probabilities; otherwise, compute probabilities.
	Returns:
		probabilities (np.ndarray): Computed probabilities.
	"""
	if log_probability:
		probabilities = np.array(jax.nn.log_softmax(model_outputs))
	else:
		probabilities = np.array(jax.nn.softmax(model_outputs))
	return probabilities

def compute_normalized_log_likelihood(model_outputs: np.ndarray, 
									  actual_choices: np.ndarray
									  ) -> float:
	"""
	Function to compute the normalized log likelihood of model outputs given actual choices.
	Args:
		model_outputs (np.ndarray): Model outputs for the trials.
		actual_choices (np.ndarray): Actual choices made during the trials.
	Returns:
		normalized_likelihood (float): Normalized likelihood of the model outputs given actual choices.
	"""
	# Ensure model_outputs and actual_choices have the same shape
	if model_outputs.shape[0] != actual_choices.shape[0] or model_outputs.shape[1] != actual_choices.shape[1]:
		raise ValueError("model_outputs and actual_choices must have the same shape.")
	if np.isnan(model_outputs[0,0,-1]):
		model_outputs = model_outputs[..., :-1]
	# Ensure model_outputs has the correct number of classes
	if model_outputs.shape[2] != actual_choices.shape[2] + 1:
		raise ValueError("model_outputs must have one more class than actual_choices.")

	n_sessions = actual_choices.shape[1]
	n_trials_per_session = actual_choices.shape[0]
	predicted_log_choice_probabilities = compute_probability_from_logit(model_outputs, log_probability=True)
	log_likelihood = 0
	n = 0  # Total number of trials across sessions.
	for sess_i in range(n_sessions):
		for trial_i in range(n_trials_per_session):
			actual_choice = int(actual_choices[trial_i, sess_i][0])
			if actual_choice >= 0:  # values < 0 are invalid trials which we ignore.
				log_likelihood += predicted_log_choice_probabilities[trial_i, sess_i, actual_choice]
				n += 1
	if n == 0:
		raise ValueError("No valid trials found in actual_choices.")
	normalized_likelihood = np.exp(log_likelihood / n)
	return normalized_likelihood

def compute_mse(model_outputs: np.ndarray, 
				actual_choices: np.ndarray
				) -> float:
	"""
	Function to compute the mean squared error (MSE) between model outputs and actual choices.
	Args:
		model_outputs (np.ndarray): Model outputs for the trials.
		actual_choices (np.ndarray): Actual choices made during the trials.
	Returns:
		mse (float): Mean squared error between model outputs and actual choices.
	"""
	# Ensure model_outputs and actual_choices have the same shape
	if model_outputs.shape != actual_choices.shape:
		raise ValueError("model_outputs and actual_choices must have the same shape.")
	# Compute the mean squared error
	sse = np.nansum((model_outputs - actual_choices) ** 2, axis=-1)
	mse = np.nanmean(sse)
	return mse	


def evaluate_model_performance(model_outputs: np.ndarray, 
							   actual_choices: np.ndarray, 
							   y_type: Literal['categorical', 'scalar', 'continous', 'mixed'] = 'categorical', 
							   verbose: bool = False
							   ) -> dict:
	"""
	Function to evaluate the performance of a model based on its outputs and actual choices.
	Args:
		model_outputs (np.ndarray): Model outputs for the trials.
		actual_choices (np.ndarray): Actual choices made during the trials.
		y_type (str): Type of output data ('categorical' or 'continuous').
		verbose (bool): If True, print detailed information about the evaluation.
	Returns:
		performance (dict): Dictionary containing performance metrics.
	"""
	if y_type == 'categorical':
		# Compute the normalized log likelihood
		performance_categorical = compute_normalized_log_likelihood(model_outputs, actual_choices)
		performance = {'norm_log_likelihood': performance_categorical}
		if verbose:
			print(f'Normalized Log Likelihood: {100 * performance_categorical:.1f}%')
   
	elif y_type in ['scalar', 'continuous']:
		# Compute the mean squared error
		performance_continuous = compute_mse(model_outputs, actual_choices)
		performance = {'mse': performance_continuous}
		if verbose:
			print(f'Mean Squared Error: {performance_continuous:.4f}')
   
	elif y_type == 'mixed':
		# Compute the normalized log likelihood for categorical part
		model_outputs_categorical = model_outputs[:, :, :2]
		actual_choices_categorical = actual_choices[:, :, :2]
		performance_categorical = compute_normalized_log_likelihood(model_outputs_categorical, actual_choices_categorical)

		# Compute the mean squared error for continuous part
		model_outputs_continuous = model_outputs[:, :, 2:]
		actual_choices_continuous = actual_choices[:, :, 2:]
		performance_continuous = compute_mse(model_outputs_continuous, actual_choices_continuous)
  
		if verbose:
			print(f'Normalized Log Likelihood (Categorical): {100 * performance_categorical:.1f}%')
			print(f'Mean Squared Error (Continuous): {performance_continuous:.4f}')
		performance = {'norm_log_likelihood': performance_categorical, 'mse': performance_continuous}
  
	else:
		raise ValueError(f"Evaluation for y_type '{y_type}' is not supported.")
	# Add any other performance metrics you want to compute here
	# For example, you could compute accuracy, precision, recall, etc. for classification tasks
	# or other metrics for regression tasks.

	return performance
