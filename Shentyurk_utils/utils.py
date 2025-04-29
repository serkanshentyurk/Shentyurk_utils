import numpy as np
import jax.numpy as jnp
import jax
import os

def load_data(data_file, verbose = True):
	"""
 	###CHANGE THIS FUNCTION DEPENING ON YOUR DATASET###

	Load data from the specified file.

	Args:
		data_file: Path to the .npz data file.

	Returns:
		data: Loaded data dictionary.
		all_animals_xs: Input data for all animals.
		all_animals_ys: Target data for all animals.
		n_animals: Total number of animals in the dataset.
	"""
	if verbose:
		print(f"\n--- Loading Data from {data_file} ---")
	if not os.path.exists(data_file):
		raise FileNotFoundError(f"Data file not found: {data_file}. Please ensure it exists.")

	try:
		data = np.load(data_file)
		all_animals_xs = data['xs']
		all_animals_ys = data['ys']
		n_animals = all_animals_xs.shape[0]
		if verbose:
			print(f"Total animals in dataset: {n_animals}")
			print("Data loaded successfully.")
	except Exception as e:
		raise IOError(f"Failed to load or read data from {data_file}: {e}")

	return data, all_animals_xs, all_animals_ys, n_animals


def compute_likelihood(network_outputs: jnp.ndarray,
                       actual_choices: np.ndarray,
                       target_size: int) -> float:
	"""
	Computes the mean log-likelihood of the actual choices given the network outputs.

	Args:
		network_outputs: The raw output logits from the network.
							Expected shape: (time, sequences, features), where the
							first `target_size` features are the logits for the choice.
		actual_choices: The ground truth target choices (integers).
						Expected shape: (time, sequences).
		target_size: The number of possible choices (classes).

	Returns:
		The mean log-likelihood per sample/timestep. Returns NaN if inputs cause issues.

	Raises:
		ValueError: If input shapes are incompatible.
	"""
	try:
		if network_outputs.ndim != 3:
				raise ValueError(f"Expected network_outputs to have 3 dimensions (time, sequences, features), but got {network_outputs.ndim}")
		if actual_choices.ndim != 2:
				raise ValueError(f"Expected actual_choices to have 2 dimensions (time, sequences), but got {actual_choices.ndim}")
		if network_outputs.shape[0] != actual_choices.shape[0] or network_outputs.shape[1] != actual_choices.shape[1]:
				raise ValueError(f"Time and sequence dimensions mismatch: network_outputs {network_outputs.shape[:2]} vs actual_choices {actual_choices.shape}")
		if network_outputs.shape[2] < target_size:
				raise ValueError(f"Network output features ({network_outputs.shape[2]}) is less than target_size ({target_size})")

		# Ensure actual_choices are integers for one-hot encoding
		actual_choices_int = actual_choices.astype(jnp.int32)

		# Convert actual choices to one-hot encoding
		targets_one_hot = jax.nn.one_hot(actual_choices_int, num_classes=target_size)
		# Expected shape: (time, sequences, target_size)

		# Calculate log probabilities using log_softmax on the relevant output slice
		log_probs = jax.nn.log_softmax(network_outputs[..., :target_size], axis=-1)
		# Expected shape: (time, sequences, target_size)

		# Calculate log-likelihood for the chosen class for each sample and time step
		# Element-wise multiply log_probs with one-hot targets and sum over the class dimension
		log_likelihood_per_sample = jnp.sum(log_probs * targets_one_hot, axis=-1)
		# Expected shape: (time, sequences)

		# Calculate the mean log-likelihood across all time steps and sequences
		mean_log_likelihood = jnp.mean(log_likelihood_per_sample).item()

		# Check for NaN mean_log_likelihood before exponentiating
		if np.isnan(mean_log_likelihood):
			print("Warning: Mean log-likelihood is NaN.")
			return np.nan

		# Exponentiate to get the likelihood
		normalized_likelihood = jnp.exp(mean_log_likelihood)
  
		return mean_log_likelihood, normalized_likelihood

	except Exception as e:
		print(f"Error computing likelihood: {e}")
		# Depending on desired behavior, you might re-raise, return NaN, or return None
		return np.nan
