import numpy as np

def preprocess_validate_data_dim(all_animals_xs, all_animals_ys,animal_idx_to_process_list, n_animals, verbose = True):
	"""
	###CHANGE THIS FUNCTION DEPENING ON YOUR DATASET###
	Validates the dimensions of the input and target data arrays.
	Args:
		all_animals_xs: (np.ndarray) Input data for all animals.
		all_animals_ys: (np.ndarray) Target data for all animals.
		animal_idx_to_process_list: (list) List of animal indices to process.
		n_animals: (int) Total number of animals in the dataset.
	Returns:
		animal_idx_to_process_list: (list) Validated list of animal indices to process.
		Note: If None is passed, it will process all animals.
  
	Raises:
		ValueError: If the input or target data dimensions are not as expected.
		IndexError: If the animal index to process is out of bounds. 
	"""
 
	if all_animals_xs.ndim != 4 or all_animals_ys.ndim != 4:
		raise ValueError(f"Expected 4D arrays (animals, time, sequences, features), but got xs shape {all_animals_xs.shape} and ys shape {all_animals_ys.shape}")

	if animal_idx_to_process_list is not None:
		if type(animal_idx_to_process_list) is not list:
			animal_idx_to_process_list = [animal_idx_to_process_list]
		if any(idx >= n_animals for idx in animal_idx_to_process_list) or any(idx < 0 for idx in animal_idx_to_process_list):
			raise IndexError(f"animal_idx_to_process ({animal_idx_to_process_list}) is out of bounds for {n_animals} animals.")
	else:
		animal_idx_to_process_list = list(range(n_animals))
	return animal_idx_to_process_list

def preprocess_extract_dimensions(xs, ys, verbose = True):
	"""
 	###CHANGE THIS FUNCTION DEPENING ON YOUR DATASET###

	Extract dimensions of the input and target data.

	Args:
		xs: Input data array.
		ys: Target data array.

	Returns:
		dimensions: (tuple) Dimensions of the input and target data.
	"""
	sequence_length, total_sequences, n_input_features = xs.shape
	_, _, n_output_features = ys.shape # Assuming target shape is [time, sequences, n_output_features]
	return sequence_length, total_sequences, n_input_features, n_output_features


def preprocess_check_for_nans(xs,ys, verbose = True):
	"""
	Check for NaNs in the input and target data.

	Args:
		xs: Input data array.
		ys: Target data array.

	Returns:
		None
	"""
	# Check for NaNs
	if verbose:
		print(f"Input data 'xs' for selected animal contains NaNs: {np.isnan(xs).any()}")
		print(f"Target data 'ys' for selected animal contains NaNs: {np.isnan(ys).any()}")
	if np.isnan(xs).any() or np.isnan(ys).any():
		print("WARNING: Data contains NaNs. This might cause issues during training.")
  
def preprocess_data_split(xs, ys, total_sequences, batch_size, test_ratio, seed = 42, verbose = True):
	"""
	Perform train/validation/test split on the data.

	Args:
		xs: Input data array.
		ys: Target data array.
		total_sequences: Total number of sequences in the dataset.
		batch_size: Batch size used for training (influences split sizes).
		test_ratio: Proportion of data to reserve for the final test set.
		seed: Random seed for reproducibility.
		verbose: If True, print detailed information during processing.

	Returns:
		test_xs: Test set input data.
		test_ys: Test set target data.
		cv_xs: Cross-validation pool input data.
		cv_ys: Cross-validation pool target data.
		fold_indices: Indices for train/validation splits within the cross-validation pool.
	"""

	if verbose:
		print("\n--- Performing Train/Validation/Test Split ---")
	if total_sequences < batch_size * 3: # Need at least one batch for train, val, test ideally
		raise ValueError(f"Total sequences ({total_sequences}) is less than 3 * batch size ({batch_size}). Cannot create train/val/test splits.")

	test_size = int(total_sequences * test_ratio)
	cv_pool_size = total_sequences - test_size

	# Adjust sizes to be multiples of batch_size for easier batching later
	test_size = max(0, (test_size // batch_size) * batch_size)
	cv_pool_size = total_sequences - test_size # Recalculate CV pool size

	if verbose:
		print(f"Target split: Test Size ~{test_ratio*100:.1f}% ({test_size} sequences), CV Pool ({cv_pool_size} sequences)")

	if cv_pool_size < 2 * batch_size:
		raise ValueError(f"CV Pool size ({cv_pool_size}) is less than 2 * batch_size ({2*batch_size}). Cannot create train/val splits for cross-validation.")
	if test_size == 0 and total_sequences > 0:
		raise ValueError(f"Test size is zero after adjusting for batch size. Cannot create test set.")

	# Shuffle indices before splitting
	all_indices = np.arange(total_sequences)
	np.random.seed(seed) # for reproducible shuffling
	np.random.shuffle(all_indices)

	test_indices = all_indices[:test_size]
	cv_pool_indices = all_indices[test_size:]

	# Create the data splits
	test_xs = xs[:, test_indices, :]
	test_ys = ys[:, test_indices, :]
	cv_xs = xs[:, cv_pool_indices, :]
	cv_ys = ys[:, cv_pool_indices, :]
	n_cv_sequences = cv_xs.shape[1]

	if verbose:
		print(f"Final Split Shapes:")
		print(f"  CV Pool: xs={cv_xs.shape}, ys={cv_ys.shape}")
		print(f"  Test Set: xs={test_xs.shape}, ys={test_ys.shape}")
	return test_xs, test_ys, cv_xs, cv_ys, n_cv_sequences

def preprocess_cross_validation_setup(n_cv_sequences, batch_size, verbose = True):
	"""
	Setup indices for 2-fold cross-validation.
	Args:
		n_cv_sequences: Number of sequences in the cross-validation pool.
		batch_size: Batch size used for training (influences split sizes).
	Returns:
		fold_indices: Indices for train/validation splits within the cross-validation pool.
	"""
	# This prepares indices for k-fold (here, k=2, using even/odd split)
	if verbose:
		print("\n--- Setting up Cross-Validation Folds (from CV Pool) ---")
	cv_internal_indices = np.arange(n_cv_sequences)
	even_indices_rel = cv_internal_indices[::2]
	odd_indices_rel = cv_internal_indices[1::2]

	# Adjust fold sizes based on relative indices and batch_size
	n_even = max(0, (len(even_indices_rel) // batch_size) * batch_size)
	n_odd = max(0, (len(odd_indices_rel) // batch_size) * batch_size)

	if n_even == 0 or n_odd == 0:
		raise ValueError(f"Insufficient data in CV pool folds ({n_even}, {n_odd}) for batch size {batch_size}.")

	even_indices_rel_adjusted = even_indices_rel[:n_even]
	odd_indices_rel_adjusted = odd_indices_rel[:n_odd]

	if verbose:
		print(f"Using {n_even} sequences for one fold split and {n_odd} for the other.")

	# Store the *relative* indices for use with cv_xs and cv_ys during training
	# These define which sequences within the cv_pool belong to train/val in each fold
	fold_indices = [
		{'train': even_indices_rel_adjusted, 'val': odd_indices_rel_adjusted}, # Fold 1: Train on Even, Val on Odd
		{'train': odd_indices_rel_adjusted,  'val': even_indices_rel_adjusted}  # Fold 2: Train on Odd, Val on Even
	]
	if verbose:
		print(f"Fold 1: Train on {len(even_indices_rel_adjusted)} sequences, Val on {len(odd_indices_rel_adjusted)} sequences")
		print(f"Fold 2: Train on {len(odd_indices_rel_adjusted)} sequences, Val on {len(even_indices_rel_adjusted)} sequences")
		print(f"Fold indices prepared for cross-validation.")
	return fold_indices

def preprocess_per_animal(animal_idx_to_process, n_animals, all_animals_xs, all_animals_ys, batch_size, test_ratio, seed = 42, verbose = True):
	"""
	###CHANGE THIS FUNCTION DEPENING ON YOUR DATASET###
	Preprocess data for a specific animal.
	Args:
		animal_idx_to_process: Index of the animal to process.
		n_animals: Total number of animals in the dataset.
		all_animals_xs: Input data for all animals.
		all_animals_ys: Target data for all animals.
		batch_size: Batch size used for training (influences split sizes).
		test_ratio: Proportion of data to reserve for the final test set.
		seed: Random seed for reproducibility.
		verbal: Flag to control verbosity of output messages.
	Returns:
		xs: Input data for the selected animal.
		ys: Target data for the selected animal.
		sequence_length: Length of sequences (time dimension).
		total_sequences: Total number of sequences in the dataset.
		n_input_features: Number of input features.
		n_output_features: Number of output features.
		test_xs: Test set input data.
		test_ys: Test set target data.
		cv_xs: Cross-validation pool input data.
		cv_ys: Cross-validation pool target data.
		n_cv_sequences: Number of sequences in the cross-validation pool.
		fold_indices: Indices for train/validation splits within the cross-validation pool.
	"""
	if animal_idx_to_process >= n_animals or animal_idx_to_process < 0:
			raise IndexError(f"animal_idx ({animal_idx_to_process}) is out of bounds for {n_animals} animals.")

	# i. Select data for the specific animal
	if verbose:
		print(f"\n--- Starting Preprocessing for Animal Index: {animal_idx_to_process} ---")

	xs = all_animals_xs[animal_idx_to_process]
	ys = all_animals_ys[animal_idx_to_process]

	# ii. Extract dimensions for the selected animal
	# Assuming shapes: [time, sequences, features] and [time, sequences, n_output_features]
	sequence_length, total_sequences, n_input_features, n_output_features = preprocess_extract_dimensions(xs, ys, verbose = verbose)

	# iii. Check for NaNs
	preprocess_check_for_nans(xs, ys, verbose = verbose)

	# iv. Train/Validation/Test Split
	test_xs, test_ys, cv_xs, cv_ys, n_cv_sequences = preprocess_data_split(xs, ys, total_sequences, batch_size, test_ratio, seed = seed, verbose = verbose)

	# v. Cross-Validation Setup
	fold_indices = preprocess_cross_validation_setup(n_cv_sequences, batch_size, verbose = verbose)
	if verbose:
		print(f"\n--- Preprocessing Complete for Animal Index: {animal_idx_to_process} ---")
	preprocessed_data = {
		'xs': xs,
		'ys': ys,
		'sequence_length': sequence_length,
		'total_sequences': total_sequences,
		'n_input_features': n_input_features,
		'n_output_features': n_output_features,
		'test_xs': test_xs,
		'test_ys': test_ys,
		'cv_xs': cv_xs,
		'cv_ys': cv_ys,
		'n_cv_sequences': n_cv_sequences,
		'fold_indices': fold_indices,
		'batch_size': batch_size,
		'animal_id': animal_idx_to_process,
		'data_file': None,  # Placeholder for data file path if needed
	}
	# vi. Print summary of available variables
	if verbose:
		print("\n--- Preprocessing Complete ---")
		print("Variables available for training:")
		print("  - xs, ys: Input and target data for the selected animal.")
		print("  - sequence_length, total_sequences: Dimensions of the data.")
		print("  - n_input_features, n_output_features: Number of features in input and target data.")
		print("  - cv_xs, cv_ys: Data pool for cross-validation training/validation.")
		print("  - test_xs, test_ys: Held-out test set.")
		print("  - n_cv_sequences: Number of sequences in the cross-validation pool.")
		print("  - fold_indices: List defining train/validation indices within cv_xs/cv_ys for each fold.")
		print("  - batch_size: Defined batch size.")
		print("  - animal_id: Index of the animal processed.")
		print("  - data_file: Path to the data file (if needed).")
		print("  - xs, ys: Input and target data for the selected animal.")
		print("  - sequence_length, total_sequences: Dimensions of the data.")
		print("  - n_input_features, n_output_features: Number of features in input and target data.")
		print("  - fold_indices: Indices for train/validation splits within the cross-validation pool.")
		print("-----------------------------------------")

	return preprocessed_data