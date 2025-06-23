import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from typing import Optional

def inspect_session_lengths(
    data: pd.DataFrame, 
    animal_id_col: str = 'Participant_ID', 
    session_col: str = 'savetime', 
    violin_plot: bool = False, 
    figsize: tuple = (10, 5)
    ):
	"""
	Inspect the distribution of session lengths for each participant.
	Args:
		data (pd.DataFrame): DataFrame containing the session data with 'Participant_ID' and 'savetime' columns.
	"""
	# Calculate session lengths: number of trials per session for each participant.
	# A session is defined by a unique ('Participant_ID', 'savetime') pair.
	# .size() counts the number of rows (trials) in each group.
	# .reset_index(name='trials_per_session') converts the resulting Series to a DataFrame.
	session_lengths_df = data.groupby([animal_id_col, session_col]).size().reset_index(name='trials_per_session')

	# Create the figure for the plot.
	# Using figsize similar to other plots in the notebook.
	plt.figure(figsize=figsize)

	# Generate the boxplot.
	# x-axis: 'Participant_ID'
	# y-axis: 'trials_per_session' (the calculated session lengths)
	# data: the DataFrame containing this information
	if violin_plot:
		sns.violinplot(x=animal_id_col, y='trials_per_session', data=session_lengths_df)
	else:
		# Use boxplot for better visualization of the distribution.
		# Boxplot is more robust to outliers and provides a clear summary of the data.
		# It shows the median, quartiles, and potential outliers.
		# sns.boxplot(x=animal_id_col, y='trials_per_session', data=session_lengths_df)
		# Use violin plot for a more detailed view of the distribution.
		# sns.violinplot(x=animal_id_col, y='trials_per_session', data=session_lengths_df)
		sns.boxplot(x=animal_id_col, y='trials_per_session', data=session_lengths_df)

	# Customize the plot for better readability.
	plt.xlabel(animal_id_col)
	plt.ylabel('Number of Trials per Session')
	plt.title('Distribution of Session Lengths per Participant')
	plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels to prevent overlap.
	plt.grid(True)  # Add a grid for easier value reading.
	plt.tight_layout()  # Adjust layout to make sure everything fits.
	plt.show()
 

def update_features(
    data_raw: pd.DataFrame, 
    features: dict, 
    verbose: bool = True
	) -> tuple:
	"""
	Function to update features in a DataFrame by mapping non-numeric features to numeric values.
	Args:
		data_raw (pd.DataFrame): The input DataFrame containing the data.
		features (dict): A dictionary containing 'input' and 'output' features.
			- 'input': List of input feature names.
			- 'output': List of output feature names.
		verbose (bool): If True, print detailed information about the mapping process.
	Returns:
		data (pd.DataFrame): The updated DataFrame with non-numeric features mapped to numeric values.
		feature_mappings (dict): A dictionary containing the mappings for non-numeric features.
	"""

	# Ensure the features dictionary contains 'input' and 'output' keys
	if 'input' not in features or 'output' not in features:
		raise ValueError("Features dictionary must contain 'input' and 'output' keys.")
	# Ensure the input and output features are lists
	if not isinstance(features['input'], list) or not isinstance(features['output'], list):
		raise ValueError("'input' and 'output' features must be lists.")
	# Ensure the data_raw is a DataFrame
	if not isinstance(data_raw, pd.DataFrame):
		raise ValueError("data_raw must be a pandas DataFrame.")
	# Ensure the DataFrame is not empty
	if data_raw.empty:
		raise ValueError("data_raw DataFrame is empty.")
	# Ensure the DataFrame contains the specified features
	if not all(feature in data_raw.columns for feature in features['input'] + features['output']):
		raise ValueError("Some features specified in 'input' or 'output' are not present in the DataFrame.")
	# Ensure the DataFrame does not contain any NaN values in the specified features
	if data_raw[features['input'] + features['output']].isnull().values.any():
		raise ValueError("DataFrame contains NaN values in the specified features.")
	# Ensure the DataFrame does not contain any infinite values in the specified features
	if np.isinf(data_raw[features['input'] + features['output']]).values.any(): # type: ignore
		raise ValueError("DataFrame contains infinite values in the specified features.")

	# Dictionary to store mappings for non-numeric features
	feature_mappings = {}
	input_features = features['input']
	output_features = features['output']

	# Create a new DataFrame 'data' as a copy to avoid modifying data_raw directly
	data = data_raw.copy()

	# Combine input and output features, ensuring uniqueness
	all_features_to_check = list(set(input_features + output_features))
	if verbose:
		print("Checking features for non-numeric types and applying mapping...")
  
	# Check features in the list
	for feature in all_features_to_check:
		if feature in data.columns:
			# Check if the column dtype is not numeric and it's not already all NaN (which can be float)
			if not pd.api.types.is_numeric_dtype(data[feature]) and data[feature].notna().any():
				
				# Get unique non-null values
				unique_values = data[feature].dropna().unique()
				# Create mapping
				mapping = {val: i for i, val in enumerate(unique_values)}
				if verbose:
					print(f"Found non-numeric feature: '{feature}'")
					print(f"Applying mapping for '{feature}': {mapping}")
				# Apply mapping
				data[feature] = data[feature].map(mapping)
				# Store mapping
				feature_mappings[feature] = mapping
			# else:
				# Optional: print(f"Feature '{feature}' is numeric or already handled.")
		else:
			if verbose:
				print(f"Warning: Feature '{feature}' specified but not found in DataFrame.")

	if verbose:
		print("\n--- Final Feature Mappings ---")
		print(feature_mappings)
	
	return data, feature_mappings

def format_data(
    data: pd.DataFrame, 
    features: dict, 
    id_col: str = 'Participant_ID', 
    exp_date_col: str = 'savetime',
    distribution_col: Optional[str] = None
    ) -> tuple:
	"""
	Function to format data into input and output arrays for each animal. 
	The output structure is a list of lists. Each inner list contains the data for a specific animal,
	and each inner list contains the data for each session of that animal.
	Animals -- Sessions -- 2D np array (trials x features).
 
	Args:
		data (pd.DataFrame): The input DataFrame containing the data.
		features (dict): A dictionary containing 'input' and 'output' features.
			- 'input': List of input feature names.
			- 'output': List of output feature names.
		id_col (str): Column name for animal IDs.
		exp_date_col (str): Column name for experiment date.
		distribution_col (Optional[str]): Column name for distribution information (if applicable).
	Returns:
		input_data_all (list): List of input data arrays for each animal.
		output_data_all (list): List of output data arrays for each animal.
		animal_ids (list): List of animal IDs 
	"""
    # Ensure the id_col and exp_date_col are present in the DataFrame
	if id_col not in data.columns or exp_date_col not in data.columns:
		raise ValueError(f"Columns '{id_col}' and '{exp_date_col}' must be present in the DataFrame.")

	# Ensure distribution_col is present if specified
	if distribution_col is not None and distribution_col not in data.columns:
		raise ValueError(f"Column '{distribution_col}' must be present in the DataFrame if specified.")

	# Get unique animal IDs
	animal_ids = data[id_col].unique()
 
	# Ensure features are present in the DataFrame
	if not all(feature in data.columns for feature in features['input'] + features['output']):
		raise ValueError("Some features specified in 'input' or 'output' are not present in the DataFrame.")

	# Extract input and output features
	input_features = features['input']
	output_features = features['output']
	# Array for input features
	input_data_all = []
	# Array for output features
	output_data_all = []
	# Array for distribution
	distribution_all = []
	# Populate the arrays
	for i, animal_id in enumerate(animal_ids):
		# contains sessions for one animal
		animal_input_formatted = []
		animal_output_formatted = []
		animal_distribution_formatted = []
  
		animal_data_loop = data[data[id_col] == animal_id]
		session_ids_animal = animal_data_loop[exp_date_col].unique()
		for j, session_id in enumerate(session_ids_animal):
			# Filter data for the specific session
			session_data = animal_data_loop[animal_data_loop[exp_date_col] == session_id]
			# Extract input and output features as numpy arrays
			# Ensure columns exist and handle potential errors if needed
			try:
				# 2d numpy array of features trials x features
				session_input = session_data[input_features].values
				session_output = session_data[output_features].values
				if distribution_col is not None:
					session_distribution = session_data[distribution_col].values
					animal_distribution_formatted.append(session_distribution)
			except KeyError as e:
				print(f"Warning: Skipping animal {animal_id}, session {session_id} due to missing feature column: {e}")
				continue # Skip this session

			# Assign session data for the current animal

			animal_input_formatted.append(session_input)
			animal_output_formatted.append(session_output)

		# Append the animal data to the overall list
		input_data_all.append(animal_input_formatted)
		output_data_all.append(animal_output_formatted)
		if distribution_col is not None:
			distribution_all.append(animal_distribution_formatted)
	return input_data_all, output_data_all, distribution_all, animal_ids


def create_k_fold_indices(
    n_trials_per_session: int, 
    k_folds: int = 5, 
    random_indices: bool = True, 
    verbose: bool = True
    ) -> list:
	"""
	Function to create k-fold indices for cross-validation.
	Args:
		n_trials_per_session (int): Number of trials per session.
		k_folds (int): Number of folds for cross-validation.
		random_indices (bool): If True, shuffle the trial indices randomly.
	Returns:
		kfold_indices (list): List of dictionaries containing training and validation indices for each fold.
	"""
	# Ensure n_trials_per_session is a positive integer
	if not isinstance(n_trials_per_session, int) or n_trials_per_session <= 0:
		raise ValueError("n_trials_per_session must be a positive integer.")
	# Ensure k_folds is a positive integer
	if not isinstance(k_folds, int) or k_folds <= 0:
		raise ValueError("k_folds must be a positive integer.")
	# Ensure k_folds is less than or equal to n_trials_per_session
	if k_folds > n_trials_per_session:
		raise ValueError("k_folds cannot be greater than n_trials_per_session.")
	# Ensure random_indices is a boolean
	if not isinstance(random_indices, bool):
		raise ValueError("random_indices must be a boolean value.")
	# Ensure n_trials_per_session is divisible by k_folds
	if n_trials_per_session % k_folds != 0:
		raise ValueError("n_trials_per_session must be divisible by k_folds.")
	# Ensure n_trials_per_session is greater than k_folds
	if n_trials_per_session <= k_folds:
		raise ValueError("n_trials_per_session must be greater than k_folds.")

	# Create indices for the trials within a session (0 to n_trials_per_session - 1)
	trial_indices = np.arange(n_trials_per_session)

	# Shuffle the trial indices randomly to ensure unbiased folds
	if random_indices:
		np.random.seed(42) # for reproducibility
		np.random.shuffle(trial_indices)

	# Split the shuffled trial indices into k folds
	# np.array_split handles cases where n_trials_per_session is not perfectly divisible by k_folds
	trial_folds = np.array_split(trial_indices, k_folds)

	# Prepare lists to store the trial indices for each fold
	kfold_indices = []

	# Generate training and validation trial indices for each fold
	for i in range(k_folds):
		# The i-th fold contains the validation trial indices
		val_trial_idx = trial_folds[i]

		# All other folds contain the training trial indices
		# Create a list of indices for folds that are NOT the current validation fold
		train_trial_idx_list = [trial_folds[j] for j in range(k_folds) if j != i]

		# Concatenate the arrays in the list to get the training trial indices
		# Check if train_trial_idx_list is not empty before concatenating
		if train_trial_idx_list:
			train_trial_idx = np.concatenate(train_trial_idx_list)
		else: # Handle edge case like k_folds=1
			train_trial_idx = np.array([], dtype=int)

		# Append the trial indices for the current fold to the lists
		train_trial_idx = np.array(train_trial_idx)
		val_trial_idx = np.array(val_trial_idx)
		kfold_indices.append({'train': train_trial_idx, 
                        'val': val_trial_idx})

	if verbose:
		print(f"Total number of trials per session being split: {n_trials_per_session}")
		print(f"Number of folds (k): {k_folds}")
		print("\n--- Example: Fold 1 (Trial Indices) ---")
		print(f"Training trial indices ({len(kfold_indices[0]['train'])} trials) -- First 5 trial indices: {kfold_indices[0]['train'][:5]}")
		print(f"Validation trial indices ({len(kfold_indices[0]['val'])} trials) -- First 5 trial indices: {kfold_indices[0]['val'][:5]}")
  
	return kfold_indices

def split_data(
    data: Optional[np.ndarray], 
    fold_idx: int = -1, 
    kfold_indices: Optional[list] = None, 
    test: bool = False
	) -> tuple:
	"""
	Function to split data into training and validation sets based on k-fold indices.
	Args:
		data (Optional[np.ndarray]): The input data to be split.
		fold_idx (int): The index of the fold to use for validation.
		kfold_indices (list): List of dictionaries containing training and validation indices for each fold.
		test (bool): If True, return the entire data without splitting.
	Returns:
		train_data (np.ndarray): The training data for the specified fold.
		val_data (np.ndarray): The validation data for the specified fold.
	"""
	if data is None:
		return None, None  # If no data is provided, return None for both train and validation data
	if test:
		return data, None
	else:
		# Ensure kfold_indices is provided
		if kfold_indices is None:
			raise ValueError("kfold_indices must be provided for data splitting.")
		# Ensure fold_idx is within the valid range
		if fold_idx < 0 or fold_idx >= len(kfold_indices):
			raise ValueError(f"fold_idx must be between 0 and {len(kfold_indices) - 1}.")

		# Split the data into training and validation sets based on the fold index
		train_data = data[kfold_indices[fold_idx]['train']]
		val_data = data[kfold_indices[fold_idx]['val']]
		return train_data, val_data


def pick_trials_inspect(input_data, trial_length_threshold = 50, verbose = True, plot = True):
	"""
	Function to inspect the fraction of sessions with trial lengths greater than a specified threshold.
	Args:
		input_data (list): List of input data arrays for each animal.
		trial_length_threshold (int): Threshold for trial length to consider.
		verbose (bool): If True, print detailed information about the analysis.
		plot (bool): If True, plot the distribution of trial lengths.
	Returns:
		freq_trials (np.ndarray): Array of fractions of sessions with trial lengths greater than the threshold for each animal.
	"""
 
	freq_trials = []

	for animal_idx in range(len(input_data)):
		trial_lengths_animal = []
		for session in input_data[animal_idx]:
			trial_lengths_animal.append(len(session))
		trial_lengths_animal = np.array(trial_lengths_animal)
		freq_trials_animal = np.sum(trial_lengths_animal > trial_length_threshold)/len(trial_lengths_animal)
		freq_trials.append(freq_trials_animal)
	freq_trials = np.array(freq_trials)

	if verbose:
		print(f"Number of animals: {len(input_data)}")
		print(f"Mean fraction of sessions with trial length > {trial_length_threshold}: {np.mean(freq_trials):.2f}")
		print(f"Median fraction of sessions with trial length > {trial_length_threshold}: {np.median(freq_trials):.2f}")
		print(f"Minimum fraction of sessions with trial length > {trial_length_threshold}: {np.min(freq_trials):.2f}")
		print(f"Maximum fraction of sessions with trial length > {trial_length_threshold}: {np.max(freq_trials):.2f}")

	if plot:
		plt.figure(figsize=(10, 6))
		plt.hist(freq_trials, cumulative=False, bins=20, color='lightblue', alpha=1)
		plt.axvline(x=np.mean(freq_trials), color='red', linestyle='--', label='Mean') # type: ignore
		plt.axvline(x=np.median(freq_trials), color='black', linestyle='--', label='Median') # type: ignore
		plt.legend()
		plt.grid()
		plt.xticks(rotation=45)
		plt.xlabel(f'Fraction of Session to Keep\n(Fraction of sessions with trial length > {trial_length_threshold})')
		plt.ylabel('Number of Animals')
		plt.title(f'Distribution of Animal Data with Trial Length > {trial_length_threshold}')
		plt.tight_layout()
		plt.show()

def pick_trials(
    input_data: list, 
    output_data: list, 
    n_trials_to_keep: Optional[int] = None, 
    fraction_to_keep: Optional[float] = None, 
    test_ratio: float = 0.2, 
    random_indices: bool = True, 
    distribution_info: Optional[list] = None):
	"""
	Pick trials from the input data and output data. It first checks if the number of trials to keep is provided. 
	If not, it calculates the number of trials to keep based on the fraction to keep. 
	It then processes the data for each animal, ensuring that the input and output data are in the correct format and split into training and testing sets. 
	
	Args:
		input_data (list): List of input data arrays for each animal.
		output_data (list): List of output data arrays for each animal.
		n_trials_to_keep (int): Number of trials to keep in each session.
		fraction_to_keep (float): Fraction of trials to keep in each session.
		test_ratio (float): Ratio of trials to use for testing.
		random_indices (bool): Whether to shuffle the trials randomly.
		distribution_info (Optional[list]): List of distribution information for each animal, if applicable.
	Returns:
		processed_input_data_train (list): List of processed input data for training.
		processed_output_data_train (list): List of processed output data for training.
		processed_input_data_test (list): List of processed input data for testing.
		processed_output_data_test (list): List of processed output data for testing.
	"""
	if fraction_to_keep is None and n_trials_to_keep is None:
		raise ValueError("Either n_trials_to_keep or fraction_to_keep must be provided.")
	if n_trials_to_keep is not None:
		if isinstance(n_trials_to_keep, (int, float)):
			n_trials_to_keep = [n_trials_to_keep] * len(input_data) # type: ignore
		elif len(n_trials_to_keep) == 1:
			n_trials_to_keep = [n_trials_to_keep[0]] * len(input_data)
		elif len(n_trials_to_keep) != len(input_data):
			raise ValueError("n_trials_to_keep must be a single value or a list with the same length as input_data.")
	else:
		raise NotImplementedError("Fraction to keep is not implemented yet.")

	# if fraction_to_keep is not None:
	# 	if isinstance(fraction_to_keep, (int, float)):
	# 		if fraction_to_keep < 0 or fraction_to_keep > 1:
	# 			raise ValueError("fraction_to_keep must be between 0 and 1.")
	# 		fraction_to_keep = [fraction_to_keep] * len(input_data)
	# 	elif len(fraction_to_keep) == 1:
	# 		fraction_to_keep = [fraction_to_keep[0]] * len(input_data)
	# 	elif len(fraction_to_keep) != len(input_data):
	# 		raise ValueError("fraction_to_keep must be a single value or a list with the same length as input_data.")

	processed_input_data = []
	processed_output_data = []
 
	processed_input_data_train = []
	processed_output_data_train = []
 
	processed_input_data_test = []
	processed_output_data_test = []
 
	processed_distribution_train = []
	processed_distribution_test = []

	# Iterate through each animal
	for animal_idx in range(len(input_data)):
		animal_input_sessions = input_data[animal_idx]
		animal_output_sessions = output_data[animal_idx]
		if distribution_info is not None:
			animal_distribution_sessions = distribution_info[animal_idx]
			valid_dist = []

		valid_input_sessions_for_animal = []
		valid_output_sessions_for_animal = []

		if n_trials_to_keep is not None:
			pass
		else:
			pass
			# frequency_to_keep related things
		n_trials_to_keep_session = n_trials_to_keep[animal_idx] # type: ignore

		# Iterate through sessions for the current animal
		for session_idx in range(len(animal_input_sessions)):
			input_session = animal_input_sessions[session_idx]
			output_session = animal_output_sessions[session_idx]
			if distribution_info is not None:
				distribution_session = animal_distribution_sessions[session_idx] # type: ignore
			# Check if the session has enough trials
			if input_session.shape[0] >= n_trials_to_keep_session:
				# Truncate to keep only the first 'trials_to_keep' trials
				input_session_truncated = input_session[:n_trials_to_keep_session, :]
				output_session_truncated = output_session[:n_trials_to_keep_session, :]

				valid_input_sessions_for_animal.append(input_session_truncated)
				valid_output_sessions_for_animal.append(output_session_truncated)
				if distribution_info is not None:
					valid_dist.append(distribution_session[:n_trials_to_keep_session, :]) # type: ignore

		# After checking all sessions, stack the valid ones into a 3D array for the animal
		if valid_input_sessions_for_animal: # Check if any sessions were valid
			animal_input_3d = np.stack(valid_input_sessions_for_animal, axis=0)
			animal_output_3d = np.stack(valid_output_sessions_for_animal, axis=0)
			if distribution_info is not None:
				animal_distribution_3d = np.stack(valid_dist, axis=0) # type: ignore

			# Ensure the dimensions are correct
			if animal_input_3d.ndim != 3 or animal_output_3d.ndim != 3:
				raise ValueError("Input and output data must be 3-dimensional arrays.")

			animal_input_3d = np.transpose(animal_input_3d, (1, 0, 2)) # Shape: (trials_to_keep, num_sessions, num_features)
			animal_output_3d = np.transpose(animal_output_3d, (1, 0, 2)) # Shape: (trials_to_keep, num_sessions, num_features)
			if distribution_info is not None:
				animal_distribution_3d = np.transpose(animal_distribution_3d, (1, 0, 2)) # type: ignore
    
			# Split the data into training and testing sets
			trial_indices = np.arange(animal_input_3d.shape[0])
   
			# Shuffle the indices if random_indices is True
			if random_indices:
				np.random.seed(42)
				np.random.shuffle(trial_indices)

			# Training and test dataset indices
			train_indices = trial_indices[:int(len(trial_indices) * (1 - test_ratio))]
			test_indices = trial_indices[int(len(trial_indices) * (1 - test_ratio)):]
   
			# Training data
			animal_input_3d_train = animal_input_3d[train_indices]
			animal_output_3d_train = animal_output_3d[train_indices]
			# Test data
			animal_input_3d_test = animal_input_3d[test_indices]
			animal_output_3d_test = animal_output_3d[test_indices]

			if distribution_info is not None:
				animal_distribution_3d_train = animal_distribution_3d[train_indices] # type: ignore
				animal_distribution_3d_test = animal_distribution_3d[test_indices] # type: ignore
   
			# Append the processed data for this animal
			processed_input_data_train.append(animal_input_3d_train)
			processed_output_data_train.append(animal_output_3d_train)
			processed_input_data_test.append(animal_input_3d_test)
			processed_output_data_test.append(animal_output_3d_test)

			if distribution_info is not None:
				processed_distribution_train.append(animal_distribution_3d_train) # type: ignore	
				processed_distribution_test.append(animal_distribution_3d_test) # type: ignore

		else:
			# Handle cases where an animal has no sessions meeting the criteria
			processed_input_data.append(None) # Or np.empty((0, trials_to_keep, num_input_features))
			processed_output_data.append(None) # Or np.empty((0, trials_to_keep, num_output_features))
			if distribution_info is not None:
				processed_distribution_train.append(None)
				processed_distribution_test.append(None)
	
			print(f"Animal {animal_idx}: No sessions met the trial threshold.")
	if distribution_info is None:
		processed_distribution_train = None
		processed_distribution_test = None
  
	processed_data = {'train': {'input': processed_input_data_train, 
                             	'output': processed_output_data_train,
                             	'distribution': processed_distribution_train},
					  'test': {'input': processed_input_data_test, 
                				'output': processed_output_data_test,
                				'distribution': processed_distribution_test}}
	return processed_data