# Let's correct the code and calculate the purity gain for the second split.
# First, we need to update the DataFrame with the values for feature 'x1' and target 'yr'.
# Then we'll calculate the purity gain for the split at 'x1 > 0.26'.

import pandas as pd
import numpy as np


# Updated data from the user's dataset for x1 and yr
data_updated = pd.DataFrame({
    'x1': [-1.1, -0.8, 0.08, 0.18, 0.34, 0.6, 1.42, 1.68],
    'yr': [12, 5, 10, 23, 6, 17, 14, 13]
})

# Define the mean squared error calculation
def mean_squared_error(y_true, y_pred):
    return ((y_true - y_pred) ** 2).mean()

# Function to calculate purity gain
def purity_gain(data, split_feature, split_value):
    left_branch = data[data[split_feature] <= split_value]
    right_branch = data[data[split_feature] > split_value]
    
    # If either branch is empty, we cannot split here according to Hunt's algorithm
    if left_branch.empty or right_branch.empty:
        return None
    
    overall_mean = data['yr'].mean()
    total_variance = mean_squared_error(data['yr'], np.full(data['yr'].shape, overall_mean))
    
    # Weighted average of the MSE of each branch
    weighted_mse = (len(left_branch) * mean_squared_error(left_branch['yr'], np.full(left_branch['yr'].shape, left_branch['yr'].mean())) +
                    len(right_branch) * mean_squared_error(right_branch['yr'], np.full(right_branch['yr'].shape, right_branch['yr'].mean()))) / len(data)
    
    # Purity gain is the total variance minus the weighted average of the MSE of each branch
    return total_variance - weighted_mse

# Calculate purity gain for the second split in the decision tree
# First, we need to find the right branch after the first split (x1 > 0.13)
right_branch_after_first_split = data_updated[data_updated['x1'] > 0.13]

# Now calculate the purity gain for the second split, which is at x1 > 0.26 for the right branch
second_split_purity_gain = purity_gain(right_branch_after_first_split, 'x1', 0.26)

print(second_split_purity_gain)
