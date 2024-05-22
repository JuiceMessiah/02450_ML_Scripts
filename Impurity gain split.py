import numpy as np

# Given data in the table\
data = {
    'x7=0': [33, 28, 30, 29],
    'x7=1': [4, 2, 3, 5],
    'x7=2': [0, 1, 0, 0]
}

def calculate_impurity(class_counts):
    total = sum(class_counts)
    return 1 - max([count/total for count in class_counts])

def calculate_impurity_gain(data, split_value):
    # Calculate impurity before the split\
    total_counts = [sum(x) for x in zip(*data.values())]
    impurity_before_split = calculate_impurity(total_counts)
    
    # Calculate total number of observations\
    total_observations = sum(total_counts)
    
    # Calculate impurity of the left branch (where x7 = split_value)
    left_branch_counts = data[f'x7={split_value}']
    left_branch_impurity = calculate_impurity(left_branch_counts)
    
    # Calculate impurity of the right branch (where x7 != split_value)\
    right_branch_counts = [total - left for total, left in zip(total_counts, left_branch_counts)]
    right_branch_impurity = calculate_impurity(right_branch_counts)
    
    # Calculate weighted average of impurities after the split\
    left_branch_weight = sum(left_branch_counts) / total_observations
    right_branch_weight = sum(right_branch_counts) / total_observations
    weighted_impurity_after_split = (left_branch_weight * left_branch_impurity +
                                     right_branch_weight * right_branch_impurity)
    
    # Impurity gain is the impurity before the split minus the weighted impurity after the split\
    impurity_gain = impurity_before_split - weighted_impurity_after_split
    return impurity_gain\

# Calculate the impurity gain for the split x7=2\
impurity_gain = calculate_impurity_gain(data, 2)
print(impurity_gain)
