import pandas as pd

# Load your dataset
# Replace 'your_dataset.csv' with the correct file path
df = pd.read_csv('xuetang.csv')

# Ensure the dataset has 'learner_id' and 'course_id' columns
# Step 1: Extract the first 25 learners and their associated courses
initial_learners = df['learner_id'].unique()[:1000]
initial_courses = df[df['learner_id'].isin(initial_learners)]['course_id'].unique()

# Step 2: Identify learners who have at least 50 courses in common with the initial course pool
# Group by learner_id and find the number of common courses for each learner
common_courses_count = df.groupby('learner_id')['course_id'].apply(
    lambda x: len(set(x) & set(initial_courses))
)

# Filter learners who have at least 50 common courses
additional_learners = common_courses_count[common_courses_count >= 10].index

# Step 3: Combine the initial 25 learners with the additional learners
selected_learners = set(initial_learners).union(set(additional_learners))

# Step 4: Create the sub-dataset with all interactions of the selected learners
final_sub_dataset = df[df['learner_id'].isin(selected_learners)]

# Save the sub-dataset to a new CSV file
final_sub_dataset.to_csv('xuetang_minimal_dataset.csv', index=False)

print("Sub-dataset created and saved as 'sub_dataset.csv'")


minimal_dataset=pd.read_csv("xuetang_minimal_dataset.csv");
# Display the number of rows and columns in the DataFrame
print(f"Number of rows: {minimal_dataset.shape[0]}")
print(f"Number of columns: {minimal_dataset.shape[1]}")

# Alternatively, use the `info` method for a detailed summary
minimal_dataset.info()

# To see the first few rows of the DataFrame (optional)

