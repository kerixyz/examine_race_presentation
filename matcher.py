import pandas as pd
import numpy as np

def categorize_age(age):
    if 0 <= age <= 21:
        return 0
    elif 21 <= age <= 28:
        return 1
    elif 28 <= age <= 35:
        return 2
    elif 35 <= age <= 42:
        return 3
    elif age > 42:
        return 4

def categorize_gender(gender):
    if gender.lower() == 'm':
        return 0
    elif gender.lower() == 'f':
        return 1
    else:
        return None  # In case there are other gender categories

def categorize_race(race, race_mapping):
    return race_mapping.get(race)  # Default to None if not in the mapping

def main(compas_path, chicago_path, output_path, setseedhere):
    # Load datasets
    compas_subset = pd.read_csv(compas_path) if compas_path else pd.DataFrame()
    chicago_faces = pd.read_csv(chicago_path) if chicago_path else pd.DataFrame()

    # Define the mapping for race categories
    race_mapping = {
        'A': 4,  # Asian
        'B': 2,  # Black (African American)
        'L': 3,  # Hispanic
        'W': 1   # White (Caucasian)
    }

    # Process compas_subset dataset
    if not compas_subset.empty:
        compas_subset['age_range'] = compas_subset['age'].apply(categorize_age)
        compas_subset['race'] = compas_subset['race'].apply(lambda x: 5 if x > 5 else x)
        compas_subset.rename(columns={'id': 'def_id'}, inplace=True)

        # Ensure the columns are strings before concatenation
        compas_subset['race'] = compas_subset['race'].astype(str)
        compas_subset['sex'] = compas_subset['sex'].astype(str)
        compas_subset['age_range'] = compas_subset['age_range'].astype(str)

        # Create the 'feature' column by concatenating race, gender, and age_range
        compas_subset['feature'] = compas_subset['race'] + '-' + compas_subset['sex'] + '-' + compas_subset['age_range']

        # Keep relevant columns
        df1 = compas_subset[['block_num', 'def_id', 'feature']]
    else:
        df1 = pd.DataFrame()

    # Process chicago_faces dataset
    if not chicago_faces.empty:
        chicago_faces[['Target', 'Race', 'Gender']] = chicago_faces[['Target', 'Race', 'Gender']].astype('string')
        chicago_faces['Race'] = chicago_faces['Race'].apply(lambda x: categorize_race(x, race_mapping))
        chicago_faces['Gender'] = chicago_faces['Gender'].apply(categorize_gender)
        chicago_faces['Age_Range'] = chicago_faces['Age'].apply(categorize_age)

        chicago_faces.rename(columns={
            'Target': 'image_id',
            'Race': 'race',
            'Gender': 'sex',
            'Age': 'age',
            'Age_Range': 'age_range'
        }, inplace=True)

        # Ensure the columns are strings before concatenation
        chicago_faces['race'] = chicago_faces['race'].astype(str)
        chicago_faces['sex'] = chicago_faces['sex'].astype(str)
        chicago_faces['age_range'] = chicago_faces['age_range'].astype(str)

        # Create the 'feature' column by concatenating race, gender, and age_range
        chicago_faces['feature'] = chicago_faces['race'] + '-' + chicago_faces['sex'] + '-' + chicago_faces['age_range']

        # Keep relevant columns
        df2 = chicago_faces[['image_id', 'feature']]
    else:
        df2 = pd.DataFrame()

    # Merge df1 and df2 based on matching 'feature'
    if not df1.empty and not df2.empty:
        merged_df = pd.merge(df1, df2, on='feature', how='left')

        # Set seed for reproducibility
        seed = int(setseedhere)
        np.random.seed(seed)

        # Assign unique image_id for each def_id and block_num based on feature
        unique_assignments = merged_df.groupby(['def_id', 'block_num']).apply(lambda x: x.sample(1)).reset_index(drop=True)

        # Save the merged DataFrame to a CSV file (optional)
        unique_assignments.to_csv(output_path, index=False)

        # Display the result (for debugging purposes)
        print("\nUnique Assignments DataFrame:")
        print(unique_assignments.head())
    else:
        print("\nOne or both datasets are empty. No merge performed.")

if __name__ == "__main__":
    compas_path = "data/broward_clean.csv"
    chicago_path = "data/CFD.csv"
    output_path = "test.csv"
    setseedhere = "12"
    main(compas_path, chicago_path, output_path,setseedhere)