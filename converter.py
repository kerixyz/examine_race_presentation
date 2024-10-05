import pandas as pd

def main():
    start_df  = pd.read_csv("./merged_df.csv")
    target_df = pd.read_csv("./defendant_image_mapping.csv", index_col = 0).reset_index(drop = True)
    target_df = target_df.sort_values(by = ["index_per_block", "block"])

    counter = {}
    index_per_block = []
    for row_idx, row in start_df.iterrows():
        if row["block_num"] not in counter:
            counter[row["block_num"]] = 1

        index_per_block.append(counter[row["block_num"]])
        counter[row["block_num"]] += 1
    
    start_df["index_per_block"] = index_per_block
        
    rename_dict = {
        "block_num": "block",
        "def_id": "defendant_id",
        "feature": "feature(race-gender-age)"
    }
    start_df.rename(rename_dict, axis = 1, inplace = True)
    start_df = start_df[[
        "feature(race-gender-age)",
        "defendant_id",
        "image_id",
        "index_per_block",
        "block"
    ]]

    start_df = start_df.sort_values(by = ["index_per_block", "block"])


    print("===")
    print(start_df.head())
    print("===")
    print(target_df.head())

if __name__ == "__main__":
    main()