from dp_utils.permutation_data import read_flip_permutations_and_rearrangements
df2 = read_flip_permutations_and_rearrangements(csv_input_file_path="/home/tadas/code/deltaProteinDesignUtils/data/deltaprot_permutation_data/non_symmetric_paths_edge_linkers_only.csv") 
df = read_flip_permutations_and_rearrangements(csv_input_file_path="/home/tadas/code/deltaprot_heme_binder_design/paths/non_symmetric_paths_edge_linkers_only.csv")



# violinplot of path_score_v3 vs orientation_code. one side is df, the other df2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
df["path_type"] = "incomplete"
df2["path_type"] = "complete"
concat_df = pd.concat([df,df2])
fig, ax = plt.subplots(figsize=(15,4))
sns.stripplot(data=concat_df, y="path_score_v3", x="orientation_code", hue="path_type", size=1,dodge=True)
# set title
ax.set_title("path_score_v3 of complete and incomplete path (n-1)")
# save fig as png and svg at /home/tadas/code/deltaprot_heme_binder_design/paths
fig.savefig("/home/tadas/code/deltaprot_heme_binder_design/paths/path_visuals/path_score_v3_vs_orientation_code.png")
fig.savefig("/home/tadas/code/deltaprot_heme_binder_design/paths/path_visuals/path_score_v3_vs_orientation_code.svg")





from dp_utils.helix_assembly.visualising_deltaprots import visualise_all_paths



np.random.seed(0)

def select_three(group):
    # get sorted unique scores
    scores = np.sort(group['path_score_v3'].unique())
    min_s, max_s = scores[0], scores[-1]
    mid_val = (min_s + max_s) / 2
    
    # find the unique score closest to the midpoint
    closest = scores[np.abs(scores - mid_val).argmin()]
    
    # helper to sample one row for a given score
    def pick(score):
        return group[group['path_score_v3'] == score].sample(1)
    
    # pick best, worst, and middle
    best_row  = pick(max_s)
    worst_row = pick(min_s)
    mid_row   = pick(closest)
    
    return pd.concat([best_row, worst_row, mid_row])


def pick_bests_max10(group: pd.DataFrame) -> pd.DataFrame:
    # 1) Find the top score in this orientation
    max_score = group['path_score_v3'].max()
    # 2) Filter to only those at that score
    bests = group[group['path_score_v3'] == max_score]
    # 3) If more than 10 tied rows, sample 10; otherwise keep all
    if len(bests) > 2:
        return bests.sample(2)
    return bests

# # apply to each orientation code
# trimmed = (
#     df
#     .groupby('orientation_code', group_keys=False)
#     .apply(select_three)
#     .reset_index(drop=True)
# )

# visualise_all_paths(df=trimmed,save_folder="/home/tadas/code/deltaprot_heme_binder_design/paths/path_visuals/best_middle_worst")



# Apply across all orientation_codes
best_capped_df = (
    df
    .groupby('orientation_code', group_keys=False)
    .apply(pick_bests_max10)
    .reset_index(drop=True)
)

# Then visualize them
visualise_all_paths(
    df=best_capped_df,
    save_folder="/home/tadas/code/deltaprot_heme_binder_design/paths/path_visuals/all_best"
)