

from dp_utils.design_evaluation.compute_pll import compute_pll_batch
from tqdm import tqdm
import os
import pandas as pd

# def compute_pll_from_name_and_sequence_lists(
#     name_list: list[str],
#     sequence_list: list[str],
#     save_dir: str
# ) -> pd.DataFrame:
#     """
#     Compute PLL scores for a batch of sequences given by parallel lists.

#     Args:
#       names_list    : list of sequence names
#       sequence_list : list of sequences (same length as names_list)
#       save_dir      : directory in which to write pll.csv

#     Returns:
#       DataFrame with columns ["name","sequence","pll"]
#       and also writes save_dir/pll.csv.
#     """
#     if len(name_list) != len(sequence_list):
#         raise ValueError("`names_list` and `sequence_list` must have the same length.")

#     records = []
#     for name, seq in tqdm(zip(name_list, sequence_list), total=len(name_list), desc="Computing PLL"):
#         pll = compute_pll(seq)  # your existing compute_pll function
#         records.append({"name": name, "sequence": seq, "pll": pll})

#     df = pd.DataFrame.from_records(records, columns=["name","sequence","pll"])
#     os.makedirs(save_dir, exist_ok=True)
#     df.to_csv(os.path.join(save_dir, "pll.csv"), index=False)
#     return df

if __name__ == "__main__":
  compute_pll_batch(fasta_file_path="/home/tadas/code/deltaprot_heme_binder_design/outputs/4_sequence_prediction/selected_sequences.fasta",
                    save_dir="/home/tadas/code/deltaprot_heme_binder_design/outputs/6_evaluation")
