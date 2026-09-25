import os
import json
import pandas as pd

def find_best_medguard_checkpoint(checkpoint_dir="Claude/SFT/new_outputs/Qwen3-4B-Instruct/checkpoint-900"):
    # Target the specific JSON file you mentioned
    state_path = os.path.join(checkpoint_dir, "trainer_state.json")
    
    if not os.path.exists(state_path):
        print(f"Error: Could not find {state_path}")
        return

    with open(state_path, "r") as f:
        state_data = json.load(f)

    # Extract log history
    history = state_data.get("log_history", [])
    if not history:
        print("No log history found in trainer_state.json.")
        return
        
    df = pd.DataFrame(history)

    # Filter to only rows that contain evaluation metrics
    eval_df = df[df['eval_loss'].notna()].copy()

    if eval_df.empty:
        print("No evaluation steps found in this trainer_state.json.")
        return

    print("--- MedGuard Checkpoint Analysis ---")
    
    # 1. Approach: Lowest Eval Loss (General Safety)
    best_loss_row = eval_df.loc[eval_df['eval_loss'].idxmin()]
    best_loss_step = int(best_loss_row['step'])
    
    # 2. ThinkGuard Approach: Highest Macro F1 (Nuanced Safety) [cite: 247, 261]
    # This ensures performance isn't just driven by the most frequent categories [cite: 261]
    best_metric_step = None
    if 'eval_macro_f1' in eval_df.columns:
        best_f1_row = eval_df.loc[eval_df['eval_macro_f1'].idxmax()]
        best_metric_step = int(best_f1_row['step'])
        print(f"Peak Macro F1: {best_f1_row['eval_macro_f1']:.4f} at Step {best_metric_step}")
    
    print(f"Lowest Eval Loss: {best_loss_row['eval_loss']:.4f} at Step {best_loss_step}")

    # Determine the winning step
    # Priority: Macro F1 (per Paper) > Eval Loss [cite: 247]
    final_best_step = best_metric_step if best_metric_step else best_loss_step
    
    # The paper notes that 'Cla-Exp' (Classification then Explanation) is the 
    # superior strategy for the final model choice [cite: 427, 429]
    recommended_path = f"medguard_outputs/checkpoint-{final_best_step}"
    
    print(f"\nFinal Selection based on Paper Logic: {recommended_path}")
    
    if not os.path.exists(recommended_path):
        print(f"Warning: Directory {recommended_path} not found on disk.")
    
    return recommended_path

if __name__ == "__main__":
    # Pointing to the specific directory you provided
    best_path = find_best_medguard_checkpoint("Claude/SFT/new_outputs/Qwen3-4B-Instruct/checkpoint-900")