import json
import matplotlib.pyplot as plt

# 1. Load trainer state from checkpoint
with open("SFT/new_outputs/Qwen3Guard-Gen-4B/checkpoint-1100/trainer_state.json", "r") as f:
    trainer_state = json.load(f)

# 2. Extract loss values from log history
steps        = []
train_losses = []
eval_losses  = []
eval_steps   = []

for entry in trainer_state["log_history"]:
    if "loss" in entry:
        steps.append(entry["step"])
        train_losses.append(entry["loss"])
    if "eval_loss" in entry:
        eval_steps.append(entry["step"])
        eval_losses.append(entry["eval_loss"])

# 3. Plot
fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(steps, train_losses, label="Training Loss", color="steelblue", linewidth=1.5)

if eval_losses:
    ax.plot(eval_steps, eval_losses, label="Validation Loss", color="tomato",
            linewidth=2, marker="o", markersize=4)

ax.set_title("MedGuard SFT — Loss Curve (up to step 3250)", fontsize=14, fontweight="bold")
ax.set_xlabel("Step", fontsize=12)
ax.set_ylabel("Loss", fontsize=12)
ax.legend(fontsize=11)
ax.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()

# 4. Save
output_path = "SFT/new_outputs/Qwen3Guard-Gen-4B/loss_Qwen3Guard-Gen-4B.png"
plt.savefig(output_path, dpi=150)
plt.show()
print(f"Plot saved to: {output_path}")