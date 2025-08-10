import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_comparison_results():
    """
    Plot the comparison results from the full_comparison_results.csv file.
    """
    try:
        df = pd.read_csv("full_comparison_results.csv")
    except FileNotFoundError:
        print("full_comparison_results.csv not found. Please run the full comparison first.")
        return

    # Set up the plot style
    sns.set_theme(style="whitegrid")

    # Plot average returns
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x="env_name", y="avg_return", hue="model_name")
    plt.title("Average Return Comparison")
    plt.xlabel("Environment")
    plt.ylabel("Average Return")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("full_comparison_returns.png")
    print("Saved return comparison plot to full_comparison_returns.png")

    # Plot average latency
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x="env_name", y="avg_latency_ms", hue="model_name")
    plt.title("Average Latency Comparison")
    plt.xlabel("Environment")
    plt.ylabel("Average Latency (ms)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("full_comparison_latency.png")
    print("Saved latency comparison plot to full_comparison_latency.png")

    # Plot average spikes
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x="env_name", y="avg_spikes_per_episode", hue="model_name")
    plt.title("Average Spikes per Episode Comparison")
    plt.xlabel("Environment")
    plt.ylabel("Average Spikes per Episode")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("full_comparison_spikes.png")
    print("Saved spike comparison plot to full_comparison_spikes.png")

def plot_training_losses():
    """
    Plot the training losses from the full_training_losses.csv file.
    """
    try:
        df = pd.read_csv("full_training_losses.csv")
    except FileNotFoundError:
        print("full_training_losses.csv not found. Please run the full comparison first.")
        return

    # Set up the plot style
    sns.set_theme(style="whitegrid")

    for index, row in df.iterrows():
        env_name = row["env"]
        snn_losses = eval(row["snn_loss"])
        dsf_losses = eval(row["dsf_loss"])

        plt.figure(figsize=(10, 6))
        plt.plot(snn_losses, label="SNN-DT")
        plt.plot(dsf_losses, label="DSF-DT")
        plt.title(f"Training Loss for {env_name}")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"full_training_loss_{env_name}.png")
        print(f"Saved training loss plot to full_training_loss_{env_name}.png")

if __name__ == "__main__":
    plot_comparison_results()
    plot_training_losses()
