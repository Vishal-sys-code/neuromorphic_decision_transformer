import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Define the color scheme for the environments
ENV_COLOR_MAP = {
    'acrobot': '#1f77b4',
    'cartpole': '#2ca02c',
    'mountaincar': '#d62728',
    'pendulum': '#9467bd'
}

# Define the plotting style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Helvetica'],
    'font.size': 14,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'lines.linewidth': 2.5,
    'grid.linestyle': '--',
    'grid.color': 'lightgray'
})

def plot_model_performance(data, model_name, output_dir):
    """
    Generates and saves a plot for a single model, showing its performance across all environments.

    Args:
        data (pd.DataFrame): The data for the model.
        model_name (str): The name of the model.
        output_dir (str): The directory to save the plots in.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    for env_name in data['environment'].unique():
        env_data = data[data['environment'] == env_name]
        
        # Calculate the Exponential Moving Average
        ema = env_data['eval_return'].ewm(alpha=0.1, adjust=False).mean()

        # Plot the faint raw data
        ax.plot(env_data['epoch'], env_data['eval_return'], color=ENV_COLOR_MAP.get(env_name, 'gray'), alpha=0.2, linewidth=1)
        
        # Plot the smoothed EMA data
        ax.plot(env_data['epoch'], ema, color=ENV_COLOR_MAP.get(env_name, 'gray'), label=env_name.replace("-", " ").title())

    ax.set_title(f'{model_name.replace("-", " ").title()} Performance')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Eval Return')
    ax.legend(loc='best')
    
    # Save the plot in all specified formats
    file_name = model_name.lower().replace(' ', '_')
    for fmt in ['pdf', 'png', 'svg']:
        output_path = os.path.join(output_dir, f'{file_name}_comparison.{fmt}')
        if fmt == 'png':
            fig.savefig(output_path, dpi=300, bbox_inches='tight')
        else:
            fig.savefig(output_path, bbox_inches='tight')

    plt.close(fig)
    print(f"Saved plots for {model_name} to {output_dir}")

def main():
    """
    Main function to read the summary CSV and generate plots for each model.
    """
    summary_file = 'results/summary.csv'
    if not os.path.exists(summary_file):
        print(f"Error: Summary file not found at {summary_file}")
        return

    df = pd.read_csv(summary_file)
    output_dir = 'results/plots'
    os.makedirs(output_dir, exist_ok=True)

    for model_name in df['model'].unique():
        model_data = df[df['model'] == model_name]
        plot_model_performance(model_data, model_name, output_dir)

if __name__ == '__main__':
    main()