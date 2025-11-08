import os
import re
import pandas as pd

def parse_log_file(file_path):
    """
    Parses a single log file to extract epoch and evaluation return.

    Args:
        file_path (str): The path to the log file.

    Returns:
        list: A list of dictionaries, each containing the model, environment, epoch, and eval_return.
    """
    results = []
    try:
        model_name = os.path.basename(file_path).split('_')[0]
        env_name = os.path.basename(file_path).split('_')[1].replace('.log', '')
        
        # A more descriptive name for the Decision Transformer
        if model_name.lower() == 'dt':
            model_name = 'Decision Transformer'

        with open(file_path, 'r') as f:
            for line in f:
                if 'Eval Return' in line:
                    epoch_match = re.search(r'Epoch (\d+)/\d+', line)
                    eval_return_match = re.search(r'Eval Return: ([-+]?\d*\.\d+|\d+)', line)
                    if epoch_match and eval_return_match:
                        epoch = int(epoch_match.group(1))
                        eval_return = float(eval_return_match.group(1))
                        results.append({
                            'model': model_name.upper(),
                            'environment': env_name,
                            'epoch': epoch,
                            'eval_return': eval_return
                        })
    except Exception as e:
        print(f"Error parsing file {file_path}: {e}")
    return results

def main():
    """
    Main function to find log files, parse them, and save the results to a CSV file.
    """
    log_dir = 'baseline_comparisons_results'
    all_results = []
    for root, _, files in os.walk(log_dir):
        for file in files:
            if file.endswith('.log'):
                file_path = os.path.join(root, file)
                all_results.extend(parse_log_file(file_path))

    if not all_results:
        print("No log files found or parsed.")
        return

    df = pd.DataFrame(all_results)
    
    # Create the results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    # Save to CSV
    output_path = 'results/summary.csv'
    df.to_csv(output_path, index=False)
    print(f"Successfully parsed all log files. Results saved to {output_path}")

if __name__ == '__main__':
    main()