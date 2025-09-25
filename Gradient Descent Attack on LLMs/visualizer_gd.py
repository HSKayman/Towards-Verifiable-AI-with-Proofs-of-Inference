# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# %%
df = pd.read_csv('llama2_activation_diff_results.csv')

# %%
df.dropna(axis=0,inplace=True)

# %%
min_columns = [f'layer_{i}_random_abs_diff' for i in range(32)]  # All 32 layers
df['all_layers_max_of_min'] = df[min_columns].max(axis=1)


# %%
df.columns

# %%
df['all_layers_max_of_min'].min(), df['all_layers_max_of_min'].max()

# %%
df.tail(55)

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('llama2_activation_diff_results.csv')
df.dropna(axis=0, inplace=True)

# Calculate max of min across all layers
min_columns = [f'layer_{i}_random_abs_diff' for i in range(32)]  # All 32 layers
df['all_layers_max_of_min'] = df[min_columns].min(axis=1)

# Set global font size
plt.rcParams.update({'font.size': 20})

# Create the cumulative distribution plot
plt.figure(figsize=(10, 8))

# Get the max_of_min values and sort them
values = df['all_layers_max_of_min'].values
sorted_data = np.sort(values)
y_values = np.arange(1, len(sorted_data) + 1) / len(sorted_data) * 100

# Main curve - step plot with fill
plt.step(sorted_data, y_values, where='post', color='steelblue', linewidth=2.5)
plt.fill_between(sorted_data, y_values, alpha=0.15, color='steelblue', step='post')

# Calculate median
median_value = np.median(sorted_data)

# Key percentiles to highlight
key_percentiles = [0, 1, 10, 25, 50, 75, 90]
for pct in key_percentiles:
    percentile_value = np.percentile(sorted_data, pct)
    plt.axhline(y=pct, color='lightgray', linestyle=':', alpha=0.5)
    plt.plot(percentile_value, pct, marker='o', color='darkred', markersize=6)
    
    # Format as decimal number with appropriate precision
    if percentile_value < 0.001:
        formatted_value = f"{percentile_value:.6f}"
    elif percentile_value < 0.1:
        formatted_value = f"{percentile_value:.4f}"
    else:
        formatted_value = f"{percentile_value:.3f}"
    
    plt.text(percentile_value, pct + 2, formatted_value, 
            color='darkred', ha='center', va='bottom', fontweight='bold', fontsize=20)

# Clean grid and labels
plt.grid(True, linestyle='-', alpha=0.3)
plt.title('Soundness Analysis: Distribution of Random Activation Differences', 
          fontsize=16, pad=20, fontweight='bold')
plt.xlabel('Cross-Model Activation Difference Value', fontsize=20)
plt.ylabel('Soundness Coverage (%)', fontsize=20)

# Set log scale for x-axis to better show the range
plt.xscale('log')

# Better axis limits
plt.xlim(sorted_data.min() * 0.5, sorted_data.max() * 2)
plt.ylim(0, 100)

# Cleaner ticks
plt.yticks(np.arange(0, 101, 20))
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

# Statistics box with decimal formatting
min_val = sorted_data.min()
max_val = sorted_data.max()
mean_val = np.mean(sorted_data)
std_val = np.std(sorted_data)

# Format statistics as decimals
if min_val < 0.001:
    min_str = f"{min_val:.6f}"
else:
    min_str = f"{min_val:.4f}"

if median_value < 0.001:
    median_str = f"{median_value:.6f}"
else:
    median_str = f"{median_value:.4f}"

if max_val < 0.001:
    max_str = f"{max_val:.6f}"
else:
    max_str = f"{max_val:.3f}"

if std_val < 0.001:
    std_str = f"{std_val:.6f}"
else:
    std_str = f"{std_val:.4f}"

stats_text = (f"Min: {min_str}\n"
              f"Median: {median_str}\n" 
              f"Max: {max_str}\n"
              f"Std: {std_str}")

plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, fontsize=20,
        verticalalignment='top', horizontalalignment='left',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.9, edgecolor='gray'))

plt.tight_layout()

# Save plot
save_path = 'DCA-SQ2.pdf'
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.show()

# Print detailed statistics with decimal formatting
print("\n" + "=" * 70)
print("CUMULATIVE DISTRIBUTION ANALYSIS")
print("=" * 70)
print(f"\nTotal samples: {len(values)}")

print("\n" + "=" * 70)
print("Summary Statistics for Max of Min values:")
print("-" * 50)
print(f"Minimum value: {min_str}")
print(f"Maximum value: {max_str}")
print(f"Mean value: {mean_val:.6f}")
print(f"Median value: {median_str}")
print(f"Standard deviation: {std_str}")

print("\n" + "=" * 70)
print("Key Percentiles:")
print("-" * 50)
for pct in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
    percentile_value = np.percentile(sorted_data, pct)
    if percentile_value < 0.001:
        formatted_pct = f"{percentile_value:.6f}"
    elif percentile_value < 0.1:
        formatted_pct = f"{percentile_value:.4f}"
    else:
        formatted_pct = f"{percentile_value:.3f}"
    print(f"{pct:2d}th percentile: {formatted_pct}")


# %%



