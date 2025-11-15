import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from adjustText import adjust_text
import os
import glob

def read_rgt_data(filepath):
    """
    Read RGT data from CSV and extract elements, constructs, ratings, and loadings.
    """
    df = pd.read_csv(filepath)

    # Get column names (elements are from column 1 onwards, excluding PC1, PC2, Variances, Implicit Pole)
    all_columns = df.columns.tolist()

    # Find where PC1 starts
    pc1_idx = None
    for i, col in enumerate(all_columns):
        if 'PC1' in str(col):
            pc1_idx = i
            break

    if pc1_idx is None:
        raise ValueError("Could not find PC1 column")

    # Elements are columns between index 1 and PC1 (excluding empty columns)
    element_columns = []
    element_indices = []
    for i, col in enumerate(all_columns[1:pc1_idx], start=1):
        if col and str(col).strip() and not str(col).startswith('Unnamed'):
            element_columns.append(col)
            element_indices.append(i)

    # Extract construct names (emergent poles) from first column, skip header row
    constructs_raw = df.iloc[:, 0].tolist()
    # Skip the first entry if it contains "Emergent Pole" (header)
    if 'Emergent Pole' in str(constructs_raw[0]):
        constructs = constructs_raw[1:]
        start_row = 1
    else:
        constructs = constructs_raw
        start_row = 0

    # Extract ratings matrix (skip header row if needed) using actual element column indices
    ratings = df.iloc[start_row:, element_indices].values.astype(float)

    # Extract PC1 and PC2 loadings
    # For P1 Grid 2, use Colab PC1 and Colab PC2 if available
    filename = os.path.basename(filepath)
    if 'P1 Grid 2' in filename and 'Colab PC1' in df.columns and 'Colab PC2' in df.columns:
        pc1_loadings = df['Colab PC1'].iloc[start_row:].values
        pc2_loadings = df['Colab PC2'].iloc[start_row:].values
        print(f"  Using Colab PC1 and Colab PC2 for {filename}")
    else:
        pc1_loadings = df['PC1'].iloc[start_row:].values
        pc2_loadings = df['PC2'].iloc[start_row:].values

    # Extract variance percentages if available
    pc1_variance = None
    pc2_variance = None
    if 'Variances' in df.columns:
        variances = df['Variances'].iloc[start_row:].values
        for var in variances:
            if pd.notna(var):
                var_str = str(var)
                if 'PC1' in var_str:
                    # Extract percentage from "PC1=62.8" format
                    try:
                        pc1_variance = float(var_str.split('=')[1])
                    except:
                        pass
                elif 'PC2' in var_str:
                    try:
                        pc2_variance = float(var_str.split('=')[1])
                    except:
                        pass

    # Remove any NaN values from loadings - require both PC1 and PC2
    valid_idx = ~(np.isnan(pc1_loadings) | np.isnan(pc2_loadings))

    # If no constructs have both PC1 and PC2, try using just PC1
    if not np.any(valid_idx):
        print(f"  Warning: No constructs have both PC1 and PC2 loadings")
        raise ValueError("Insufficient PC data for biplot")

    constructs = [constructs[i] for i in range(len(constructs)) if valid_idx[i]]
    ratings = ratings[valid_idx, :]
    pc1_loadings = pc1_loadings[valid_idx]
    pc2_loadings = pc2_loadings[valid_idx]

    return {
        'elements': element_columns,
        'constructs': constructs,
        'ratings': ratings,
        'pc1_loadings': pc1_loadings,
        'pc2_loadings': pc2_loadings,
        'pc1_variance': pc1_variance,
        'pc2_variance': pc2_variance
    }

def calculate_element_scores(ratings, pc1_loadings, pc2_loadings):
    """
    Calculate element scores by projecting ratings onto principal components.
    """
    # Center the ratings
    ratings_centered = ratings - np.mean(ratings, axis=1, keepdims=True)

    # Create loading matrix
    loadings = np.column_stack([pc1_loadings, pc2_loadings])

    # Calculate scores: scores = ratings_centered.T @ loadings
    scores = ratings_centered.T @ loadings

    return scores

def create_biplot(data, title, output_path, element_scale=1.0):
    """
    Create a biplot with element scores and construct loadings.
    Vectors are flipped (multiplied by -1) and labels are adjusted to avoid overlap.

    Args:
        element_scale: Scaling factor for element scores (default 1.0)
    """
    # Calculate element scores
    scores = calculate_element_scores(
        data['ratings'],
        data['pc1_loadings'],
        data['pc2_loadings']
    )

    # Scale element scores if requested
    scores = scores * element_scale

    # Flip the loadings by -1
    pc1_loadings_flipped = -1 * data['pc1_loadings']
    pc2_loadings_flipped = -1 * data['pc2_loadings']

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 10))

    # Plot element scores as points
    ax.scatter(scores[:, 0], scores[:, 1], c='blue', s=100, alpha=0.6, label='Elements', zorder=3)

    # Add element labels
    element_texts = []
    for i, element in enumerate(data['elements']):
        txt = ax.text(scores[i, 0], scores[i, 1], element,
                     fontsize=10, ha='center', va='bottom', color='darkblue', weight='bold',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='none', alpha=0.7),
                     zorder=4)
        element_texts.append(txt)

    # Plot construct loadings as vectors (flipped)
    vector_scale = 3.0  # Scale factor for visibility
    construct_texts = []

    for i, construct in enumerate(data['constructs']):
        # Draw arrow from origin to loading (flipped)
        ax.arrow(0, 0,
                pc1_loadings_flipped[i] * vector_scale,
                pc2_loadings_flipped[i] * vector_scale,
                head_width=0.15, head_length=0.15, fc='red', ec='red',
                alpha=0.6, linewidth=1.5, zorder=2)

        # Add construct label at the end of the arrow
        txt = ax.text(pc1_loadings_flipped[i] * vector_scale * 1.15,
                     pc2_loadings_flipped[i] * vector_scale * 1.15,
                     construct,
                     fontsize=9, ha='center', va='center', color='darkred',
                     style='italic',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='none', alpha=0.7),
                     zorder=4)
        construct_texts.append(txt)

    # Get current axis limits before adjusting text
    all_x = list(scores[:, 0]) + [pc1_loadings_flipped[i] * vector_scale * 1.2 for i in range(len(data['constructs']))]
    all_y = list(scores[:, 1]) + [pc2_loadings_flipped[i] * vector_scale * 1.2 for i in range(len(data['constructs']))]

    # Remove any NaN or Inf values
    all_x = [x for x in all_x if np.isfinite(x)]
    all_y = [y for y in all_y if np.isfinite(y)]

    if len(all_x) == 0 or len(all_y) == 0:
        raise ValueError("No valid coordinates for plotting")

    x_range = max(all_x) - min(all_x)
    y_range = max(all_y) - min(all_y)
    x_margin = max(x_range * 0.2, 0.5)  # Ensure minimum margin
    y_margin = max(y_range * 0.2, 0.5)

    ax.set_xlim(min(all_x) - x_margin, max(all_x) + x_margin)
    ax.set_ylim(min(all_y) - y_margin, max(all_y) + y_margin)

    # Adjust labels to avoid overlap
    all_texts = element_texts + construct_texts
    try:
        adjust_text(all_texts,
                    arrowprops=dict(arrowstyle='->', color='gray', lw=0.5, alpha=0.5),
                    ax=ax,
                    expand_points=(1.3, 1.3),
                    force_points=(0.2, 0.2),
                    force_text=(0.2, 0.2),
                    expand_text=(1.3, 1.3),
                    expand_objects=(1.3, 1.3),
                    lim=200)
    except Exception as e:
        print(f"  Warning: adjust_text had issues: {e}")

    # Add grid and axes
    ax.axhline(y=0, color='k', linestyle='--', linewidth=0.5, alpha=0.3, zorder=1)
    ax.axvline(x=0, color='k', linestyle='--', linewidth=0.5, alpha=0.3, zorder=1)
    ax.grid(True, alpha=0.3, zorder=1)

    # Labels and title with variance percentages
    pc1_label = 'PC1'
    pc2_label = 'PC2'

    if data.get('pc1_variance') is not None:
        pc1_label = f'PC1 ({data["pc1_variance"]:.2f}%)'
    if data.get('pc2_variance') is not None:
        pc2_label = f'PC2 ({data["pc2_variance"]:.2f}%)'

    ax.set_xlabel(pc1_label, fontsize=12, weight='bold')
    ax.set_ylabel(pc2_label, fontsize=12, weight='bold')
    ax.set_title(title, fontsize=14, weight='bold', pad=20)

    # Legend
    ax.legend(loc='best', fontsize=10)

    # Tight layout
    plt.tight_layout()

    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved biplot to: {output_path}")

    return fig

def main():
    # Find all CSV files
    csv_files = glob.glob('/home/user/pca/*.csv')
    csv_files.sort()

    print(f"Found {len(csv_files)} CSV files")
    print("=" * 80)

    # Create output directory for plots
    output_dir = '/home/user/pca/biplots'
    os.makedirs(output_dir, exist_ok=True)

    # Process each CSV file
    for csv_file in csv_files:
        filename = os.path.basename(csv_file)
        print(f"\nProcessing: {filename}")

        try:
            # Read data
            data = read_rgt_data(csv_file)

            print(f"  Elements: {len(data['elements'])} - {data['elements']}")
            print(f"  Constructs: {len(data['constructs'])}")

            # Create title from filename
            title = filename.replace('.csv', '').replace('RGT Data - ', '')

            # Output path
            output_path = os.path.join(output_dir, filename.replace('.csv', '_biplot.png'))

            # Apply scaling for P1 Grid 2 element scores
            element_scale = 1.5 if 'P1 Grid 2' in filename else 1.0

            # Create biplot
            fig = create_biplot(data, f"Biplot: {title}", output_path, element_scale=element_scale)
            plt.close(fig)

            print(f"  ✓ Successfully created biplot")

        except Exception as e:
            print(f"  ✗ Error processing {filename}: {str(e)}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 80)
    print(f"All biplots saved to: {output_dir}")

if __name__ == "__main__":
    main()
