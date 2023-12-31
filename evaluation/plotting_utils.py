import seaborn as sns
import numpy as np
import string
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from matplotlib.patches import Patch


def measures_boxplot_close(
    scores_db,
    measures,
    models,
    labels,
    colors,
    measure_tags=[],
    ylim=None,
    show_outliers=False,
    savef="",
    title="",
):
    """
    Generates a box plot comparing different models across various measures.
    Creates a box plot for each model, displaying the distribution of scores for each measure.
    Parameters:
    scores_db (dict): A nested dictionary where the first key is the measure, the second key is the model,
                      and the value is a list of scores.
    measures (list of str): A list of measures (e.g., 'Accuracy', 'Precision') to be plotted.
    models (list of str): A list of model names corresponding to keys in scores_db.
    colors (list of str): A list of colors to be used for each model's box plot.
    ylim (tuple of float, optional): A tuple specifying the y-axis limits (min, max).
                    If None, the limits are set automatically.
    Returns:
    None: This function creates and displays a matplotlib plot but does not return any value.
    """
    sns.set_style("whitegrid")
    num_measures = len(measures)
    box_width = 0.8  # Width of each boxplot
    spacing = 1.6  # Spacing between groups of boxplots
    positions = [
        np.arange(len(models)) * (num_measures + spacing) + i
        for i in range(num_measures)
    ]

    fig, ax = plt.subplots(figsize=(max(12, 3.5 * len(models)), 6), dpi=150)

    for i, measure in enumerate(measures):
        # Plotting boxplots
        data = [scores_db[measure][model] for model in models]
        boxplot_elements = ax.boxplot(
            data,
            positions=positions[i],
            widths=box_width,
            patch_artist=True,
            boxprops=dict(alpha=0.6),
            medianprops=dict(color="black"),
            showfliers=show_outliers,
        )

        # Setting the same color for all boxes of a measure
        for patch, color in zip(boxplot_elements["boxes"], colors):
            patch.set_facecolor(color)

        # Adding letter labeling
        if measure_tags:
            letter = measure_tags[i]
        else:
            letter = measure[:2]
        for pos in positions[i]:
            ax.text(
                pos,
                -1.3,
                letter,
                ha="center",
                va="top",
                fontsize=9,
                # fontweight="bold",
            )

    # Adjusting x-axis labels and positions
    ax.set_xticks(
        np.arange(len(models)) * (num_measures + spacing) + num_measures / 2 - 0.5
    )
    ax.set_xticklabels(labels, fontsize=14)

    ax.set_ylabel("Projected Scores", fontsize=14)
    ax.set_title(
        "Model Performance Comparison by Measure" if not title else title, fontsize=16
    )

    if ylim is not None:
        ax.set_ylim(*ylim)

    if savef:
        plt.savefig(savef, facecolor=(1, 1, 1), dpi=100)
    plt.show()


def measures_boxplot(
    scores_db,
    measures,
    models,
    colors,
    labels,
    ylim=None,
    show_outliers=False,
    savef="",
    measure_tags=[],
):
    """
    Generates a box plot comparing different models across various measures.
    Creates a box plot for each measure, displaying the distribution of scores for each model.
    Also annotates each box plot with the mean score value.

    Parameters:
    scores_db (dict): A nested dictionary where the first key is the measure, the second key is the model,
                        and the value is a list of scores.
    measures (list of str): A list of measures (e.g., 'Accuracy', 'Precision') to be plotted.
    models (list of str): A list of model names corresponding to keys in scores_db.
    colors (list of str): A list of colors to be used for each model's box plot.
    ylim (tuple of float, optional): A tuple specifying the y-axis limits (min, max).
                        If None, the limits are set automatically.

    Returns:
    None: This function creates and displays a matplotlib plot but does not return any value.
    """
    sns.set_style("whitegrid")
    num_models = len(scores_db[measures[0]])
    positions = [
        (np.arange(len(measures)) * num_models)
        + np.linspace(0, (len(measures) * 0.5), len(measures), endpoint=False)
        + i
        for i in range(num_models)
    ]
    fig, ax = plt.subplots(figsize=(max(12, 2 * len(models) * len(measures)), 7))

    for i, model in enumerate(models):
        # Plotting boxplots
        data = [scores_db[measure][model] for measure in measures]
        boxplot_elements = ax.boxplot(
            data,
            positions=positions[i],
            widths=0.6,
            patch_artist=True,
            boxprops=dict(facecolor=colors[i], alpha=0.6),
            medianprops=dict(color="black"),
            showfliers=show_outliers,
        )

        # overlaying text
        for j, pos in enumerate(positions[i]):
            med_val = np.median(data[j])
            x_position = pos - 0.4
            y_position = min(med_val, 95)
            ax.text(
                x_position,
                y_position,
                f"{med_val:.2f}",
                ha="center",
                va="center",
                fontsize=14,
                rotation=90,
            )

    ax.set_xticks((positions[-1] + positions[0]) / 2)  # tick in the middle plot
    ax.set_xticklabels(measures)
    legend_patches = [
        Patch(facecolor=color, label=label) for color, label in zip(colors, labels)
    ]
    ax.legend(handles=legend_patches, loc="upper left", bbox_to_anchor=(1, 1))
    plt.subplots_adjust(right=0.75)
    ax.set_ylabel("Scores")
    ax.set_title(
        "Model Performance Comparison: " + " ".join(measure_tags)
        if measure_tags
        else "",
        fontsize=16,
    )

    if ylim is not None:
        ax.set_ylim(*ylim)

    if savef:
        plt.tight_layout()
        plt.savefig(savef, facecolor=(1, 1, 1), dpi=100)
    plt.show()


def plot_3d_centroids(
    data,
    labels,
    colors,
    centroids=True,
    range_limit=0.3,
    plot_size=(1100, 900),
    title="3D PCA Plot",
):
    """
    Plots a 3D PCA scatter plot using Plotly.

    Parameters:
    - data: List of datasets (each dataset is a numpy array with 3 PCA components).
    - labels: List of labels for each dataset.
    - colors: List of colors for each dataset.
    - centroids: If True, will plot centroids for each dataset.
    - range_limit: Limit for the x, y, and z axes.
    - plot_size: Tuple indicating the size of the plot (width, height).
    - title: Title of the plot.
    """
    fig = go.Figure(layout=dict(width=plot_size[0], height=plot_size[1]))
    centroid_list = []
    for dataset, label, color in zip(data, labels, colors):
        fig.add_trace(
            go.Scatter3d(
                x=dataset[:, 0],
                y=dataset[:, 1],
                z=dataset[:, 2],
                mode="markers",
                marker=dict(size=2, color=color, opacity=0.6),
                name=label,
            )
        )
        # Plot centroids if enabled
        if centroids:
            centroid = np.mean(dataset, axis=0)
            centroid_list.append(centroid)
            fig.add_trace(
                go.Scatter3d(
                    x=[centroid[0]],
                    y=[centroid[1]],
                    z=[centroid[2]],
                    mode="markers",
                    marker=dict(size=7, color=color, opacity=1),
                    name=f"{label} centroid",
                )
            )

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis=dict(title="1st PC", range=[-range_limit, range_limit]),
            yaxis=dict(title="2nd PC", range=[-range_limit, range_limit]),
            zaxis=dict(title="3rd PC", range=[-range_limit, range_limit]),
        ),
        legend_title="Legend",
        legend=dict(title="Legend", font=dict(size=20)),
    )

    fig.show()
    return centroid_list
