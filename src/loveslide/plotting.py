import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns
import os
import numpy as np
import pandas as pd
import seaborn.objects as so
import networkx as nx


class Plotter:
    def __init__(self):
        pass
    
    @staticmethod
    def plot_latent_factors(lfs, outdir=None, title='Significant Latent Factors'):
        """
        Plot genes for each latent factor, colored by their sign and ordered by absolute loading values.

        Parameters:
        - lfs: Dictionary of DataFrames, where each key is a latent factor name and value is a DataFrame
            with 'loading', 'AUC', 'corr', and 'color' columns. Index should be gene names.
        - outdir: Optional directory to save the plot
        - title: Title for the plot and output filename
        """

        # Set up colors and style
        colors = {'red': '#FF4B4B', 'gray': '#808080', 'blue': '#4B4BFF'}
        plt.style.use('default')

        # Determine plot dimensions based on number of factors and genes
        n_lfs = len(lfs)
        max_genes = max(len(df) for df in lfs.values())
        
        fig_width = min(20, max(10, n_lfs * 2.5)) + 2  # extra space for title
        fig_height = min(30, max(6, max_genes * 0.6))  # scaled height for readability

        fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=300)
        fig.patch.set_facecolor('white')

        # Plot each latent factor
        for i, (lf_name, lr_info) in enumerate(lfs.items()):
            # Sort genes by loading ascending (so highest loadings at top)
            lr_info = lr_info.sort_values(by='loading', ascending=True)
            n_genes = len(lr_info)

            # Define vertical spacing
            spacing = fig_height / (max_genes + 2)  # avoid edges
            start_y = spacing  # bottom padding

            for j, (gene, row) in enumerate(lr_info.iterrows()):
                y_pos = start_y + j * spacing
                color = colors.get(row['color'], 'black')  # fallback to black

                ax.text(i, y_pos, gene,
                        color=color,
                        fontsize=10,
                        fontweight='bold',
                        ha='center',
                        va='center')

        # Title
        ax.set_title(title.replace('_', ' '), fontsize=14, weight='bold')

        # Customize appearance
        ax.set_xlim(-0.5, n_lfs - 0.5)
        ax.set_ylim(0, fig_height)
        ax.set_xticks(range(n_lfs))
        ax.set_xticklabels(lfs.keys(), ha='center', rotation=45, fontsize=12)
        ax.set_yticks([])

        # Remove spines
        for spine in ax.spines.values():
            spine.set_visible(False)

        ax.grid(False)

        plt.tight_layout()
        if outdir:
            os.makedirs(outdir, exist_ok=True)
            plt.savefig(os.path.join(outdir, f'{title}.png'),
                        dpi=300, bbox_inches='tight', facecolor='white')

        return fig

    
    @staticmethod
    def plot_corr_network(X, lf_dict, outdir=None, minimum=0.25):

        colors = {'red': '#FF4B4B', 'gray': '#808080', 'blue': '#4B4BFF'}

        for lf, lf_loadings in lf_dict.items():
            lf_genes = lf_loadings.index.tolist()
            color_dict = lf_loadings['color'].map(colors).to_dict()
            
            features = X[lf_genes]
            corr = features.corr().where(lambda x: abs(x) > minimum, 0)
            np.fill_diagonal(corr.values, 0)

            G = nx.from_pandas_adjacency(corr)

            for gene in G.nodes():
                G.nodes[gene]['color'] = color_dict[gene]

            fig, ax = plt.subplots(figsize=(5,5), facecolor='white')
            ax.grid(False)
            # pos = nx.spring_layout(G, seed=42, k=1, iterations=50)
            pos = nx.circular_layout(G)
            nx.draw_networkx_nodes(G, pos, 
                                 node_color=[G.nodes[node]['color'] for node in G.nodes()],
                                 node_size=600,
                                 alpha=0.4,
                                 ax=ax)
            # Draw edges with alpha based on correlation strength
            for (node1, node2, data) in G.edges(data=True):
                weight = abs(data['weight'])
                nx.draw_networkx_edges(G, pos,
                                     edgelist=[(node1, node2)],
                                     width=weight*5,
                                     alpha=min(weight, 1.0),
                                     edge_color=['green' if data['weight'] < 0 else 'pink'])
                
            nx.draw_networkx_labels(G, pos, font_size=10)
            # nx.draw_networkx_edge_labels(G, pos, font_size=10)

            plt.tight_layout()
            plt.gca().set_aspect('equal')
            plt.savefig(os.path.join(outdir, f'corr_{lf}.png'), 
                       dpi=300, bbox_inches='tight', facecolor='white')

    @staticmethod
    def plot_controlplot(scores, outdir=None, title='Control Plot'):
        """
        Plot control plot for different latent factor configurations.
        
        Parameters:
        - scores: Dictionary where keys are latent factor configurations (e.g., 'z_matrix', 'marginals', 'marginals&interactions')
                 and values are lists of performance scores (e.g., AUC values)
        - outdir: Optional directory to save the plot
        - title: Title for the plot and output filename
        """
        # Create figure with white background
        sns.set_style('whitegrid')
        fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
        fig.patch.set_facecolor('white')
        
        # Plot density for s1_random and s2_random with filled area
        for score_type, color in [('full_random', 'blue'), ('partial_random', 'green'), ('s3', 'red')]:
            if score_type not in scores or len(scores[score_type]) == 0:
                continue
            if len(set(scores[score_type])) == 1:  # If all values are the same
                ax.axvline(x=scores[score_type][0], color=color, label=f'{score_type}', linewidth=2)
            else:
                sns.kdeplot(scores[score_type], label=score_type, ax=ax, fill=True, alpha=0.3, color=color)
            
        s3_max = np.max([x for x in scores['s3'] if x is not None])
        ax.axvline(x=s3_max, color='red', linestyle='--', label=f's3 best: {s3_max:.3f}')
        
        # Customize plot appearance
        ax.set_title(title, fontsize=14, pad=15, fontweight='bold')
        ax.set_xlabel('Score', fontsize=12, labelpad=10)
        ax.set_ylabel('Density', fontsize=12, labelpad=10)
        ax.set_xlim(-0.1, 1.1)
        
        # Customize grid and spines
        ax.grid(True, linestyle='--', alpha=0.3)
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
            
        ax.legend(loc='upper left')
        plt.tight_layout()
        plt.title(title)
        
        # Save plot if outdir is provided
        if outdir:
            plt.savefig(os.path.join(outdir, f'{title}.png'), 
                       dpi=300, bbox_inches='tight', facecolor='white')
        return fig
    
    @staticmethod
    def plot_interactions(interaction_pairs, outdir=None, title='Interaction Pairs'):
        """
        Plot interaction pairs for different latent factor configurations.
        
        Parameters:
        - interaction_pairs: np array of [[marginal_lfs], [interaction_lfs]]
        - outdir: Optional directory to save the plot
        - title: Title for the plot and output filename
        """
        G = nx.Graph()
        marginal_lfs = set(interaction_pairs[0])
        interaction_lfs = set(interaction_pairs[1])
        G.add_nodes_from(marginal_lfs | interaction_lfs)
        for marginal_lf, interaction_lf in zip(interaction_pairs[0], interaction_pairs[1]):
            G.add_edge(marginal_lf, interaction_lf)

        node_colors = [
            'salmon' if node in marginal_lfs else 'gray'
            for node in G.nodes()
        ]

        fig, ax = plt.subplots(figsize=(5, 5), facecolor='white')
        pos = nx.spring_layout(G, seed=42)
        nx.draw_networkx(
            G,
            pos=pos,
            with_labels=True,
            node_size=600,
            font_size=10,
            node_color=node_colors,
            edge_color='#888',
            linewidths=1,
            ax=ax
        )

        legend_elements = [
            Patch(facecolor='salmon', edgecolor='k', label='Marginal LF'),
            Patch(facecolor='gray', edgecolor='k', label='Interacting LF')
        ]
        ax.legend(handles=legend_elements, loc='best')
        ax.set_axis_off()
        plt.tight_layout()

        if outdir:
            plt.savefig(os.path.join(outdir, f'{title}.png'), dpi=300, bbox_inches='tight', facecolor='white')
        return fig







