import numpy as np
import itertools

from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from scipy.cluster import hierarchy as sch

import plotly.figure_factory as ff
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

from typing import Callable, List, Union
from textwrap import wrap

def wrap_topic_idx(
        topic_model,
        top_n: int=None,
        topic_idx: List[int]=None
    ):

    topic_weights = topic_model.get_topic_weights()

    if top_n is None and topic_idx is None:
        top_n = 5
        topic_idx = np.argsort(topic_weights)[:-(top_n + 1):-1]
    elif top_n is not None:
        assert (top_n > 0) and (topic_idx is None)
        topic_idx = np.argsort(topic_weights)[:-(top_n + 1):-1]

    return topic_idx


def visualize_topic(topic_model,
                    top_n: int=None,
                    topic_idx: List[int]=None,
                    n_label_words=5,
                    title: str = "Topic Overview",
                    width: int = 250,
                    height: int = 250,
                    topic_labels: List[str]=None,
                ):

    topic_idx = wrap_topic_idx(topic_model, top_n, topic_idx)

    top_words = topic_model.top_words
    beta = topic_model.get_beta()

    if topic_labels is None:
        subplot_titles = [f"Topic {i}" for i in topic_idx][:top_n]
    else:
        assert len(topic_labels) == topic_model.topic_embeddings.shape[0], "Number of provided topic labels differs from the true number of topics."
        subplot_titles = ['<br>'.join(wrap(topic_labels[i], width=24)) for i in topic_idx][:top_n]

    columns = 4
    rows = int(np.ceil(len(topic_idx) / columns))

    colors = itertools.cycle(px.colors.qualitative.Alphabet)

    fig = make_subplots(rows=rows,
                        cols=columns,
                        shared_xaxes=False,
                        horizontal_spacing=.1,
                        subplot_titles=subplot_titles,
                    )

    row = 1
    column = 1
    for i in topic_idx:
        words = top_words[i].split()[:n_label_words][::-1]
        scores = np.sort(beta[i])[:-(n_label_words + 1):-1][::-1]

        fig.add_trace(
                go.Bar(x=scores,
                    y=words,
                    orientation='h',
                    marker_color=next(colors)
                ),
                row=row,
                col=column
            )

        if column == columns:
            column = 1
            row += 1
        else:
            column += 1

    title_lines = max(len(wrap(topic_labels[i], width=24)) for i in topic_idx) if topic_labels is not None else 1
    buffer_space = title_lines * 20 # approx. height of title in pixels

    fig.update_layout(
        template="plotly_white",
        showlegend=False,
        title={
            'text': f"{title}",
            'y': .95,
            'x': .5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(
                size=22,
                color="Black")
        } if top_n is not None else None,
        margin=dict(t=100 + buffer_space), # increase top margin to fit title
        width=width * 4,
        height=height * rows if rows > 1 else height * 1.3,
        hoverlabel=dict(
            bgcolor="white",
            font_size=16,
            font_family="Rockwell"
        ),
    )

    fig.update_xaxes(showgrid=True)
    fig.update_yaxes(showgrid=True)

    return fig


def visualize_activity(topic_model,
                       topic_activity: np.ndarray,
                       time_slices: Union[np.ndarray, List],
                       top_n: int=None,
                       topic_idx: List[int]=None,
                       n_label_words:int=5,
                       title: str="Topic Activity over Time",
                       width: int=1000,
                       height: int=600,
                       topic_labels: List[str]=None,
                    ):

    topic_idx = wrap_topic_idx(topic_model, top_n, topic_idx)

    colors = itertools.cycle(px.colors.qualitative.Alphabet)

    fig = go.Figure()
    topic_top_words = topic_model.top_words

    if topic_labels is None:
        topic_labels = []
        for i, words in enumerate(topic_top_words):
            topic_labels.append(f"{i}_{'_'.join(words.split()[:n_label_words])}")
    else:
        assert len(topic_labels) == topic_model.topic_embeddings.shape[0], "Number of provided topic labels differs from the true number of topics."

    labels = [str(x) for x in np.unique(time_slices)]

    for i, k in enumerate(topic_idx):

        fig.add_trace(go.Scatter(
            x=labels,
            y=topic_activity[k].tolist(),
            mode='lines',
            marker_color=next(colors),
            hoverinfo="text",
            name=topic_labels[k],
            hovertext=topic_labels[k])
        )

    # Styling of the visualization
    fig.update_xaxes(showgrid=True)
    fig.update_yaxes(showgrid=True)
    fig.update_xaxes(tickmode='linear', tick0=labels[0], dtick=1, tickformat='.0f')
    # fig.update_xaxes(type='category') # treat x-axis as categorical to avoid gaps in time series
    fig.update_layout(
        yaxis_title="Topic Weight",
        title={
            'text': f"{title}",
            'y': .95,
            'x': 0.40,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(
                size=22,
                color="Black")
        } if title is not None else None,
        template="simple_white",
        width=width,
        height=height,
        hoverlabel=dict(
            bgcolor="white",
            font_size=16,
            font_family="Rockwell"
        )
    )

    return fig


def visualize_topic_weights(topic_model,
                            top_n: int=50,
                            topic_idx: List[int]=None,
                            n_label_words: int=5,
                            title: str="Topic Weights",
                            width: int=1000,
                            height: int=1000,
                            _sort: bool=True,
                            topic_labels: List[str]=None,
                        ):

    topic_weights = topic_model.get_topic_weights()
    topic_idx = wrap_topic_idx(topic_model, top_n, topic_idx)

    labels = []
    vals = []
    topic_top_words = topic_model.top_words
    
    for i in topic_idx:
        words = topic_top_words[i]
        labels.append(topic_labels[i] if topic_labels is not None else f"{i}_{'_'.join(words.split()[:n_label_words])}")
        vals.append(topic_weights[i])

    if _sort:
        sorted_idx = np.argsort(vals)
        labels = np.asarray(labels)[sorted_idx].tolist()
        vals = np.asarray(vals)[sorted_idx].tolist()

    # Create Figure
    fig = go.Figure(go.Bar(
        x=vals,
        y=labels,
        marker=dict(
            color='#C8D2D7',
            line=dict(
                color='#6E8484',
                width=1),
        ),
        orientation='h')
    )

    fig.update_layout(
        xaxis_title="Weight",
        title={
            'text': f"{title}",
            'y': .95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(
                size=22,
                color="Black")
        } if title is not None else None,
        template="simple_white",
        width=width,
        height=height,
        hoverlabel=dict(
            bgcolor="white",
            font_size=16,
            font_family="Rockwell"
        ),
    )

    return fig


def visualize_hierarchy(topic_model,
                        title: str = "Topic Hierarchy",
                        orientation: str = "left",
                        width: int = 1000,
                        height: int = 1000,
                        linkage_function: Callable = None,
                        distance_function: Callable = None,
                        n_label_words: int = 5,
                        color_threshold: int = None,
                        topic_labels: List[str] = None, # labels for the topics
                    ):

    topic_embeddings = topic_model.topic_embeddings

    if distance_function is None:
        # distance_function = lambda x: 1 - cosine_similarity(x)
        distance_function = euclidean_distances

    if linkage_function is None:
        linkage_function = lambda x: sch.linkage(x, 'ward', optimal_ordering=True)

    topic_top_words = topic_model.top_words
    
    if topic_labels is None:
        topic_labels = []
        for i, words in enumerate(topic_top_words):
            topic_labels.append(f"{i}_{'_'.join(words.split()[:n_label_words])}")
    else:
        assert len(topic_labels) == topic_embeddings.shape[0], "Labels length must match the number of topics."
        
    fig = ff.create_dendrogram(
        topic_embeddings,
        orientation=orientation,
        labels=topic_labels,
        distfun=distance_function,
        linkagefun=linkage_function,
        color_threshold=color_threshold,
    )

    fig.update_layout({
        'title': {
            'text': f"{title}",
            'y': .95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(
                size=22,
                color="Black")
        } if title is not None else None,
        'width': width, 
        'height': height
        })

    return fig
