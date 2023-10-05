import argparse
import subprocess
import torch
import pyannote.audio
import whisper
import numpy as np
import datetime
import librosa
from pyannote.audio import Audio
from pyannote.core import Segment
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import plotly.graph_objects as go
import matplotlib.cm as cm
from segment import segment_embedding


def time(secs):
        return datetime.timedelta(seconds=round(secs))


# Function to process the audio and perform speaker diarization
def speaker_diarization(audio_path, num_speakers, model_size, language='English'):

    model = whisper.load_model(model_size)
    result = model.transcribe(audio_path)
    segments = result["segments"]


    y, sr = librosa.load(audio_path, sr=None)  # sr=None preserves the native sampling rate
    # Get the duration, frame rate, and frames
    duration = librosa.get_duration(y=y, sr=sr)
    frames = len(y)  # Total samples in the audio file
    rate = sr  # Sampling rate of the audio file

    embeddings = np.zeros(shape=(len(segments), 192))
    for i, segment in enumerate(segments):
        embeddings[i] = segment_embedding(segment, audio_path, duration)

    embeddings = np.nan_to_num(embeddings)

    # Your existing code for clustering and labeling segments...
    clustering = AgglomerativeClustering(num_speakers).fit(embeddings)
    labels = clustering.labels_
    for i in range(len(segments)):
        segments[i]["speaker"] = 'SPEAKER ' + str(labels[i] + 1)

    f = open("transcript.txt", "w")
    for (i, segment) in enumerate(segments):
        if i == 0 or segments[i - 1]["speaker"] != segment["speaker"]:
            f.write("\n" + segment["speaker"] + ' ' + str(time(segment["start"])) + '\n')
        f.write(segment["text"][1:] + ' ')
    f.close()

    # Perform PCA to reduce the dimensionality of embeddings to 2D
    pca = PCA(n_components=2, random_state=42)
    embeddings_2d = pca.fit_transform(embeddings)

    # Plot the clusters
    plt.figure(figsize=(10, 8))
    for i, segment in enumerate(segments):
        speaker_id = labels[i] + 1
        x, y = embeddings_2d[i]
        plt.scatter(x, y, label=f'SPEAKER {speaker_id}')

    plt.title("Speaker Diarization Clusters (PCA Visualization)")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend()
    plt.show()

    # Perform PCA to reduce the dimensionality of embeddings to 3D
    pca = PCA(n_components=3, random_state=42)
    embeddings_3d = pca.fit_transform(embeddings)

    # Get the number of unique speakers from the labels
    num_unique_speakers = len(np.unique(labels))

    # Create a colormap for speakers, ensuring each speaker gets a unique color
    colors = cm.tab20b(np.linspace(0, 1, num_unique_speakers))

    # Prepare the data for the 3D scatter plot
    data = []
    for i, segment in enumerate(segments):
        speaker_id = labels[i] + 1
        x, y, z = embeddings_3d[i]
        color = colors[labels[i] % num_unique_speakers]  # Get the corresponding color for the speaker
        trace = go.Scatter3d(x=[x], y=[y], z=[z], mode='markers',
                             marker=dict(size=5, color=color),
                             name=f'SPEAKER {speaker_id}')
        data.append(trace)

    # Layout for the 3D scatter plot
    layout = go.Layout(
        title="Speaker Diarization Clusters (3D Visualization)",
        scene=dict(
            xaxis_title="Principal Component 1",
            yaxis_title="Principal Component 2",
            zaxis_title="Principal Component 3"
        )
    )

    # Create the figure and plot the 3D scatter plot
    fig = go.Figure(data=data, layout=layout)
    fig.show()

def main():
    parser = argparse.ArgumentParser(description='Speaker Diarization')
    parser.add_argument('--audio_path', help='Path to the audio file')
    parser.add_argument('--num_speakers', type=int, default=2, help='Number of speakers (default: 2)')
    parser.add_argument('--model_size', default='large', help='Model size (default: large)')
    args = parser.parse_args()

    speaker_diarization(args.audio_path, args.num_speakers, args.model_size)

if __name__ == "__main__":
    main()
