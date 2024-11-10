import streamlit as st
import librosa
import soundfile as sf
import numpy as np
from scipy.signal import medfilt, istft
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
import pandas as pd

def reduce_noise(y, sr):
    nperseg = 2048
    noverlap = 1536

    D = librosa.stft(y, n_fft=2048, hop_length=512)
    S_full = np.abs(D)
    noise_power = np.mean(S_full[:, :int(sr*0.1)], axis=1)
    mask = S_full > noise_power[:, None]
    mask = mask.astype(float)
    mask = medfilt(mask, kernel_size=(1, 5))
    S_clean_complex = D * mask
    S_clean_complex = S_clean_complex.astype(np.complex64)

    _, y_clean = istft(S_clean_complex, fs=sr, nperseg=2048, noverlap=1536, input_onesided=True)

    if len(y_clean) > len(y):
        y_clean = y_clean[:len(y)]
    elif len(y_clean) < len(y):
        y_clean = np.pad(y_clean, (0, len(y) - len(y_clean)))

    y_clean = librosa.util.normalize(y_clean)
    return y_clean, S_clean_complex, D

def plot_spectrogram(D, sr, title):
    D_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    times = np.arange(D_db.shape[1]) * 512 / sr
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)

    fig = go.Figure(data=go.Heatmap(
        z=D_db,
        x=times,
        y=freqs,
        colorscale='Viridis',
        colorbar=dict(title='dB')
    ))

    fig.update_layout(
        title=title,
        xaxis_title='Time (s)',
        yaxis_title='Frequency (Hz)',
        yaxis_type='log',
        height=400
    )

    return fig

def prepare_pca_data(D):
    """Prepare spectrogram data for PCA analysis"""
    # Convert to magnitude spectrogram
    X = np.abs(D)

    # Reshape to (time_frames, frequencies)
    X = X.T

    # Normalize
    X = (X - X.mean()) / X.std()

    return X.astype(np.float32)

def perform_pca(X, n_components=10):
    """Perform PCA on the prepared data"""
    if X is None or X.size == 0:
        return None, None

    # Ensure we don't request more components than possible
    n_components = min(n_components, min(X.shape[0], X.shape[1]))

    try:
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X)
        return pca, X_pca
    except Exception as e:
        st.error(f"PCA computation error: {str(e)}")
        return None, None

def plot_pca_components(pca, X_pca, title_prefix=""):
    """Plot PCA analysis results"""
    if pca is None or X_pca is None:
        return

    # Plot explained variance ratio
    explained_var = pca.explained_variance_ratio_
    cumulative_var = np.cumsum(explained_var)

    fig_var = go.Figure()
    fig_var.add_trace(go.Scatter(
        y=cumulative_var,
        mode='lines+markers',
        name='Cumulative Explained Variance'
    ))
    fig_var.update_layout(
        title=f"{title_prefix} Cumulative Explained Variance Ratio",
        xaxis_title="Number of Components",
        yaxis_title="Cumulative Explained Variance Ratio",
        height=400
    )
    st.plotly_chart(fig_var, use_container_width=True)

    # Plot first two principal components
    if X_pca.shape[1] >= 2:
        fig_scatter = go.Figure()
        fig_scatter.add_trace(go.Scatter(
            x=X_pca[:, 0],
            y=X_pca[:, 1],
            mode='markers',
            marker=dict(
                size=8,
                color=np.arange(len(X_pca)),
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title='Time Frame')
            )
        ))
        fig_scatter.update_layout(
            title=f"{title_prefix} First Two Principal Components",
            xaxis_title="First Principal Component",
            yaxis_title="Second Principal Component",
            height=400
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

import librosa
import numpy as np
import plotly.graph_objects as go

def compute_and_plot_mfcc(y, sr, title):
    """Compute and create visualization for MFCC"""
    # Compute MFCC features
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

    # Create time axis
    times = np.arange(mfccs.shape[1]) * 512 / sr

    # Create frequency axis (MFCC coefficients)
    mfcc_indices = np.arange(mfccs.shape[0])

    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=mfccs,
        x=times,
        y=mfcc_indices,
        colorscale='Viridis',
        colorbar=dict(title='Magnitude')
    ))

    fig.update_layout(
        title=title,
        xaxis_title='Time (s)',
        yaxis_title='MFCC Coefficients',
        height=400
    )

    return fig

def plot_mfcc_line_plots(mfccs, sr, title):
    """Create line plots for individual MFCC coefficients"""
    times = np.arange(mfccs.shape[1]) * 512 / sr

    fig = go.Figure()

    # Plot first 5 coefficients (most significant ones)
    for i in range(5):
        fig.add_trace(go.Scatter(
            x=times,
            y=mfccs[i],
            name=f'MFCC {i+1}',
            mode='lines'
        ))

    fig.update_layout(
        title=f"{title} - First 5 Coefficients",
        xaxis_title='Time (s)',
        yaxis_title='Coefficient Value',
        height=400,
        showlegend=True
    )

    return fig

def main():
    st.set_page_config(layout="wide")
    st.title('Audio Analysis Dashboard')

    uploaded_file = st.file_uploader("Choose an audio file", type=['wav'])

    if uploaded_file is not None:
        try:
            # Load audio with lower duration if needed
            y, sr = librosa.load(uploaded_file, duration=30)  # Limit to 30 seconds
            y_clean, S_clean_complex, D_original = reduce_noise(y, sr)

            # Audio playback
            st.header('Audio Comparison')
            col1, col2 = st.columns(2)
            with col1:
                st.subheader('Original Audio')
                st.audio(uploaded_file)
            with col2:
                st.subheader('Cleaned Audio')
                cleaned_audio = librosa.util.normalize(y_clean) * 0.95
                sf.write('temp_clean.wav', cleaned_audio, sr)
                st.audio('temp_clean.wav')

            # Spectrograms
            st.header('Spectrogram Analysis')
            col3, col4 = st.columns(2)
            with col3:
                fig_orig = plot_spectrogram(D_original, sr, 'Original Spectrogram')
                st.plotly_chart(fig_orig, use_container_width=True)
            with col4:
                fig_clean = plot_spectrogram(S_clean_complex, sr, 'Cleaned Spectrogram')
                st.plotly_chart(fig_clean, use_container_width=True)

            # MFCC Analysis
            st.header('MFCC Analysis')
            col5, col6 = st.columns(2)

            with col5:
                # Original audio MFCC
                fig_mfcc_orig = compute_and_plot_mfcc(y, sr, 'Original Audio MFCC')
                st.plotly_chart(fig_mfcc_orig, use_container_width=True)

                # Line plots for original audio MFCC
                mfccs_orig = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
                fig_mfcc_lines_orig = plot_mfcc_line_plots(mfccs_orig, sr, 'Original Audio')
                st.plotly_chart(fig_mfcc_lines_orig, use_container_width=True)

            with col6:
                # Cleaned audio MFCC
                fig_mfcc_clean = compute_and_plot_mfcc(y_clean, sr, 'Cleaned Audio MFCC')
                st.plotly_chart(fig_mfcc_clean, use_container_width=True)

                # Line plots for cleaned audio MFCC
                mfccs_clean = librosa.feature.mfcc(y=y_clean, sr=sr, n_mfcc=13)
                fig_mfcc_lines_clean = plot_mfcc_line_plots(mfccs_clean, sr, 'Cleaned Audio')
                st.plotly_chart(fig_mfcc_lines_clean, use_container_width=True)

            # PCA Analysis
            st.header('PCA Analysis')

            # Prepare and perform PCA on original data
            X_orig = prepare_pca_data(D_original)
            pca_orig, X_pca_orig = perform_pca(X_orig)
            if pca_orig is not None:
                st.subheader("Original Audio PCA")
                plot_pca_components(pca_orig, X_pca_orig, "Original - ")

            # Prepare and perform PCA on cleaned data
            X_clean = prepare_pca_data(S_clean_complex)
            pca_clean, X_pca_clean = perform_pca(X_clean)
            if pca_clean is not None:
                st.subheader("Cleaned Audio PCA")
                plot_pca_components(pca_clean, X_pca_clean, "Cleaned - ")

        except Exception as e:
            st.error(f"Error processing audio: {str(e)}")
            st.write("Please try a different audio file or check if the file is corrupted.")

if __name__ == '__main__':
    main()
