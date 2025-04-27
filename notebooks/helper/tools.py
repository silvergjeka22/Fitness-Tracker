from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from scipy.stats import normaltest
from scipy.signal import butter, filtfilt, lfilter
from sklearn.decomposition import PCA
import copy
from sklearn.metrics import silhouette_score


############### Outlier detection ####################
class OutlierDetection:
    def __init__(self):
        pass

    def tune_dbscan(self, df, outliers_cols, eps_range=(0.1, 2.0, 0.1), min_samples_range=(5, 20)):
        scaler = StandardScaler()
        df_scaled = df.copy()
        df_scaled[outliers_cols] = scaler.fit_transform(df[outliers_cols])
        
        best_eps = None
        best_min_samples = None
        best_score = -1  # Silhouette score

        total_steps = len(np.arange(eps_range[0], eps_range[1], eps_range[2])) * (min_samples_range[1] - min_samples_range[0] + 1)
        step = 1
        
        print(f"Starting DBSCAN tuning: {total_steps} total combinations to check...\n")
        for eps in np.arange(eps_range[0], eps_range[1], eps_range[2]):
            print(f"Checking eps = {eps:.2f}")
            for min_samples in range(min_samples_range[0], min_samples_range[1] + 1):
                print(f"  Step {step}/{total_steps}: eps={eps:.2f}, min_samples={min_samples}", end=" --> ")
                dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                labels = dbscan.fit_predict(df_scaled[outliers_cols])

                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                n_outliers = (labels == -1).sum()
                n_points = len(labels)

                if n_clusters > 1:  # for >1 clusters
                    try:
                        score = silhouette_score(df_scaled[outliers_cols][labels != -1], labels[labels != -1])
                    except:
                        score = -1
                else:
                    score = -1

                print(f"Clusters: {n_clusters}, Outliers: {n_outliers}, Silhouette Score: {score:.4f}")

                # Optional: Only accept if outliers are less than 10%
                if score > best_score and (n_outliers / n_points) < 0.1:
                    print(f"    New best! Silhouette Score {score:.4f}")
                    best_score = score
                    best_eps = eps
                    best_min_samples = min_samples

                step += 1

        print(f"\nBest DBSCAN parameters: eps = {best_eps}, min_samples = {best_min_samples}")
        print(f"Best silhouette score: {best_score:.4f}")

        return best_eps, best_min_samples, df_scaled


    def check_normal_distribution(self, data, columns):
        for col in columns:
            stat, p = normaltest(data[col])
            if p > 0.15: # 0.15 -> 85% normal distribution
                print(f'{col} is normally distributed (p-value: {p})')
            else:
                print(f'{col} is not normally distributed (p-value: {p})')

############### Feature engineering ####################

class PrincipalComponentAnalysis:

    pca = []

    def __init__(self):
        self.pca = []

    def normalize_dataset(self, data_table, columns):

        dt_norm = copy.deepcopy(data_table)
        for col in columns:
            # formula max-min normalization -> (x - mean) / (max - min)
            dt_norm[col] = (data_table[col] - data_table[col].mean()) / (
                data_table[col].max()
                - data_table[col].min()
            )
        return dt_norm

    def pca_variance(self, data_table, cols):

        # Normalize the data first.
        dt_norm = self.normalize_dataset(data_table, cols)

        # perform the PCA.
        self.pca = PCA(n_components=len(cols))
        self.pca.fit(dt_norm[cols])
        # And return the explained variances.
        return self.pca.explained_variance_ratio_ # The explained variance of each component.

    def apply_pca(self, data_table, cols, number_comp):

        # Normalize the data first.
        dt_norm = self.normalize_dataset(data_table, cols)

        # perform the PCA.
        self.pca = PCA(n_components=number_comp)
        self.pca.fit(dt_norm[cols])

        # Transform our old values.
        new_values = self.pca.transform(dt_norm[cols])

        # And add the new ones:
        for comp in range(0, number_comp):
            data_table["pca_" + str(comp + 1)] = new_values[:, comp]

        return data_table

###### Filtering functions ######
class LowPassFilter:
        def low_pass_filter(self, data_table, col, sampling_frequency, cutoff_frequency, order=5, phase_shift=True):
            nyquist = 0.5 * sampling_frequency
            cutoff_norm = cutoff_frequency / nyquist
            b, a = butter(order, cutoff_norm, btype="low")
            filtered = filtfilt(b, a, data_table[col]) if phase_shift else lfilter(b, a, data_table[col])
            data_table[col + "_lowpass"] = filtered
            return data_table

###### Fourier transform ######
class FourierTransformation:
        def find_fft_transformation(self, data, sampling_rate):
            transformation = np.fft.rfft(data)
            return transformation.real, transformation.imag

        def abstract_frequency(self, data_table, cols, window_size, sampling_rate):
            freqs = np.round(np.fft.rfftfreq(window_size) * sampling_rate, 3)

            for col in cols:
                for freq in freqs:
                    data_table[f"{col}_freq_{freq}_Hz_ws_{window_size}"] = np.nan
                data_table[f"{col}_max_freq"] = np.nan
                data_table[f"{col}_freq_weighted"] = np.nan
                data_table[f"{col}_pse"] = np.nan

            for i in range(window_size, len(data_table)):
                for col in cols:
                    segment = data_table[col].iloc[i - window_size: i + 1]
                    real_ampl, _ = self.find_fft_transformation(segment, sampling_rate)

                    for j, freq in enumerate(freqs):
                        data_table.loc[i, f"{col}_freq_{freq}_Hz_ws_{window_size}"] = real_ampl[j]

                    max_freq = freqs[np.argmax(real_ampl)]
                    weighted_freq = np.sum(freqs * real_ampl) / np.sum(real_ampl)
                    psd = real_ampl ** 2 / len(real_ampl)
                    psd_pdf = psd / np.sum(psd)
                    pse = -np.sum(np.log(psd_pdf) * psd_pdf)

                    data_table.loc[i, f"{col}_max_freq"] = max_freq
                    data_table.loc[i, f"{col}_freq_weighted"] = weighted_freq
                    data_table.loc[i, f"{col}_pse"] = pse

            return data_table
