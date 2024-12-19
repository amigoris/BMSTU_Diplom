from flask import Flask, render_template, request, redirect, url_for
import json
import joblib
import numpy as np
import pandas as pd

from scipy.signal import welch
from scipy.stats import kurtosis, skew
from scipy.signal.windows import blackman


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'


def feature_extraction(file_path):
    try:
        sequence = ["Ошибка: Неверный файл"]
        with open(file_path, 'r') as jsonl_file:
            combined_data = []
            for line in jsonl_file:

                data_json = json.loads(line)
                combined_data.append(data_json)

            data = pd.DataFrame(combined_data)

            for i, row in enumerate(data['gyro_rad']):
                data.loc[i, ['gx', 'gy', 'gz']] = row

            for i, row in enumerate(data['accelerometer_m_s2']):
                data.loc[i, ['ax', 'ay', 'az']] = row

            data.drop(columns=["gyro_rad", "accelerometer_m_s2",
                               "accelerometer_timestamp_relative",
                               "accelerometer_clipping",
                               "gyro_integral_dt",
                               "accelerometer_integral_dt"], inplace=True)

            data.rename(columns={'timestamp': 'time'}, inplace=True)

            cols = ["ЧО Медиана", "ЧО Средняя",
                    "ЧО Std", "ЧО Max", "ЧО Min",
                    "ЧО 90 процентиль", "ЧО 75 процентиль", "ЧО 25 процентиль",
                    "ЧО Куртозис", "ЧО Ассиметрия", "ЧО Энергия", "ЧО Вариация",
                    "ЧО Количество пиков",
                    "ЧО Средняя > 10 Гц", "ЧО Std > 10 Гц", "ЧО Энергия > 10 Гц",

                    "ВО Медиана", "ВО Средняя",
                    "ВО Std", "ВО Max", "ВО Min",
                    "ВО 90 процентиль", "ВО 75 процентиль", "ВО 25 процентиль",
                    "ВО Куртозис", "ВО Ассиметрия", "ВО Энергия", "ВО Вариация",
                    "ВО Количество пиков",
                    "ВО Общая мощность", "ВО Средняя мощность"]

            channels = ["gx", "gy", "gz", "ax", "ay", "az"]
            cols = sum([[col[:3] + name + col[2:] for col in cols] for name in channels], [])

            features_df = pd.DataFrame(columns=cols)

            Fs = 195
            window_size = 256
            overlap = window_size // 2

            window = blackman(window_size)
            more10hz = round(256 / Fs * 10)

            num_windows = (len(data) - overlap) // (window_size - overlap)

            for i in range(num_windows):

                start = i * (window_size - overlap)
                end = start + window_size
                segment = data.iloc[start:end, :]

                segment = pd.DataFrame(segment, columns=["time", *channels]).sort_values(by='time')
                time_differences = (segment['time'].diff().abs()).dropna()
                if max(time_differences) > 150000: continue
                segment = segment.drop(columns=["time"]).values

                if segment.shape[0] == window_size:
                    segment_windowed = segment * window[:, None]
                    yf = np.fft.fft(segment_windowed, axis=0)
                    amplitude = np.abs(yf)

                    features = []
                    for channel in range(amplitude.shape[1]):
                        amp_half = amplitude[:overlap, channel]
                        seg = segment[:, channel]

                        features.append(np.median(amp_half))
                        features.append(np.mean(amp_half))
                        features.append(np.std(amp_half))
                        features.append(np.max(amp_half))
                        features.append(np.min(amp_half))
                        features.append(np.percentile(amp_half, 90))
                        features.append(np.percentile(amp_half, 75))
                        features.append(np.percentile(amp_half, 25))
                        features.append(kurtosis(amp_half))
                        features.append(skew(amp_half))
                        features.append(np.trapz(amp_half ** 2))

                        features.append(np.std(amp_half) / np.mean(amp_half))

                        features.append(np.sum(amp_half > np.median(amp_half) + 0.5 * np.std(amp_half)))

                        amp_half_10 = amp_half[more10hz:]
                        features.append(np.mean(amp_half_10))
                        features.append(np.std(amp_half_10))
                        features.append(np.trapz(amp_half_10 ** 2))

                        features.append(np.median(seg))
                        features.append(np.mean(seg))
                        features.append(np.std(seg))
                        features.append(np.max(seg))
                        features.append(np.min(seg))
                        features.append(np.percentile(seg, 90))
                        features.append(np.percentile(seg, 75))
                        features.append(np.percentile(seg, 25))
                        features.append(kurtosis(seg))
                        features.append(skew(seg))
                        features.append(np.trapz(seg ** 2))

                        features.append(np.std(seg) / np.mean(seg))

                        features.append(np.sum(seg > np.median(seg) + 0.5 * np.std(seg)))

                        f, Pxx = welch(seg, fs=Fs, window='hann', nperseg=window_size, noverlap=overlap)
                        features.append(np.sum(Pxx[:overlap]))
                        features.append(np.mean(Pxx[:overlap]))

                    features_df.loc[len(features_df)] = features

            features_df = features_df.to_numpy()

            power_transformer = joblib.load('power_transformer.pkl')
            features_df = power_transformer.transform(features_df)

            standard_scaler = joblib.load('standard_scaler.pkl')
            features_df = standard_scaler.transform(features_df)

            scaler = joblib.load('minmax_scaler.pkl')
            features_df = scaler.transform(features_df)

            loaded_model = joblib.load('trained_model.pkl')
            sequence = [int(i) for i in list(loaded_model.predict(features_df))]

    except Exception as e:
        print(f"Ошибка: {e}")
    return sequence


@app.route('/')
def home_page():
    return render_template('home_page.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        return redirect(request.url)

    if file and file.filename.endswith('.jsonl'):
        file_path = f"{app.config['UPLOAD_FOLDER']}/{file.filename}"
        file.save(file_path)

        sequence = feature_extraction(file_path)

        return render_template('result.html', sequence=sequence)

    return render_template('error.html')


if __name__ == '__main__':
    app.run(debug=True)
