"""
automate_Sony-Alfauzan.py
=========================
Script otomatisasi preprocessing dataset Wine Quality.
Mengkonversi seluruh langkah preprocessing dari notebook eksperimen
menjadi pipeline otomatis yang dapat dijalankan secara independen.

Author: Sony Alfauzan
Dataset: Wine Quality (UCI ML Repository)
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import os
import sys
import logging

# Konfigurasi logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def load_data(raw_data_path: str) -> pd.DataFrame:
    """
    Memuat dataset Wine Quality dari folder raw data.
    Menggabungkan red wine dan white wine menjadi satu DataFrame.

    Parameters
    ----------
    raw_data_path : str
        Path ke folder berisi file CSV raw dataset.

    Returns
    -------
    pd.DataFrame
        DataFrame gabungan red wine dan white wine.
    """
    logger.info("Memuat dataset Wine Quality...")

    red_path = os.path.join(raw_data_path, 'winequality-red.csv')
    white_path = os.path.join(raw_data_path, 'winequality-white.csv')

    if not os.path.exists(red_path):
        raise FileNotFoundError(f"File tidak ditemukan: {red_path}")
    if not os.path.exists(white_path):
        raise FileNotFoundError(f"File tidak ditemukan: {white_path}")

    df_red = pd.read_csv(red_path, sep=';')
    df_white = pd.read_csv(white_path, sep=';')

    # Menambahkan kolom wine_type
    df_red['wine_type'] = 'red'
    df_white['wine_type'] = 'white'

    df = pd.concat([df_red, df_white], axis=0, ignore_index=True)

    logger.info(f"Dataset red wine: {df_red.shape[0]} sampel")
    logger.info(f"Dataset white wine: {df_white.shape[0]} sampel")
    logger.info(f"Dataset gabungan: {df.shape[0]} sampel, {df.shape[1]} kolom")

    return df


def perform_eda(df: pd.DataFrame) -> dict:
    """
    Melakukan Exploratory Data Analysis dan mengembalikan ringkasan statistik.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame yang akan dianalisis.

    Returns
    -------
    dict
        Dictionary berisi hasil analisis EDA.
    """
    logger.info("Melakukan Exploratory Data Analysis...")

    eda_results = {}

    # Statistik deskriptif
    eda_results['shape'] = df.shape
    eda_results['dtypes'] = df.dtypes.to_dict()
    eda_results['describe'] = df.describe().to_dict()

    # Missing values
    missing = df.isnull().sum()
    eda_results['missing_values'] = missing.to_dict()
    eda_results['total_missing'] = int(missing.sum())
    logger.info(f"Total missing values: {eda_results['total_missing']}")

    # Duplikat
    duplicates = int(df.duplicated().sum())
    eda_results['duplicates'] = duplicates
    logger.info(f"Jumlah baris duplikat: {duplicates}")

    # Distribusi target
    quality_dist = df['quality'].value_counts().sort_index().to_dict()
    eda_results['quality_distribution'] = quality_dist
    logger.info(f"Distribusi quality: {quality_dist}")

    # Outlier per fitur (IQR method)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    outlier_counts = {}
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        count = int(((df[col] < lower) | (df[col] > upper)).sum())
        outlier_counts[col] = count
    eda_results['outlier_counts'] = outlier_counts
    logger.info(f"Fitur dengan outlier terbanyak: {max(outlier_counts, key=outlier_counts.get)}")

    # Korelasi dengan target
    corr_with_quality = df[numeric_cols].corr()['quality'].drop('quality').sort_values(ascending=False)
    eda_results['correlation_with_quality'] = corr_with_quality.to_dict()
    logger.info(f"Fitur paling berkorelasi dengan quality: {corr_with_quality.index[0]} ({corr_with_quality.iloc[0]:.3f})")

    logger.info("EDA selesai.")
    return eda_results


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Menghapus baris duplikat dari DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame input.

    Returns
    -------
    pd.DataFrame
        DataFrame tanpa duplikat.
    """
    before = len(df)
    df_clean = df.drop_duplicates()
    after = len(df_clean)
    logger.info(f"Menghapus duplikat: {before} -> {after} ({before - after} baris dihapus)")
    return df_clean


def handle_outliers(df: pd.DataFrame, exclude_cols: list = None) -> pd.DataFrame:
    """
    Menangani outlier menggunakan IQR Capping (Winsorization).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame input.
    exclude_cols : list, optional
        Kolom yang tidak di-cap. Default: ['quality'].

    Returns
    -------
    pd.DataFrame
        DataFrame dengan outlier yang sudah di-cap.
    """
    if exclude_cols is None:
        exclude_cols = ['quality']

    df_capped = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [col for col in numeric_cols if col not in exclude_cols]

    logger.info("Menangani outlier dengan IQR Capping...")

    for col in feature_cols:
        Q1 = df_capped[col].quantile(0.25)
        Q3 = df_capped[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        outlier_count = int(((df_capped[col] < lower) | (df_capped[col] > upper)).sum())
        if outlier_count > 0:
            df_capped[col] = df_capped[col].clip(lower=lower, upper=upper)
            logger.info(f"  {col}: {outlier_count} outlier di-cap")

    return df_capped


def encode_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """
    Melakukan encoding pada kolom kategorikal.
    - wine_type: Label Encoding (red=0, white=1)

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame input.

    Returns
    -------
    pd.DataFrame
        DataFrame dengan kolom kategorikal ter-encode.
    """
    df_encoded = df.copy()

    if 'wine_type' in df_encoded.columns:
        le = LabelEncoder()
        df_encoded['wine_type_encoded'] = le.fit_transform(df_encoded['wine_type'])
        df_encoded = df_encoded.drop('wine_type', axis=1)
        logger.info(f"Label Encoding wine_type: red={le.transform(['red'])[0]}, white={le.transform(['white'])[0]}")

    return df_encoded


def categorize_quality(df: pd.DataFrame) -> pd.DataFrame:
    """
    Membuat kategori kualitas wine untuk klasifikasi.
    - 0 (Low): quality 3-4
    - 1 (Medium): quality 5-6
    - 2 (High): quality 7-9

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame dengan kolom 'quality'.

    Returns
    -------
    pd.DataFrame
        DataFrame dengan kolom 'quality_category' baru.
    """
    df_cat = df.copy()

    def _categorize(quality):
        if quality <= 4:
            return 0  # Low
        elif quality <= 6:
            return 1  # Medium
        else:
            return 2  # High

    df_cat['quality_category'] = df_cat['quality'].apply(_categorize)
    logger.info("Binning quality ke 3 kategori: Low(0), Medium(1), High(2)")
    logger.info(f"Distribusi kategori:\n{df_cat['quality_category'].value_counts().sort_index().to_string()}")

    return df_cat


def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple:
    """
    Melakukan standarisasi fitur menggunakan StandardScaler.
    Scaler di-fit hanya pada X_train untuk mencegah data leakage.

    Parameters
    ----------
    X_train : pd.DataFrame
        DataFrame fitur training.
    X_test : pd.DataFrame
        DataFrame fitur testing.

    Returns
    -------
    tuple
        (X_train_scaled, X_test_scaled, scaler)
    """
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )
    logger.info("Standarisasi fitur selesai: fit pada X_train, transform pada X_test")
    return X_train_scaled, X_test_scaled, scaler


def split_data(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42) -> tuple:
    """
    Membagi data menjadi training dan test set dengan stratified split.

    Parameters
    ----------
    X : pd.DataFrame
        DataFrame fitur.
    y : pd.Series
        Series target.
    test_size : float
        Proporsi test set. Default: 0.2.
    random_state : int
        Random seed. Default: 42.

    Returns
    -------
    tuple
        (X_train, X_test, y_train, y_test)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    logger.info(f"Split data: train={X_train.shape[0]} ({(1-test_size)*100:.0f}%), test={X_test.shape[0]} ({test_size*100:.0f}%)")
    return X_train, X_test, y_train, y_test


def preprocess_data(raw_data_path: str, output_dir: str = None) -> tuple:
    """
    Pipeline preprocessing lengkap: load -> clean -> encode -> categorize -> split -> scale.
    Urutan split sebelum scale mencegah data leakage.

    Parameters
    ----------
    raw_data_path : str
        Path ke folder raw data.
    output_dir : str, optional
        Path untuk menyimpan output. Default: direktori saat ini.

    Returns
    -------
    tuple
        (X_train_scaled, X_test_scaled, y_train, y_test, scaler)
    """
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(__file__))

    logger.info("=" * 60)
    logger.info("MEMULAI PIPELINE PREPROCESSING")
    logger.info("=" * 60)

    # Step 1: Load data
    df = load_data(raw_data_path)

    # Step 2: EDA (ringkasan)
    eda_results = perform_eda(df)

    # Step 3: Remove duplicates
    df = remove_duplicates(df)

    # Step 4: Handle outliers (dihitung dari seluruh data sebelum split)
    df = handle_outliers(df, exclude_cols=['quality'])

    # Step 5: Encode categorical
    df = encode_categorical(df)

    # Step 6: Categorize quality
    df = categorize_quality(df)

    # Step 7: Reset index
    df = df.reset_index(drop=True)

    # Step 8: Separate features and target
    X = df.drop(['quality', 'quality_category'], axis=1)
    y = df['quality_category']

    # Step 9: Split data SEBELUM scaling (mencegah data leakage)
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Step 10: Scale features - fit hanya pada X_train, transform keduanya
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    # Step 11: Save preprocessed data
    df_full = pd.concat([X_train_scaled, y_train], axis=1)
    df_test_full = pd.concat([X_test_scaled, y_test], axis=1)
    df_final = pd.concat([df_full, df_test_full], axis=0).reset_index(drop=True)

    output_full = os.path.join(output_dir, 'wine_quality_preprocessing.csv')
    df_final.to_csv(output_full, index=False)
    logger.info(f"Dataset preprocessing disimpan ke: {output_full}")

    train_data = pd.concat([X_train_scaled.reset_index(drop=True), y_train.reset_index(drop=True)], axis=1)
    test_data = pd.concat([X_test_scaled.reset_index(drop=True), y_test.reset_index(drop=True)], axis=1)

    train_path = os.path.join(output_dir, 'wine_quality_train.csv')
    test_path = os.path.join(output_dir, 'wine_quality_test.csv')
    train_data.to_csv(train_path, index=False)
    test_data.to_csv(test_path, index=False)

    logger.info(f"Training data disimpan: {train_path} ({train_data.shape})")
    logger.info(f"Test data disimpan: {test_path} ({test_data.shape})")

    logger.info("=" * 60)
    logger.info("PREPROCESSING SELESAI")
    logger.info("=" * 60)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


if __name__ == "__main__":
    # Menentukan path relatif ke raw data
    script_dir = os.path.dirname(os.path.abspath(__file__))
    raw_path = os.path.join(script_dir, '..', 'wine_quality_raw')
    output_path = script_dir

    # Menjalankan pipeline preprocessing
    X_train, X_test, y_train, y_test, scaler = preprocess_data(raw_path, output_path)

    print("\n" + "=" * 60)
    print("RINGKASAN HASIL PREPROCESSING")
    print("=" * 60)
    print(f"Training set  : {X_train.shape[0]} sampel, {X_train.shape[1]} fitur")
    print(f"Test set      : {X_test.shape[0]} sampel, {X_test.shape[1]} fitur")
    print(f"Target classes: {sorted(y_train.unique().tolist())}")
    print(f"\nDistribusi target (train):")
    print(y_train.value_counts().sort_index().to_string())
    print(f"\nDistribusi target (test):")
    print(y_test.value_counts().sort_index().to_string())
    print("\nData siap digunakan untuk pelatihan model!")
