# Eksperimen_SML_Sony-Alfauzan

Repository eksperimen Machine Learning untuk kelas Membangun Sistem Machine Learning.

## Dataset
**Wine Quality Dataset** - UCI Machine Learning Repository
- Red Wine: 1,599 sampel
- White Wine: 4,898 sampel
- Total: 6,497 sampel, 12 fitur

## Struktur Repository
```
Eksperimen_SML_Sony-Alfauzan/
├── .github/workflows/preprocessing.yml
├── wine_quality_raw/
│   ├── winequality-red.csv
│   └── winequality-white.csv
├── preprocessing/
│   ├── Eksperimen_Sony-Alfauzan.ipynb
│   ├── automate_Sony-Alfauzan.py
│   ├── wine_quality_preprocessing.csv
│   ├── wine_quality_train.csv
│   └── wine_quality_test.csv
└── README.md
```

## Tahapan Eksperimen
1. **Data Loading** - Memuat dataset red & white wine
2. **EDA** - Statistik deskriptif, distribusi, korelasi, outlier detection
3. **Preprocessing** - Hapus duplikat, IQR capping, encoding, binning, standarisasi, train-test split

## Otomatisasi (Skilled)
File `automate_Sony-Alfauzan.py` mengkonversi seluruh tahapan notebook menjadi pipeline otomatis.

```bash
cd preprocessing
python automate_Sony-Alfauzan.py
```

## GitHub Actions (Advanced)
Workflow otomatis berjalan setiap push/PR untuk melakukan preprocessing dan menyimpan dataset terbaru.

## Author
Sony Alfauzan
