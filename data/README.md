# Bengaluru housing data

The website can train from the cleaned Bengaluru housing dataset:

```
data/bengaluru_clean.csv
```

The CSV needs these columns: `location`, `area_type`, `bhk`, `total_sqft`, `bath`, `balcony`, `is_ready_to_move`, and `price_inr`. The training loader removes duplicate feature/price rows before splitting the data.

You can select another compatible CSV with `python train_model.py --data "path\to\file.csv"` or set `HPP_DATA_PATH`. Prices are in Indian rupees by default.

The cleaned CSV is not included in the public Git repository. Obtain or prepare a dataset you are allowed to use and save it as `bengaluru_clean.csv` in this directory. `Bengaluru_House_Data.csv` is the original/raw dataset and does not have the same cleaned, model-ready columns.

CSV files are ignored by Git; do not commit datasets unless you have permission to redistribute them. To try the app without a dataset, run `python train_model.py --demo` from the project root; this trains on synthetic data.
