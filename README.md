# QuoteGuard 🛡️

QuoteGuard is a lightweight risk-tier tool for insurance quote analysis.

## What it does
- Upload a CSV of quote data
- Select the premium/quote column
- Computes percentile thresholds (P60 / P85) or uses manual thresholds
- Assigns risk tiers: Low / Medium / High
- Visualizes distribution + tier counts
- Exports a tiered CSV

## Tech
- Streamlit
- Pandas / NumPy
- Matplotlib

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
