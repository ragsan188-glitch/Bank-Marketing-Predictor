# TermWise — Clean design update

This folder contains the updated Streamlit interface. It removes decorative emojis/icons, keeps the dark/light theme toggle, leaves the model explanation visible on the page, preserves repeated predictions, and uses the provided LinkedIn and email links.

## Run locally
1. Place your fitted `bank_marketing_knn.pkl` in the same folder as `app.py`.
2. Open Terminal in this folder.
3. Run:
   ```bash
   python3 -m pip install -r requirements.txt
   python3 -m streamlit run app.py
   ```

The model file is not included; use the same fitted pipeline that worked in your current website.


The Contact section includes the approved visitor feedback message. Your trained model file is not included; copy your existing `bank_marketing_knn.pkl` into this folder.
