# 🎈 Blank app template

A simple Streamlit app template for you to modify!

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://blank-app-template.streamlit.app/)

### How to run it on your own machine

1. Install the requirements

   ```
   $ pip install -r requirements.txt
   ```

2. Run the app

   ```
   $ streamlit run streamlit_app.py
   ```

### Automated Jobs (GitHub Actions)

This repository uses GitHub Actions to run automated jobs for generating and revising plans. The workflows are defined in `.github/workflows/automated_planner.yml`.

For the automated jobs to run successfully, you must configure the following secrets in your GitHub repository's settings (`Settings > Secrets and variables > Actions`):

- `SUPABASE_URL`: Your Supabase project URL.
- `SUPABASE_ANON_KEY`: Your Supabase anonymous key.
- `OPENAI_API_KEY`: Your OpenAI API key.
- `OPENAI_ASSISTANT_ID`: Your OpenAI Assistant ID.

The jobs are scheduled as follows:
- **New Plan Generation**: Runs at 12:00 AM on the 1st of every month.
- **Plan Revision**: Runs at 12:00 PM every Sunday.
