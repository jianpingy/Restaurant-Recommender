## Overview
This repo builds a chatbot for personalized restaurant recommendations.

## Get started
Clone the repo.
```bash
git clone https://github.com/jianpingy/Restaurant-Recommender.git
cd Restaurant-Recommender
```

Install the required packages.
```bash
pip install -r requirements.txt
```

Create an ``.env`` file for ``WATSONX_AI_PROJECT_ID``.
```bash
cp .env.example .env
```

Run the gradio interface.
```bash
gradio ui.py
```