# 📁 Best Time to Post on Social Media This Christmas by Calista Jajalla ᓚᘏᗢ

## Table of Contents

1. [Project Folder Structure](#1-project-folder-structure)  
2. [About the Dataset](#2-about-the-dataset)  
3. [Setup Python Virtual Environment (venv)](#3-setup-python-virtual-environment-venv)  
4. [Data Generation, ML Training & Visualization Notebook](#4-data-generation-ml-training--visualization-notebook)  
5. [Docker Compose Setup](#docker-compose-setup)
6. [References](#references)

---

## 1. Project Folder Structure

Organize your project directory to keep code, data, and outputs tidy:

```bash
ph-social-posting-times/
├── README.md
├── data/
│   ├── ml/                                       # ML training script
│   │   ├── platform_encoder_christmas.joblib
│   │   └── posting_time_model_christmas.joblib                       
│   └── posting_times_christmas_ph.csv            # Synthetic engagement data  
├── notebook/
│   └── posting_time_optimizer.ipynb              # All steps combined notebook  
├── output/
│   └── interactive_christmas_blog.html           # Dashboard HTML output
├── docker-compose.yml                            # Docker Compose configuration  
├── requirements.txt                
├── .gitignore
├── scripts/                                      # (optional for extra scripts)  
└── venv/                                         # Python virtual environment  
```

---

## 2. About the Dataset

A generated synthetic dataset simulates hourly social media engagement across five platforms (TikTok, Facebook, Instagram, YouTube, Twitter) over December 15, 2025, to January 1, 2026.

Features: 
- Engagement metrics: likes, shares, comments, combined into an engagement score.
- Hourly patterns with platform-specific peak times.
- Holiday effects on Christmas Eve, Christmas, New Year's Eve, and New Year's Day.
- Flags for weekends and holidays to reflect increased or decreased activity.
- Hourly and daily seasonality (Commute, work and school hours, etc.)
- Platform-specific peak hours

This synthetic data is designed for training models and exploring optimal posting schedules during the festive season.

## 3. Setup Python Virtual Environment (venv)

Isolate dependencies by creating and activating a Python virtual environment, then install required packages:

```bash
python3 -m venv venv  
source venv/bin/activate       # Windows: .\venv\Scripts\activate  
pip install --upgrade pip  
pip install requirements.txt
```

---

## 4. Data Generation, ML Training & Visualization Notebook

All key steps are integrated in one Jupyter notebook (`notebooks/ml/posting_time_optimizer.ipynb`) for easier experimentation and visualization.

### Section 1: Data Generation (Snippets and explanation):

1. Define holiday flags for special dates:

```python
special_dates = {
    "2025-12-24": "christmas_eve",
    "2025-12-25": "christmas",
    "2025-12-31": "new_year_eve",
    "2026-01-01": "new_year"
}
```

2. Use smooth Gaussian functions to model platform-specific engagement peaks throughout the day:

```python
def smooth_peak(hour, center, width, height):
    return height * np.exp(-((hour - center) ** 2) / (2 * width ** 2))
```

3. Loop through each date, platform, and hour to calculate engagement base, adjust for weekend/holiday effects, add noise for realism, and generate engagement from Poisson distributions:

```bash
for date in pd.date_range(christmas_start, christmas_end):
    is_weekend = date.weekday() >= 5
    holiday = special_dates.get(date.strftime("%Y-%m-%d"), "")
    
    for platform in platforms:
        peaks = platform_peak_definitions[platform]  # e.g. [(7,6), (12.5,5.5), ...]
        platform_variation = np.random.uniform(0.85, 1.2)
        
        for hour in range(24):
            base_engagement = 5.0
            
            for center, height in peaks:
                width = np.random.uniform(0.7, 1.3)
                hour_offset = np.random.uniform(-0.5, 0.5)
                peak_height = height * platform_variation * np.random.uniform(0.85, 1.15)
                base_engagement += smooth_peak(hour, center + hour_offset, width, peak_height)
            
            if is_weekend:
                base_engagement *= np.random.uniform(1.05, 1.2)
            
            if holiday == "christmas_eve":
                base_engagement += smooth_peak(hour, 23, 1.2, 1.5 * platform_variation * 10)
            # ... other holiday adjustments here
            
            base_engagement += np.random.normal(0, 0.5)
            base_engagement = max(5.0, base_engagement)
            
            likes = np.random.poisson(base_engagement * 2.0)
            shares = np.random.poisson(base_engagement * 0.65)
            comments = np.random.poisson(base_engagement * 0.45)
            
            rows.append({
                "platform": platform,
                "date": date.strftime("%Y-%m-%d"),
                "hour": hour,
                "likes": likes,
                "shares": shares,
                "comments": comments,
                "engagement_score": likes + shares + comments,
                # ...other fields
            })
```

4. Save the resulting DataFrame to CSV for downstream use:

```python
df = pd.DataFrame(rows)
df.to_csv("data/posting_times_christmas_ph.csv", index=False)
```

### (Optional) Section 2: Machine Learning Model Training (Snippets and explanation):

1. Load the generated CSV dataset and create features:

```python
df = pd.read_csv("data/posting_times_christmas_ph.csv")

df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
df["week_of_year"] = pd.to_datetime(df["date"]).dt.isocalendar().week.astype(int)

encoder = OneHotEncoder(sparse_output=False)
platform_encoded = encoder.fit_transform(df[["platform"]])
platform_df = pd.DataFrame(platform_encoded, columns=encoder.get_feature_names_out(["platform"]))

features = pd.concat([
    platform_df,
    df[["hour_sin", "hour_cos", "day_num", "is_weekend", "is_holiday", "is_christmas_season", "week_of_year"]]
], axis=1)
target = df["engagement_score"]
```

2. Split data and train a Random Forest regressor:

```python
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

model = RandomForestRegressor(
    n_estimators=300, max_depth=14, min_samples_leaf=5,
    random_state=42, n_jobs=-1
)
model.fit(X_train, y_train)

print(f"Test MSE: {((model.predict(X_test) - y_test) ** 2).mean():.2f}")
```

3. Save model and encoder for reuse:

```python
joblib.dump(model, "notebooks/ml/random_forest_model.joblib")
joblib.dump(encoder, "notebooks/ml/encoder.joblib")
```

### Section 3: Visualization and Interactive Dashboard (Snippets and explanation):

1. Use Plotly to build heatmaps of average engagement per platform by hour, split into morning and evening:

```python
import plotly.graph_objects as go

fig_morning = go.Figure(go.Heatmap(
    z=pivot_morning.values,
    x=pivot_morning.columns,
    y=pivot_morning.index,
    colorscale=['#F6E7D8','#B11226'],
    zmin=0,
    zmax=70,
    hovertemplate='Platform: %{y}<br>Hour: %{x}<br>Engagement: %{z:.1f}<extra></extra>'
))
fig_morning.update_layout(title='Morning Engagement')
fig_morning.show()
```

2. Create interactive widgets to select date and update top 3 posting hours and line charts:

```python
from ipywidgets import Dropdown, VBox, Output
from IPython.display import display, HTML

dropdown = Dropdown(options=list(day_data.keys()), description="Select Date:")
out_table = Output()
out_line = Output()

def update_dashboard(change):
    date = change["new"]
    table = day_data[date]
    with out_table:
        out_table.clear_output()
        html = "<table><tr><th>Platform</th><th>Top 3 Hours</th><th>Average</th></tr>"
        for row in table:
            top_str = ", ".join([f"{x['hour_label']} ({x['predicted_score']:.1f})" for x in row["top3"]])
            html += f"<tr><td>{row['platform']}</td><td>{top_str}</td><td>{row['average']:.1f}</td></tr>"
        html += "</table>"
        display(HTML(html))
    # update line charts similarly...

dropdown.observe(update_dashboard, names="value")
display(VBox([dropdown, out_table, out_line]))
update_dashboard({"new": list(day_data.keys())[0]})
```

3. Export full interactive dashboard as self-contained HTML:

```
with open("output/interactive_christmas_blog.html", "w", encoding="utf-8") as f:
    f.write(html_template)
```

4. Open this file in any modern browser to explore best posting hours by platform and date with rich interactivity.

### Link: 

5. Docker Compose Setup

```yaml
For easier deployment, create a docker-compose.yml to run Jupyter Lab with your environment:

version: '3.8'
services:
  jupyter:
    image: jupyter/datascience-notebook:latest
    ports:
      - "8888:8888"
    volumes:
      - .:/home/jovyan/work
    environment:
      - JUPYTER_ENABLE_LAB=yes
    command: start-notebook.sh --NotebookApp.token=''
```

To launch:

```bash
docker-compose up -d
```

Visit http://localhost:8888/lab in your browser to start working in Jupyter with your project files.


---

## Stopping the Application

Stop all services with:

```bash
docker-compose down
```

This command shuts down the containers and cleans up the network, freeing system resources after you’re done using the dashboard and backend services.

---
## References

- Altrue PH – Best Times to Post on Social Media in the Philippines (2025)  
  https://altrue.ph/articles-and-news/best-times-to-post-on-social-media-in-the-philippines-2025/

- Ghosh et al., *Understanding Engagement Dynamics on Social Platforms*  
  https://arxiv.org/abs/1901.00076

- Philippine Christmas season behavior patterns (industry heuristics)
