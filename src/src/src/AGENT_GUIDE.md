# PGA-Analysis – Agent Guide

## 🧠 Purpose of This Project
This repository analyzes **PGA Tour professionals** using strokes gained and driving statistics to:
- Identify player archetypes (e.g., “High-variance bomber”, “Elite ball-striker”)
- Evaluate course fit for different playing styles
- Predict which golfers will gain or lose the most performance under the **2028 golf ball rollback**
- Perform venue-specific analyses (e.g., which players fit a given Ryder Cup course)

**This project does NOT use personal golf stats — only PGA Tour player data.**

---

## 📁 Expected Data Files (in `data/raw/`)
The agent can assume these CSV files *will* exist later:

### **1. Season-level PGA player stats (e.g., `pga_2025_stats.csv`)**
Columns typically include:
- `season`
- `player_name`
- `player_id`
- `events_played`
- `rounds_played`

Strokes gained:
- `sg_total_per_round`
- `sg_off_tee_per_round`
- `sg_approach_per_round`
- `sg_around_green_per_round`
- `sg_putting_per_round`

Driving / accuracy:
- `driving_distance_avg`
- `fairway_pct`

---

### **2. Course profiles (e.g., `course_profiles.csv`)**
Columns:
- `course_id`
- `course_name`
- `season`
- `yardage`
- `par`

Categorical course features:
- `fairway_width_category` (narrow / medium / wide)
- `rough_penalty_category` (low / medium / high)
- `green_size_category`
- `green_speed_category`
- `wind_exposure_category`

---

### **3. Player-course performance (optional)**
(e.g., `player_course_results.csv`)

Columns:
- `season`
- `player_id`
- `course_id`
- `rounds_played_at_course`
- `sg_total_per_round_at_course`
- `sg_off_tee_per_round_at_course`
- `sg_approach_per_round_at_course`
- `sg_putting_per_round_at_course`
- `scoring_avg_at_course`

---

## 🗂️ Existing Code Layout (the files the agent will modify)

### `src/config.py`
Defines base paths and filenames:
- `data/raw/`
- `data/processed/`
- CSV file locations

### `src/load_data.py`
Loads:
- season-level PGA stats
- course profiles
- player-by-course results

Agent should **not** hardcode file paths. Always use config constants.

---

### `src/players.py`
Contains the function:
- `add_player_archetypes(df)`

This:
- Computes percentiles
- Assigns a player archetype label
- Returns an enriched DataFrame

Agent can extend this with:
- new archetype criteria  
- additional percentile metrics  
- more advanced clustering  

---

### `src/rollback_model.py`
Core function:
- `simulate_rollback(df_players, yards_lost, sensitivity_off_tee)`

This:
- Reduces driving distance  
- Estimates SG Off-the-Tee loss  
- Calculates overall SG change  
- Produces a “rollback delta” column  

Agent can extend this later to:
- model approach shot changes  
- model long-iron penalty under rollback  
- create per-player distance-loss curves  

---

### `src/course_profiles.py`
Currently a placeholder.  
Agent will eventually implement:
- course modeling helpers  
- course feature normalization  
- course similarity scoring  

---

### `src/course_fit.py`
Currently empty.  
This will house:
- course fit scoring models  
- functions like:
  - `compute_course_fit(df_players, df_courses, course_id)`
  - `rank_players_for_course(course_id)`  
  - “best fits for Ryder Cup venue” logic  

---

## ⚙️ Guidelines for the Agent

### **DO:**
- Keep code organized inside `src/`
- Use `config.py` for paths and filenames
- Write docstrings for every new function
- Return **DataFrames**, not printed output
- Create small, modular functions (not giant scripts)
- Store any cleaned or derived data in `data/processed/`
- Add new analyses in:
  - `notebooks/`
  - OR new `.py` files within `src/analysis/` (if created)

---

### **DON’T:**
- Do NOT modify or overwrite files inside `data/raw/`
- Do NOT hardcode absolute paths  
- Do NOT add unrelated functionality  
- Do NOT produce HTML/JS front-end unless instructed  

---

## 📊 Core Tasks the Agent Will Perform Later

When the user uploads real data, the agent should be prepared to:

### **1. Classify PGA players**
Using:
- strokes gained
- driving distance
- fairway %
- approach metrics

### **2. Simulate ball rollback**
Given:
- yards lost per player or global
- SG sensitivity values

Output:
- players most hurt  
- players least affected  
- players who might *benefit* (accuracy-weighted types)

### **3. Compute course fit**
Based on:
- narrow vs wide fairways  
- length  
- rough severity  
- green size  
- wind exposure  

### **4. Produce rankings for venues**
Examples:
- “Rank best fits for Bethpage Black”
- “Which Americans should be picked for the next home Ryder Cup?”

### **5. Build visualizations / tables**
Placed in:
- `notebooks/`  
- OR a `dashboard.py` Streamlit app  

---

## 🏁 Summary to the Agent
This repo is an analytics lab for professional golf.  
Your job is to help with:

- strokes gained modeling  
- course fit scoring  
- ball rollback simulations  
- data cleaning and loading  
- creating analysis scripts/notebooks

Follow the structure.  
Keep code modular.  
Document everything.  

