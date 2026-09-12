# 🏀 NBA MVP Predictive Exploratory Data Analysis (2023–24)

**Author:** Isar Joshi  
**Course:** DTSC 2301: Data Science Principles  

---

## 1. Problem Definition

* **Specific Research Question:** How effectively can regular season player performance metrics and team impact statistics predict candidates for the NBA Most Valuable Player (MVP) award under modern eligibility criteria?
* **Context & Background:** The NBA Most Valuable Player (MVP) award is chosen annually by a panel of sportswriters and broadcasters. Unlike awards determined purely by stat leadership (such as the scoring title), MVP voting balances individual efficiency (e.g., scoring, rebounding, playmaking) with collective team success (win total, net plus/minus) and subjective voter narrative. In the 2023–24 season, the NBA introduced a new rule under the Collective Bargaining Agreement requiring players to appear in at least 65 games to remain eligible for major postseason awards.
* **Relevance & Audience:** Understanding these dynamics reveals how sports media and analysts value quantitative statistical thresholds relative to team success. These findings care to sports analytics departments, media analysts, sports journalists, and predictive modeling developers evaluating award futures markets.

---

## 2. Data Description

### Key Variables (Conceptualized & Operationalized)

| Variable | Conceptualization | Operationalization |
| :--- | :--- | :--- |
| **`PTS_PG`** | Raw scoring volume | Total regular-season points divided by games played ($\text{PTS} / \text{GP}$) |
| **`AST_PG`** | Playmaking contribution | Total assists divided by games played ($\text{AST} / \text{GP}$) |
| **`REB_PG`** | Glass/possession control | Total rebounds divided by games played ($\text{REB} / \text{GP}$) |
| **`PLUS_MINUS`** | Net team impact while on floor | Total cumulative point differential during player's minutes |
| **`FG_PCT`** | Shooting efficiency | Field goals made divided by total field goal attempts ($\text{FGM} / \text{FGA}$) |
| **`GP`** | Availability | Total regular-season appearances |

* **Data Source & Attribution:** Extracted directly via Python using the `nba_api` library querying official statistical endpoints from `stats.nba.com`.
* **Unit of Analysis & Features:** Each row represents an individual NBA player's aggregated statistics for the 2023–24 regular season. Key features include player identifiers (`PLAYER_NAME`, `TEAM_ABBREVIATION`), games/minutes played (`GP`, `MIN`), counting statistics (`PTS`, `REB`, `AST`, `STL`, `BLK`), shooting efficiency (`FG_PCT`), and cumulative team impact (`PLUS_MINUS`).
* **Dataset Size & Assumptions:** The raw API pull contains 572 player records. After applying the official 65-game eligibility filter, the dataset consists of qualified candidate records. We assume the API data correctly reflects official NBA box score tracking.

---

## 3. Data Cleaning and Preparation

### Pandas Cleaning Script

```python
import pandas as pd
from nba_api.stats.endpoints import leaguedashplayerstats

# 1. Fetch live 2023-24 regular season stats via NBA API
api_call = leaguedashplayerstats.LeagueDashPlayerStats(
    season='2023-24',
    season_type_all_star='Regular Season'
)
df_raw = api_call.get_data_frames()[0]

# 2. Filter essential features
cols_to_keep = [
    'PLAYER_NAME', 'TEAM_ABBREVIATION', 'GP', 'MIN', 
    'PTS', 'REB', 'AST', 'STL', 'BLK', 'FG_PCT', 'PLUS_MINUS'
]
df_clean = df_raw[cols_to_keep].copy()

# 3. Apply 65-Game Award Eligibility Requirement
df_clean = df_clean[df_clean['GP'] >= 65].reset_index(drop=True)

# 4. Feature Engineering: Compute per-game rate statistics
df_clean['PTS_PG'] = round(df_clean['PTS'] / df_clean['GP'], 1)
df_clean['REB_PG'] = round(df_clean['REB'] / df_clean['GP'], 1)
df_clean['AST_PG'] = round(df_clean['AST'] / df_clean['GP'], 1)

# 5. Handle missing values
df_clean.fillna(0, inplace=True)
