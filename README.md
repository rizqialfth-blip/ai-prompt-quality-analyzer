# AI Prompt Quality Analyzer

A Python data analysis project that scores and visualizes the quality of ChatGPT prompts across different use cases.

## Why This Project Exists

Built to demonstrate real-world skills in AI data annotation and quality evaluation — the same judgment used in RLHF (Reinforcement Learning from Human Feedback) work. Instead of manually evaluating prompts, this project automates and visualizes that process at scale.

## Dataset

**Awesome ChatGPT Prompts**
- Source: Kaggle → search `awesome chatgpt prompts`
- URL: `kaggle.com/datasets/fka/awesome-chatgpt-prompts`
- Columns: `act` (use case/role), `prompt` (the prompt text)

## Project Structure

```
ai-response-quality-analyzer/
├── 01_intro_data_cleaning.py           # Load, explore, clean raw data
├── 02_intermediate_quality_scoring.py  # Score and categorize prompt quality
├── 03_visualization_dashboard.py       # Build visual analytics dashboard
├── requirements.txt
└── README.md
```

## How to Run

**Step 1: Install dependencies**
```bash
pip install pandas numpy matplotlib seaborn
```

**Step 2: Download dataset**
- Go to Kaggle, search "awesome chatgpt prompts"
- Download the CSV and rename it `archive.csv` in this folder

**Step 3: Run scripts in order**
```bash
python 01_intro_data_cleaning.py
python 02_intermediate_quality_scoring.py
python 03_visualization_dashboard.py
```

## Quality Scoring Method

Each prompt is scored 0–100 across 4 dimensions:

| Dimension | Max Points | What It Measures |
|-----------|-----------|-----------------|
| Clarity | 25 | Word count adequacy |
| Structure | 25 | Sentence variety |
| Vocabulary | 25 | Unique word ratio |
| Specificity | 25 | Use of directive language |

## Output Files

- `cleaned_prompts.csv` — cleaned dataset
- `scored_prompts.csv` — dataset with quality scores
- `ai_quality_dashboard.png` — 6-panel visualization dashboard

## Skills Demonstrated

- Python (pandas, numpy, matplotlib)
- Data cleaning and preprocessing
- Feature engineering and custom scoring
- Statistical analysis
- Data visualization and dashboard design
- Understanding of AI prompt quality (RLHF context)

## Author

**Rizqi Alfatah**  
AI Data Annotator & RLHF Specialist
