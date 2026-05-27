# ============================================================
# SCRIPT 3: Visualization Dashboard
# AI Prompt Quality Analyzer
# ============================================================
# INPUT : scored_prompts.csv
# OUTPUT: ai_quality_dashboard.png
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

print("=" * 50)
print("STEP 1: LOADING DATA")
print("=" * 50)

df = pd.read_csv('scored_prompts.csv')
print(f"Loaded {len(df)} rows")

COLORS = {
    'primary':  '#4ECDC4',
    'secondary':'#FF6B6B',
    'accent':   '#f9c74f',
    'bg':       '#0f0f23',
    'card':     '#1a1a2e',
    'text':     '#e0e0e0',
    'grid':     '#2a2a4a'
}
CATEGORY_COLORS = {
    'High': '#00d4aa', 'Medium': '#f9c74f',
    'Low': '#f77f00', 'Very Low': '#e63946'
}

plt.style.use('dark_background')
fig = plt.figure(figsize=(20, 13), facecolor=COLORS['bg'])
fig.suptitle('AI Prompt Quality Analyzer — Dashboard',
             fontsize=24, fontweight='bold', color='white', y=0.97)
fig.text(0.5, 0.935,
         'Analyzing ChatGPT Prompt Quality Across Use Cases',
         ha='center', fontsize=12, color='#aaaaaa')

gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35,
                       left=0.06, right=0.97, top=0.90, bottom=0.06)


# ── Chart 1: Score Distribution
ax1 = fig.add_subplot(gs[0, 0])
ax1.set_facecolor(COLORS['card'])
ax1.hist(df['quality_score'], bins=20, color=COLORS['primary'],
         alpha=0.8, edgecolor='none')
ax1.axvline(df['quality_score'].mean(), color=COLORS['secondary'],
            linestyle='--', linewidth=2, label=f"Mean: {df['quality_score'].mean():.1f}")
ax1.set_title('Quality Score Distribution', fontweight='bold', color='white', pad=10)
ax1.set_xlabel('Quality Score (0–100)', color=COLORS['text'])
ax1.set_ylabel('Count', color=COLORS['text'])
ax1.tick_params(colors=COLORS['text'])
ax1.legend(facecolor=COLORS['card'], labelcolor='white')
for spine in ax1.spines.values(): spine.set_edgecolor(COLORS['grid'])


# ── Chart 2: Quality Category Pie
ax2 = fig.add_subplot(gs[0, 1])
ax2.set_facecolor(COLORS['card'])
cat_counts = df['quality_category'].value_counts()
cat_order  = [c for c in ['High','Medium','Low','Very Low'] if c in cat_counts.index]
sizes  = [cat_counts[c] for c in cat_order]
colors = [CATEGORY_COLORS[c] for c in cat_order]
wedges, texts, autotexts = ax2.pie(
    sizes, labels=cat_order, colors=colors,
    autopct='%1.1f%%', startangle=90,
    textprops={'color': 'white'}
)
for at in autotexts: at.set_fontsize(10)
ax2.set_title('Quality Category Breakdown', fontweight='bold', color='white', pad=10)


# ── Chart 3: Top 10 Categories by Score
ax3 = fig.add_subplot(gs[0, 2])
ax3.set_facecolor(COLORS['card'])
top10 = df.groupby('act')['quality_score'].mean().sort_values(ascending=True).tail(10)
bars = ax3.barh(range(len(top10)), top10.values,
                color=COLORS['primary'], edgecolor='none')
ax3.set_yticks(range(len(top10)))
ax3.set_yticklabels([t[:28] for t in top10.index], color=COLORS['text'], fontsize=8)
ax3.set_title('Top 10 Prompt Categories\nby Avg Quality Score',
              fontweight='bold', color='white', pad=10)
ax3.set_xlabel('Average Score', color=COLORS['text'])
ax3.tick_params(colors=COLORS['text'])
for spine in ax3.spines.values(): spine.set_edgecolor(COLORS['grid'])


# ── Chart 4: Word Count Distribution
ax4 = fig.add_subplot(gs[1, 0])
ax4.set_facecolor(COLORS['card'])
wc_cap = df['word_count'].quantile(0.95)
ax4.hist(df['word_count'].clip(upper=wc_cap), bins=25,
         color=COLORS['accent'], alpha=0.8, edgecolor='none')
ax4.axvline(df['word_count'].mean(), color=COLORS['secondary'],
            linestyle='--', linewidth=2,
            label=f"Mean: {df['word_count'].mean():.0f} words")
ax4.set_title('Prompt Word Count Distribution', fontweight='bold', color='white', pad=10)
ax4.set_xlabel('Word Count', color=COLORS['text'])
ax4.set_ylabel('Count', color=COLORS['text'])
ax4.tick_params(colors=COLORS['text'])
ax4.legend(facecolor=COLORS['card'], labelcolor='white')
for spine in ax4.spines.values(): spine.set_edgecolor(COLORS['grid'])


# ── Chart 5: Score by Quality Category (Box)
ax5 = fig.add_subplot(gs[1, 1])
ax5.set_facecolor(COLORS['card'])
cat_order_box = [c for c in ['Very Low','Low','Medium','High']
                 if c in df['quality_category'].unique()]
data_by_cat = [df[df['quality_category'] == c]['quality_score'].values
               for c in cat_order_box]
bp = ax5.boxplot(data_by_cat, labels=cat_order_box, patch_artist=True,
                 medianprops=dict(color='white', linewidth=2))
for patch, cat in zip(bp['boxes'], cat_order_box):
    patch.set_facecolor(CATEGORY_COLORS[cat])
for el in ['whiskers','caps','fliers']:
    for item in bp[el]: item.set_color(COLORS['text'])
ax5.set_title('Score Distribution\nby Quality Category',
              fontweight='bold', color='white', pad=10)
ax5.set_ylabel('Quality Score', color=COLORS['text'])
ax5.tick_params(colors=COLORS['text'])
for spine in ax5.spines.values(): spine.set_edgecolor(COLORS['grid'])


# ── Chart 6: Quality Score vs Word Count Scatter
ax6 = fig.add_subplot(gs[1, 2])
ax6.set_facecolor(COLORS['card'])
sample = df.sample(min(500, len(df)), random_state=42)
sc = ax6.scatter(
    sample['word_count'].clip(upper=wc_cap),
    sample['quality_score'],
    c=sample['quality_score'], cmap='RdYlGn',
    alpha=0.6, s=20, edgecolors='none', vmin=0, vmax=100
)
plt.colorbar(sc, ax=ax6, label='Score').ax.yaxis.label.set_color('white')
ax6.set_title('Quality Score vs Word Count', fontweight='bold', color='white', pad=10)
ax6.set_xlabel('Word Count', color=COLORS['text'])
ax6.set_ylabel('Quality Score', color=COLORS['text'])
ax6.tick_params(colors=COLORS['text'])
for spine in ax6.spines.values(): spine.set_edgecolor(COLORS['grid'])


plt.savefig('ai_quality_dashboard.png', dpi=150, bbox_inches='tight',
            facecolor=COLORS['bg'], edgecolor='none')
print("\n✅ Saved: ai_quality_dashboard.png")
print("🎉 Project complete! Upload folder ke GitHub.")
