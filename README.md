# Remote Data Careers Pipeline

## Introduction

I built this project using a 2023 dataset of data job postings collected by Luke Barousse. The data skews heavily toward the US, but I wanted to see what it could still tell us about remote work — where skills tend to matter more than geography, and where a US-centric dataset can still be genuinely useful to someone job-hunting anywhere else.

Instead of jumping straight into analysis, I first spent time cleaning and structuring the raw dataset myself: handling missing values, parsing nested fields, standardizing formats. It's a step that doesn't show up in most portfolio projects, but it's usually where the real work happens — and where I wanted to practice fundamentals I'm building as I move between data analysis and data engineering.

From there, I focused the analysis around a few concrete questions:

1. **Data Cleaning & Structuring** — How can raw job posting data be transformed into a clean, analysis-ready dataset?
2. **Remote Landscape** — Which countries and roles offer the most remote data job opportunities?
3. **Remote Salary Premium** — Does remote work pay more or less than on-site roles, and does this vary by skill or role?
4. **Top Skills for Remote Roles** — Which skills are most in-demand and best-paid specifically for remote positions?
5. **Beyond the US** — What does the data job market look like when we exclude US-based postings?

I chose to center this project on remote work specifically because it's the one angle where a US-heavy dataset still translates directly into something I can use: if a skill or role consistently shows up as valuable for remote positions, that's a signal worth acting on regardless of where I'm located. Rather than treating the US bias as a limitation to work around, I wanted to make it part of the story.

Data source: [Luke Barousse's Data Nerd Skills dataset](lien)
Full code and notebooks: [lien vers ton repo]

---

## Data Preparation & Cleanup

Before diving into analysis, I spent time cleaning and structuring the raw dataset — a step that's easy to skip in portfolio projects, but where most of the real work actually happens.

### Parsing nested columns

`job_skills` and `job_type_skills` were stored as string representations of Python objects (a list and a dictionary, respectively) rather than usable data types.

```python
def parse_skills(x):
    if isinstance(x, str):
        return ast.literal_eval(x)
    return x

df['job_skills'] = df['job_skills'].apply(parse_skills)
df['job_type_skills'] = df['job_type_skills'].apply(parse_skills)
```

### Fixing data types

`job_posted_date` was stored as a plain string and converted to a proper `datetime` type, enabling time-based analysis later on.

```python
df['job_posted_date'] = pd.to_datetime(df['job_posted_date'])
```

### Handling missing values — without losing data

Rather than dropping every row with a missing value, I checked each affected column individually to see whether the missing data was randomly distributed or concentrated in specific segments.

```python
cols_to_check = ['job_title', 'job_location', 'job_via', 'job_schedule_type', 'job_country', 'company_name']

for col in cols_to_check:
    print(f"--- Missing values in '{col}' ---")
    print(f"Total missing: {df[col].isnull().sum()} ({df[col].isnull().mean()*100:.2f}%)")
    null_dist = df[df[col].isnull()]['job_country'].value_counts(normalize=True).head(5)
    overall_dist = df['job_country'].value_counts(normalize=True).head(5)
    print("\nTop countries among missing rows:")
    print(null_dist)
    print("\nTop countries overall (for comparison):")
    print(overall_dist)
```

**Findings**:
- `job_location` (0.13% missing) and `job_country` (0.01% missing): negligible volume, left as-is.
- `job_schedule_type` (1.6% missing): missing values were concentrated in a few countries (notably the Philippines and Malaysia) rather than randomly spread. Given the low overall volume, these rows were kept and filled instead of dropped:

```python
df['job_schedule_type'] = df['job_schedule_type'].fillna('Not Specified')
```

- `job_skills` / `job_type_skills`: left untouched at this stage — missing values are filtered out only within the specific analyses that require skill data.

### Isolating salary data

Only about 4% of postings include salary information. Rather than dropping the other 96% or dragging incomplete columns through every analysis, these rows were isolated into a dedicated subset.

```python
df_salary = df[df['salary_year_avg'].notna() | df['salary_hour_avg'].notna()].copy()
```

**Result**: 32,665 rows out of 785,741 (~4.2%) — kept separate so the main dataset stays fully usable for non-salary analyses.

### Validating the remote work flag

Since this project focuses heavily on remote opportunities, I checked whether `job_work_from_home` could be trusted as the single indicator of remote status.

```python
df[df['job_work_from_home'] == True]['job_location'].value_counts().head(10)
```

**Finding**: Every row flagged as remote had `job_location == 'Anywhere'`, with no exceptions — confirming `job_work_from_home` as a fully reliable single source of truth for remote status. Remote postings make up **8.85%** of the dataset (69,552 out of 785,741 rows).

### Exporting clean data

```python
df.to_parquet('../data/clean/data_jobs_clean.parquet', index=False)
df_salary.to_parquet('../data/clean/data_jobs_salary.parquet', index=False)
```

The `.parquet` format preserves data types (dates, booleans, parsed lists/dicts) without needing to re-parse them on reload — a small but deliberate choice to mirror a reusable data pipeline rather than a one-off cleaning script.

---

## Remote Job Landscape

**Question**: Which countries and roles offer the most remote data job opportunities?

Remote postings make up **8.85%** of the dataset — a relatively small slice, but one worth digging into carefully, since where the *volume* of remote jobs is highest isn't always where the *share* of remote jobs is highest.

### Remote share by country

```python
country_remote_share = (
    df.groupby('job_country')['job_work_from_home']
    .agg(['sum', 'count'])
    .rename(columns={'sum': 'remote_count', 'count': 'total_postings'})
)
country_remote_share['remote_pct'] = (country_remote_share['remote_count'] / country_remote_share['total_postings']) * 100
country_remote_share = country_remote_share[country_remote_share['total_postings'] >= 500]
top_remote_share = country_remote_share.sort_values('remote_pct', ascending=False).head(10)
```

```python
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
bg_color = '#1e1e1e'
fig.patch.set_facecolor(bg_color)

def get_dark_bg_palette(n):
    full_palette = sns.color_palette('Blues', n_colors=n + 4)
    return full_palette[4:][::-1]

top_volume = top_remote_countries.reset_index()
top_volume.columns = ['job_country', 'remote_count']
palette_volume = get_dark_bg_palette(len(top_volume))
sns.barplot(data=top_volume, y='job_country', x='remote_count', hue='job_country',
            palette=palette_volume, legend=False, ax=axes[0])
axes[0].set_facecolor(bg_color)
axes[0].set_title('Top Countries by Remote Job Volume', fontsize=12, weight='bold', color='white')
axes[0].set_xlabel('Number of Remote Postings', color='white')
axes[0].set_ylabel('')
axes[0].tick_params(colors='white')
axes[0].grid(False)
axes[0].spines[['top', 'right']].set_visible(False)
axes[0].spines[['left', 'bottom']].set_color('white')

top_share = top_remote_share.reset_index()
palette_share = get_dark_bg_palette(len(top_share))
sns.barplot(data=top_share, y='job_country', x='remote_pct', hue='job_country',
            palette=palette_share, legend=False, ax=axes[1])
axes[1].set_facecolor(bg_color)
axes[1].set_title('Top Countries by Remote Job Share (%)', fontsize=12, weight='bold', color='white')
axes[1].set_xlabel('Remote Postings (% of Total)', color='white')
axes[1].set_ylabel('')
axes[1].tick_params(colors='white')
axes[1].grid(False)
axes[1].spines[['top', 'right']].set_visible(False)
axes[1].spines[['left', 'bottom']].set_color('white')

plt.tight_layout()
plt.savefig('../images/remote_volume_vs_share.png', dpi=300, bbox_inches='tight', facecolor=bg_color)
```

![Remote Volume vs Share by Country](images/remote_volume_vs_share.png)

The US, India, and the UK post the most remote jobs in absolute numbers — unsurprising, since they also post the most jobs overall. But the share of remote postings *within* each country tells a different story: Ukraine (28.7%), Turkey (26.0%), and Kazakhstan (24.6%) show a much higher proportion of remote work relative to their total job volume, despite having far smaller overall markets than the US.

### Remote share by role

```python
role_remote_share = (
    df.groupby('job_title_short')['job_work_from_home']
    .agg(['sum', 'count'])
    .rename(columns={'sum': 'remote_count', 'count': 'total_postings'})
)
role_remote_share['remote_pct'] = (role_remote_share['remote_count'] / role_remote_share['total_postings']) * 100
top_role_share = role_remote_share.sort_values('remote_pct', ascending=False)
```

```python
fig, ax = plt.subplots(figsize=(10, 6))
bg_color = '#1e1e1e'
fig.patch.set_facecolor(bg_color)
ax.set_facecolor(bg_color)

data = top_role_share.reset_index().sort_values('remote_pct', ascending=True)
colors = sns.color_palette('Blues', n_colors=len(data) + 4)[4:]

ax.hlines(y=data['job_title_short'], xmin=0, xmax=data['remote_pct'], color='#555555', linewidth=1.5)
ax.scatter(data['remote_pct'], data['job_title_short'], color=colors, s=200, zorder=3)

for i, (val, role) in enumerate(zip(data['remote_pct'], data['job_title_short'])):
    ax.text(val + 0.4, i, f'{val:.1f}%', va='center', color='white', fontsize=9)

ax.set_title('Remote Job Share by Role', fontsize=13, weight='bold', color='white')
ax.set_xlabel('Remote Postings (% of Total)', color='white')
ax.set_ylabel('')
ax.tick_params(colors='white')
ax.grid(False)
ax.spines[['top', 'right']].set_visible(False)
ax.spines[['left', 'bottom']].set_color('white')

plt.tight_layout()
plt.savefig('../images/remote_share_by_role.png', dpi=300, bbox_inches='tight', facecolor=bg_color)
```

![Remote Share by Role](images/remote_share_by_role.png)

At the role level, volume and share align consistently: **Data Engineer** positions have a remote share of 11.4%, almost double that of **Data Analyst** roles (6.8%). Senior-level roles are also more remote-friendly than junior ones across the board.

**Takeaway**: for someone building toward a hybrid Data Analyst/Data Engineer path, this is a useful signal — moving toward data engineering skills and gaining seniority seems to open up remote opportunities more reliably than focusing purely on which country posts the most jobs.

---
## Remote Salary Premium

**Question**: Does remote work pay more or less than on-site roles, and does this vary by role or skill?

Remote roles show a clear overall premium — **+12% in median annual salary** ($128,830 remote vs $115,000 on-site). But this headline number hides a lot of nuance once you break it down by role.

### Premium by role

Looking only at roles with a reliable sample size (150+ remote postings with salary data), the premium is much more modest — and sometimes negative:

![Remote Salary Premium by Role](images/remote_salary_premium_by_role.png)

**Data Scientist** (+8.8%) and **Data Engineer** (+4.0%) show a small positive premium, while **Data Analyst** (-3.6%) and **Senior Data Analyst** (-5.6%) actually pay slightly *less* remotely than on-site. This means the global +12% premium is mostly driven by a mix of roles and seniority levels — not something Analyst/Engineer-track roles can count on by default.

### Top paying skills

```python
df_remote_skills = df_salary[
    (df_salary['job_work_from_home'] == True) & 
    (df_salary['job_skills'].notna())
].copy()
df_remote_skills = df_remote_skills.explode('job_skills')

skill_salary = (
    df_remote_skills.groupby('job_skills')['salary_year_avg']
    .agg(['median', 'count'])
    .rename(columns={'median': 'median_salary', 'count': 'postings_count'})
)
skill_salary = skill_salary[skill_salary['postings_count'] >= 50]
top_paying_skills = skill_salary.sort_values('median_salary', ascending=False).head(15)
```

![Top 5 Paying Remote Skills](images/top_paying_remote_skills.png)

The highest-paying remote skills are dominated by cloud/data engineering tools (Kubernetes, Terraform, GCP, Airflow, Kafka — ~$145K median) and machine learning frameworks (PyTorch, TensorFlow, Scikit-learn — $145K–$150K). Notably, traditional analyst tools like Excel, Tableau, or Power BI don't appear anywhere near this top tier.

I also checked whether broader skill *categories* (programming, cloud, databases, etc.) showed any salary difference — they didn't. The distributions were nearly identical across categories, meaning the specific skill matters far more than the category it belongs to.

**Takeaway**: the clearest path to higher-paying remote work isn't the job title alone — it's specific technical skills, particularly around cloud infrastructure, orchestration, and machine learning.

---
