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

## Data Preparation & Cleanup

Before diving into analysis, I spent time cleaning and structuring the raw dataset — a step that's easy to skip in portfolio projects, but where most of the real work actually happens.

### Parsing nested columns
`job_skills` and `job_type_skills` were stored as string representations of Python objects (a list and a dictionary, respectively) rather than usable data types. Both were parsed using `ast.literal_eval()` to restore their original structure.

### Fixing data types
`job_posted_date` was stored as a plain string and converted to a proper `datetime` type, enabling time-based analysis later on (monthly trends, posting patterns, etc.).

### Handling missing values — without losing data
Rather than dropping every row with a missing value, I looked at each affected column individually and checked whether the missing data was randomly distributed or concentrated in specific segments (e.g., certain countries).

- **`job_location`** (0.13% missing) and **`job_country`** (0.01% missing): negligible volume, left as-is.
- **`job_schedule_type`** (1.6% missing): missing values were noticeably concentrated in a few countries (e.g., the Philippines and Malaysia), rather than randomly spread. Given the low overall volume, these rows were kept and the missing values filled with `'Not Specified'` instead of being dropped — preserving the data while staying transparent about what's actually known.
- **`job_skills` / `job_type_skills`**: left untouched at this stage. Missing values are filtered out only within the specific analyses that require skill data, rather than removed from the dataset as a whole.

### Isolating salary data
Only about 4% of postings (32,665 out of 785,741) include salary information. Rather than dropping the other 96% or dragging incomplete columns through every analysis, these rows were isolated into a dedicated `df_salary` subset — keeping the main dataset fully usable for non-salary analyses, while salary-specific questions draw from a clean, complete sample.

### Validating the remote work flag
Since this project focuses heavily on remote opportunities, I checked whether `job_work_from_home` could be trusted as the single indicator of remote status. Cross-checking it against `job_location` confirmed a perfect match: every row flagged as remote had `job_location == 'Anywhere'`, with no exceptions. Remote postings make up about **8.85%** of the dataset (69,552 out of 785,741 rows).

### Exporting clean data
The cleaned dataset — along with the isolated salary subset — was exported to `.parquet` format rather than `.csv`, preserving data types (dates, booleans, parsed lists/dicts) without needing to re-parse them on reload. This mirrors a lightweight, reusable data pipeline structure rather than a one-off cleaning script.
