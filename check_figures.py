import json
from pathlib import Path

# Find latest summary file
summary_files = sorted(Path('data/output/summaries').glob('summaries_*.json'))

if not summary_files:
    print('❌ No summary files found in data/output/summaries/')
    exit(1)

summary_file = summary_files[-1]
print(f'📂 Reading: {summary_file}\n')

with open(summary_file) as f:
    data = json.load(f)

print('📊 Figure Usage Report')
print('='*70)

# Handle both list and dict formats
if isinstance(data, dict):
    summaries = data.get('summaries', [])
    metadata = data.get('metadata', {})
    if metadata:
        print(f"Total sections: {metadata.get('total_sections', 'N/A')}")
        print(f"Figures available: {metadata.get('figures_available', 'N/A')}")
        print('='*70)
else:
    # Data is directly a list
    summaries = data

found_any = False
for summary in summaries:
    figs = summary.get('figures_referenced', [])
    if figs:
        found_any = True
        section_id = summary['section_id']
        title = summary['title'][:40]
        figs_str = ', '.join(figs)
        print(f'Section {section_id:10s} | {title:40s} | Figures: {figs_str}')

if not found_any:
    print('\n⚠️  No figures referenced in any section')
    print('   This is expected if:')
    print('   1. No figures placed in data/processed/figures/ yet')
    print('   2. Or figure names don\'t match (should be figure_1.png, figure_2.png, etc.)')
    print('\n💡 To add figures:')
    print('   mkdir -p data/processed/figures')
    print('   # Place figure_1.png, figure_2.png, etc. there')
else:
    print(f'\n✅ Found {len([s for s in summaries if s.get("figures_referenced")])} sections with figures')
