import csv, shutil, os

input_path = "output/matching_results.tsv"
backup_path = "output/matching_results_v12_raw_0.587.tsv"
output_path = "output/matching_results.tsv"

# 1. Backup the 0.587 submission file
if not os.path.exists(backup_path):
    print(f"Backing up 0.587 file to {backup_path}...")
    shutil.copyfile(input_path, backup_path)

print("Applying High-Precision Top-1 Pruning (Strict Top 1 S2 + Top 1 S3)...")
temp_out = input_path + ".tmp"
total_before = 0
total_after = 0
total_entities = 0

with open(backup_path, 'r', encoding='utf-8') as f_in, \
     open(temp_out, 'w', encoding='utf-8', newline='') as f_out:
    
    r = csv.reader(f_in, delimiter='\t')
    header = next(r)
    f_out.write(f"{header[0]}\t{header[1]}\n")

    for row in r:
        total_entities += 1
        s1_id = row[0]
        if len(row) > 1 and row[1].strip():
            raw_matches = row[1].strip().split(',')
            total_before += len(raw_matches)
            
            # Highest confidence match is at index 0 for each source
            s2_top = [m for m in raw_matches if m.startswith('S2-')][:1]
            s3_top = [m for m in raw_matches if m.startswith('S3-')][:1]
            
            filtered = s2_top + s3_top
            total_after += len(filtered)
            f_out.write(f"{s1_id}\t{','.join(filtered)}\n")
        else:
            f_out.write(f"{s1_id}\t\n")

# Atomically replace
shutil.move(temp_out, output_path)

print("=" * 65)
print("HIGH-PRECISION TOP-1 PRUNING COMPLETE:")
print(f"  Total S1 Entities:     {total_entities:,}")
print(f"  Matches Before:        {total_before:,} (avg {total_before/total_entities:.2f}/entity)")
print(f"  Matches After:         {total_after:,} (avg {total_after/total_entities:.2f}/entity)")
print(f"  Lower-Rank Noise Cut:  {total_before - total_after:,} lower-confidence candidates removed")
print(f"  Output Saved to:       {output_path}")
print("=" * 65)
