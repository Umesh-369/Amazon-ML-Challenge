import os
import shutil

OUTPUT_DIR = "output"
SRC_FILE = os.path.join(OUTPUT_DIR, "matching_results_v13_raw_0.652.tsv")
DEST_FILE = os.path.join(OUTPUT_DIR, "matching_results.tsv")
TEMP_FILE = os.path.join(OUTPUT_DIR, "temp_pruned_matching.tsv")

print(f"Reading from: {SRC_FILE}")
print(f"Writing to:   {DEST_FILE}")

total_rows = 0
total_matches_orig = 0
total_matches_pruned = 0
empty_orig = 0
empty_pruned = 0
pruned_3rd_s2 = 0
pruned_3rd_s3 = 0

with open(SRC_FILE, "r", encoding="utf-8") as fin, open(TEMP_FILE, "w", encoding="utf-8", newline="") as fout:
    header = fin.readline()
    fout.write(header)
    
    for line in fin:
        total_rows += 1
        line = line.rstrip("\r\n")
        parts = line.split("\t")
        s1_id = parts[0]
        matches = [m.strip() for m in parts[1].split(",") if m.strip()] if len(parts) > 1 and parts[1].strip() else []
        
        total_matches_orig += len(matches)
        if not matches:
            empty_orig += 1
            empty_pruned += 1
            fout.write(f"{s1_id}\t\n")
            continue
            
        s2 = [m for m in matches if m.startswith("S2-")]
        s3 = [m for m in matches if m.startswith("S3-")]
        
        # High-precision pruning: cap at top 2 S2 and top 2 S3
        if len(s2) > 2:
            pruned_3rd_s2 += (len(s2) - 2)
            s2 = s2[:2]
        if len(s3) > 2:
            pruned_3rd_s3 += (len(s3) - 2)
            s3 = s3[:2]
            
        kept = s2 + s3
        total_matches_pruned += len(kept)
        if not kept:
            empty_pruned += 1
            fout.write(f"{s1_id}\t\n")
        else:
            fout.write(f"{s1_id}\t{','.join(kept)}\n")

print("\n" + "=" * 60)
print("FINAL PRUNING SUMMARY:")
print(f"  Total S1 Entities:     {total_rows:,}")
print(f"  Empty Entities:        {empty_pruned:,} ({empty_pruned/total_rows*100:.2f}%)")
print(f"  Original Matches:      {total_matches_orig:,} (avg: {total_matches_orig/total_rows:.4f})")
print(f"  Optimized Matches:     {total_matches_pruned:,} (avg: {total_matches_pruned/total_rows:.4f})")
print(f"  Pruned 3rd+ S2:        {pruned_3rd_s2:,}")
print(f"  Pruned 3rd+ S3:        {pruned_3rd_s3:,}")
print(f"  Total False Positives: {pruned_3rd_s2 + pruned_3rd_s3:,}")
print("=" * 60)

# Replace target file atomically
shutil.move(TEMP_FILE, DEST_FILE)
print("Successfully replaced output/matching_results.tsv!")
