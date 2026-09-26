import csv

with open('Amazon-ML-Challenge/dataset/test/test_source1.tsv', 'r', encoding='utf-8') as f_s1, \
     open('Amazon-ML-Challenge/output/matching_results_raw_0.459.tsv', 'r', encoding='utf-8') as f_res:
    r_s1 = csv.reader(f_s1, delimiter='\t')
    r_res = csv.reader(f_res, delimiter='\t')
    next(r_s1); next(r_res)
    for i in range(20):
        s1 = next(r_s1)
        res = next(r_res)
        m_list = res[1].split(',') if res[1].strip() else []
        print(f"[{s1[0]}] NAME: '{s1[1]}' | ADDR: '{s1[2]}' | CTRY: '{s1[3]}'")
        print(f"   PRED ({len(m_list)}): {res[1]}")
