dir=processed_data/synthetic_experiments/uniform
mkdir -p $dir

for i in {1..20}; do ./sim -d 4 -n 100 -s 200 -o $dir/seqs-4-100-200.$i.txt -t $dir/P-4-100-200.$i.txt; done
for i in {1..20}; do head -80 $dir/seqs-4-100-200.$i.txt > $dir/seqs-4-100-200-train.$i.txt; done
for i in {1..20}; do tail -20 $dir/seqs-4-100-200.$i.txt > $dir/seqs-4-100-200-test.$i.txt; done
