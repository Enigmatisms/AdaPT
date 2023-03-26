echo "" &> log.txt

while IFS= read -r file_name; do
    if [ ! ${file_name:0:1} = "#" ]; then
        echo "$file_name"
        echo "Processing $file_name ..."
        python ./tdom_analyze.py --config ../configs/tdom_sim.conf --sim_name $file_name --save_fig >> log.txt
    fi
done < file_list.conf