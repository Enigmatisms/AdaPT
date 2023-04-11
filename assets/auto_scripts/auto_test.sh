test_folder=$1
# for file in `ls ${test_folder}*.xml`; do
#     file_name=${file##*/}
#     CUDA_VISIBLE_DEVICES=0 python ./render.py --scene $2 --name ${file_name} --iter_num 200000 --arch cuda --type bdpt --no_gui -a --no_watermark >> log.txt
#     echo "Processing '$file_name'"
# done

sample_nums=(0 0 2 2 2 3 4 5)

for ((num=5;num<=7;num++)); do
    sample_num=$((${sample_nums[$num]}*100000))
    for file in `ls ${test_folder}foam-${num}*.xml`; do
        file_name=${file##*/}
        echo "Processing '$file_name' with ${sample_num} samples."
        CUDA_VISIBLE_DEVICES=3 python ./render.py --scene $2 --name ${file_name} --iter_num ${sample_num} --arch cuda --type bdpt --no_gui -a --no_save_fig --no_watermark >> log.txt
    done

    for file in `ls ${test_folder}nw-foam-${num}*.xml`; do
        file_name=${file##*/}
        echo "Processing '$file_name' with ${sample_num} samples."
        CUDA_VISIBLE_DEVICES=3 python ./render.py --scene $2 --name ${file_name} --iter_num ${sample_num} --arch cuda --type bdpt --no_gui -a --no_save_fig --no_watermark >> log.txt
    done
done