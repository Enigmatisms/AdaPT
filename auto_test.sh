test_folder=$1
for file in `ls ${test_folder}*.xml`; do
    file_name=${file##*/}
    CUDA_VISIBLE_DEVICES=0 python ./render.py --scene $2 --name ${file_name} --iter_num 200000 --arch cuda --type bdpt --no_gui -a --no_watermark >> log.txt
    echo "Processing '$file_name'"
done