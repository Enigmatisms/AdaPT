test_folder=$1
for file in `ls ${test_folder}*.xml`; do
    file_name=${file##*/}
    CUDA_VISIBLE_DEVICES=1 python ./render.py --scene $2 --name ${file_name} --iter_num 4300 --arch cuda --type bdpt --no_gui --no_watermark
    echo "Processing $file_name"
done