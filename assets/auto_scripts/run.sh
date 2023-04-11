python ./render.py --scene cbox --name cbox.xml --iter_num 8000 --arch cuda --type bdpt
python ./render.py --scene trans --name foam.xml --iter_num 200000 --arch cuda --type bdpt --normalize 0.99 -a --no_gui
python ./render.py --scene foam_test --name nw-foam-4-nc-500.xml --iter_num 200000 --arch cuda --type bdpt --normalize 0.99 -a --no_save_fig --no_gui