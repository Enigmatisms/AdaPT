python ./render.py --scene cbox --name cbox.xml --iter_num 8000 --arch cpu --type bdpt
python ./render.py --scene trans --name foam.xml --iter_num 40000 --arch cuda --type bdpt --normalize 0.99 -a