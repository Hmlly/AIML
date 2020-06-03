由于直接安装cvxopt包会出问题，并且我没有找到在pycharm里解决
该问题的方法，所以推荐用conda来运行，请注意，需要将main.py中
第141-142行的地址改为当前有效地址，即可运行。在176行-191行
是整个程序的测试部分，可以进行适当的注释，以得到想要的结果。

运行python代码的环境可以进行如下的配置：

conda install pandas
conda install numpy (如果没有在安装Pandas时安装numpy的话)
conda install -c conda-forge/label/gcc7 cvxopt （Win10）
conda install -c conda-forge cvxopt （linux）


本人使用环境：

Win10 pro + pycharm + miniconda3