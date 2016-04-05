
===
配置数据集的根目录
xucb@txjy-GPU-ubuntu:~/fuel$ more ~/.fuelrc 
data_path: "/data/fuelsets"

====
所有数据集都在 /data/fuelsets/

===使用流程
cd /data/fuelsets/
fuel-download mnist
fuel-convert mnist

即可在自己的代码中 
import MNIST 
train = MNIST(('train'))
test = MNIST(('test'))

====概念 ====
source的 意思是 lable, image这些
split 的意思是 把source分段， 即不同的数据集


===增加自己的数据集
在downloaders中增加自己的download，这是可选的

在converters中增加自己的convert, 把所有原始数据写到一个 hdf5文件中(不管是分成test,train,还是只有一个train,或test中没有label项,都写到这一个hdf5文件中，后者存放到/data/fuelsets中)

进行转换
cd /data/fuelsets/
fuel-conver xxx

使用
import XXX
train = XXX(('train'))
=====
