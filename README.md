# youku-sryun
steps:
1.下载训练、验证及测试集至train、val和test文件夹下，并且解压成一个个y4m文件
2.修改y4mtobmp.sh中的目录名，运行，可批量解码视频至图片
3.进入Anime-Super-Resolution-master目录下运行python train.py即可训练
4.修改test.py中的权重目录及输出目录，运行python test.py即可测试
5.修改bmptoy4m.sh中的目录名，运行，可批量转化result中的图片至视频。
[ok, all finish]可以压缩提交了(初赛压缩结果大小600m左右)
