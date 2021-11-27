
# myWolfTeamWork
基于    tensorrtx v5.0  
添加了  上交的四点模型  
数据处理版本  
---
---
① 注意版本  
② trtx的 classnum 在推理的时候不用改, 但wts转换engine的时候要改成对应的  
③ 上交的opt4需要C++17上才能编译, 而trtx则需要c++14  `貌似`

前置安装：
    使用vscode
    需要ubuntu系统>=18  https://www.roboxx.ltd/2104.html
1、cuda>=11
2、cudnn  https://www.roboxx.ltd/2182.html

3、cmake>=3.15
4、g++/gcc==11
5、opencv>=4.10
6、tensorrt https://github.com/wang-xinyu/tensorrtx/tree/master/yolov5
7、zeromq https://zeromq.org/

以上链接仅供参考，可以不完全按照。

注：完成前置安装后，需要先进行编译，将build文件夹删掉，在建立一个build并进入其中执行
cmake ..
make

有兴趣的可以学学vscode和cmake的知识，视频：https://www.bilibili.com/video/BV1fy4y1b7TC?spm_id_from=333.999.0.0

编译成功后会生成main可执行文件，使用./main 运行
视频和模型github上传不了，编译成功后需要的找我拷贝。模型是yolov5，视频为1g多。
运行后点击图片的四个箱子与地面的交点，分别是左上、右上、左下、右下，然后再点击一次生成矩阵，按q完成运行。