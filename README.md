
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
    需要ubuntu系统>=18
1、cuda
2、cudnn
3、cmake>=3.15
4、g++/gcc==11
5、opencv>=4.10
6、tensorrt