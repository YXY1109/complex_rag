## 百度网盘下载包：

```
安装顺序：Base->LibreOffice_7.1.0.2_Linux_x86-64_rpm->LibreOffice_7.1.0.2_Linux_x86-64_rpm_langpack_zh-CN
rpm -Uivh *.rpm

libreoffice7.1 --version

sudo useradd -s /sbin/nologin mockbuild

缺少：libXinerama.so.1
安装命令：yum install libXinerama

缺少：libcairo.so.2
安装命令：yum install cairo

缺少：libcups.so.2
安装命令：yum install cups-libs

缺少：libSM.so.6
安装命令：yum install libSM

#安装java11
yum install java-11-openjdk
java -version
```