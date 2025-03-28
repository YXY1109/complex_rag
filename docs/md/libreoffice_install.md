# 安装LibreOffice

```安装命令
yum install LibreOffice_24.8.3.2_Linux_x86-64_rpm/RPMS/*.rpm

ln -s /opt/libreoffice24.8/ libreoffice
libreoffice24.8 -version
```

```
GLIBCXX_3.4.xx版本不够，升级命令：
yum install gcc-c++
yum provides libstdc


问题1：
/opt/libreoffice24.8/program/oosplash: error while loading shared libraries: libXinerama.so.1: cannot open shared object file: No such file or directory
处理方案：
安装：yum install libXinerama


问题2:
/opt/libreoffice24.8/program/oosplash: /lib64/libstdc++.so.6: version `GLIBCXX_3.4.20' not found (required by /opt/libreoffice24.8/program/libuno_sal.so.3)
/opt/libreoffice24.8/program/oosplash: /lib64/libstdc++.so.6: version `CXXABI_1.3.9' not found (required by /opt/libreoffice24.8/program/libuno_sal.so.3)
/opt/libreoffice24.8/program/oosplash: /lib64/libstdc++.so.6: version `CXXABI_1.3.8' not found (required by /opt/libreoffice24.8/program/libuno_sal.so.3)
/opt/libreoffice24.8/program/oosplash: /lib64/libstdc++.so.6: version `GLIBCXX_3.4.21' not found (required by /opt/libreoffice24.8/program/libuno_sal.so.3)
处理方案：
参考文章1：https://www.jianshu.com/p/903ff504b7b7
参考文章2：https://blog.csdn.net/dm569263708/article/details/125198722
strings /usr/lib64/libstdc++.so.6 | grep GLIBC
find / -name "libstdc++.so*"
cd libstdc++.so.6.0.28所在目录下
cp libstdc++.so.6.0.28 /usr/lib64
cd /usr/lib64
rm -rf libstdc++.so.6
ln -s libstdc++.so.6.0.28 libstdc++.so.6

问题3:
/lib64/libc.so.6: version `GLIBC_2.18'
处理方案：
参考文章：https://www.02405.com/archives/1992

问题3:
/opt/libreoffice24.8/program/oosplash: error while loading shared libraries: libc.musl-x86_64.so.1: cannot open shared object file: No such file or directory
处理方案1（无效）：
安装：yum install musl-libc.x86_64
处理方案2:
find / -name libc.musl-x86_64.so.1 2>/dev/null
cd libc.musl-x86_64.so.1文件目录下
cp ld-musl-x86_64.so.1 /usr/lib64/
cd /usr/lib64
ln -s ld-musl-x86_64.so.1 libc.musl-x86_64.so.1


#安装升级gcc
参考文章：https://tianlingqun.blog.csdn.net/article/details/121990272
wget http://ftp.gnu.org/gnu/gcc/gcc-11.2.0/gcc-11.2.0.tar.gz
tar -zxvf gcc-11.2.0.tar.gz
cd gcc-11.2.0
下载预编译文件：./contrib/download_prerequisites 。这个不执行，运行下面的命令
下载预编译文件：sudo yum install gmp gmp-devel mpfr mpfr-devel libmpc libmpc-devel
mkdir build
cd build 
../configure --prefix=/root/local/gcc-11.2.0 --enable-bootstrap --enable-languages=c,c++ --enable-threads=posix --enable-checking=release --enable-multilib --with-system-zlib
../configure --prefix=/root/local/gcc-11.2.0 --enable-bootstrap --enable-languages=c,c++ --enable-threads=posix --enable-checking=release --enable-multilib
make -j4
make install
g++ -v
gcc -v


#编译gcc报错：
/usr/include/gnu/stubs.h:7:11: fatal error: gnu/stubs-32.h: 没有那个文件或目录
参考文章：https://stackoverflow.com/questions/7412548/error-gnu-stubs-32-h-no-such-file-or-directory-while-compiling-nachos-source
yum install glibc-devel.i686
```

```
https://blog.csdn.net/SerMa/article/details/131226445
https://blog.csdn.net/weixin_41010198/article/details/106780572

第一步：gcc4.8.5升级到9.3
参考文章：
https://blog.csdn.net/b_ingram/article/details/121569398

第二步：gcc9.3源码升级到11.2
参考文章：
https://www.jianshu.com/p/0012996cbdb4

第三步：升级glicxx到2.28
/usr/local/gcc

export LD_LIBRARY_PATH=
echo $LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH/usr/local/gcc/lib64:/usr/local/isl/lib:/usr/local/isl/lib

vim ~/.bashrc
source ~/.bashrc

rm -rf glibc-2.28
tar -zxf glibc-2.28.tar.gz
cd glibc-2.28 && mkdir build && cd build
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH/usr/local/gcc/lib64:/usr/local/isl/lib:/usr/local/isl/lib
../configure --prefix=/usr/local/glibc --disable-werror
make && make install

```

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib64/:/usr/lib/
    echo $LD_LIBRARY_PATH
