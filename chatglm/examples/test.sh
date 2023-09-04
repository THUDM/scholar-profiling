#!/bin/bash
dir="../output/test_dir/"
if [ ! -d "$dir" ];then
mkdir -p $dir
echo "创建文件夹成功"
else
echo "文件夹已经存在"
fi