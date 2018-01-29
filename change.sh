#! /bin/bash
if [[ $1 = "inference" ]]
then
	unlink Makefile 
	unlink src/main.cc
	ln -s Makefile.inference Makefile 
	ln -s main.inference.cc src/main.cc 
	make clean
	make 
elif [[ $1 = "train" ]]
then
	unlink Makefile 
	unlink src/main.cc
	ln -s Makefile.train Makefile 
	ln -s main.train.cc src/main.cc 
	make clean
	make 
fi
