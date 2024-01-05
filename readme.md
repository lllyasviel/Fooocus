# Fooocus-inswapper

This is a fork of [Fooocus](https://github.com/lllyasviel/Fooocus).  This fork integrates the popular Insightface/[inswapper](https://github.com/haofanwang/inswapper) library used by roop, ReActor, and others.  For more detailed and official documentation, please refer to lllyasviel's repository.

A standalone installation does not exist for this repository.

## Installation (Windows)

The installation assumes CUDA 11.8.  If you need a different version, please update `configure.bat` with the correct URL to the desired CUDA version.

1. Run `git clone https://github.com/haofanwang/inswapper.git`
2. Execute `configure.bat`

## Usage

1. `.\venv\Scripts\activate`
2. `python launch.py`



## Issues

Please report any issues in the Issues tab.  I will try to help as much as I can.

## To Do

1. Allow changing of insightface parameters
2. Allow customizable target image