# Fooocus-inswapper

This is a fork of [Fooocus](https://github.com/lllyasviel/Fooocus).  This fork integrates the popular Insightface/[inswapper](https://github.com/haofanwang/inswapper) library used by roop, ReActor, and others.  The goal of this repository is to stay up-to-date with the main repository, while also maintaining the inswapper integration.

For more detailed and official documentation, please refer to [lllyasviel's repository](https://github.com/lllyasviel/Fooocus).

A standalone installation does not exist for this repository.

## Installation (Windows)

The installation assumes CUDA 11.8.  If you need a different version, please update `configure.bat` with the correct URL to the desired CUDA version.

1. Run `git clone https://github.com/machineminded/Fooocus-inswapper.git`
2. Execute `configure.bat`

## Usage

Inswapper will activate if "Input Image" and "Enabled" are both checked.

1. `.\venv\Scripts\activate`
2. `python launch.py`

https://github.com/machineminded/Fooocus-inswapper/assets/155763297/68f69e95-8306-4c7b-8f9b-0013352460b6

## Issues

Please report any issues in the Issues tab.  I will try to help as much as I can.

## To Do

1. 🚀 Allow changing of insightface parameters
2. 🚀 Allow customizable target image
3. 🐛 [Fix an issue with multiple faces](https://github.com/machineminded/Fooocus-inswapper/issues/2)
