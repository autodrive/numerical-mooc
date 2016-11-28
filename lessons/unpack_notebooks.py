# -*- coding: utf8 -*-
import unpack_ipynb as ui
import os


def main():
    ui.convert_tree(os.path.abspath(os.curdir))


if '__main__' == __name__:
    main()
