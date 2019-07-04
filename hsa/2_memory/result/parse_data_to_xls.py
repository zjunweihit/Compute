#!/usr/bin/python

import xlsxwriter
import re
#import sys

def read_file(fname):
    d2d_list = []
    h2d_list = []
    d2h_list = []

    f = open(fname)
    lines = f.readlines()

    for line in lines:
        l = line.split()
        item = (l[2], l[-2])
        copy_t = re.findall(r'[[](.*?)[]]', l[0])
        if copy_t[0] == 'D2D':
            d2d_list.append(item)
        elif copy_t[0] == 'H2D':
            h2d_list.append(item)
        elif copy_t[0] == 'D2H':
            d2h_list.append(item)
        else:
            print "Unknown Type: ", copy_t

    f.close()
    return d2d_list, h2d_list, d2h_list

#wb = xlsxwriter.Workbook("data.xlsx")
#ws = wb.add_worksheet()
#
#ws.write(0, 0, 'hello')
#ws.write(1, 0, '1 0')
#ws.write(0, 1, '0 1')
#
#wb.close()

if __name__ == "__main__":
    (d2d, h2d, d2h) = read_file("shader_dma.log")
    #write_xlsx(engine, d2d, h2d, d2h, base_line)

