#!/usr/bin/python

import xlsxwriter
import re

global ws

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

def collect_data(name, a, b, c, rowBase):
    if len(a) != len(b):
        print "Error: input arrays length are not equal!\n"

    j = 1
    colBase = 2
    ws.write(rowBase,     colBase - 2, name)
    ws.write(rowBase + 1, colBase - 1, "Shader DMA")
    ws.write(rowBase + 2, colBase - 1, "CPDMA")
    ws.write(rowBase + 3, colBase - 1, "SDMA")
    #print ("(%d, %d): %s" % (rowBase,     colBase - 1, name))
    #print ("(%d, %d): %s" % (rowBase + 1, colBase - 1, "Shader DMA"))
    #print ("(%d, %d): %s" % (rowBase + 2, colBase - 1, "CPDMA"))
    #print ("(%d, %d): %s" % (rowBase + 3, colBase - 1, "SDMA"))

    for i in range(len(a)):
        ws.write(rowBase,     colBase + i, int(a[i][0]))
        ws.write(rowBase + 1, colBase + i, float(a[i][1]))
        ws.write(rowBase + 2, colBase + i, float(b[i][1]))
        ws.write(rowBase + 3, colBase + i, float(c[i][1]))
        #print ("(%d, %d): %s" % (rowBase,     colBase + i, a[i][0]))
        #print ("(%d, %d): %s" % (rowBase + 1, colBase + i, a[i][1]))
        #print ("(%d, %d): %s" % (rowBase + 2, colBase + i, b[i][1]))
        #print ("(%d, %d): %s" % (rowBase + 3, colBase + i, c[i][1]))


if __name__ == "__main__":

    wb = xlsxwriter.Workbook("data.xlsx")
    ws = wb.add_worksheet()

    (shdma_d2d, shdma_h2d, shdma_d2h) = read_file("shdma.log")
    (cpdma_d2d, cpdma_h2d, cpdma_d2h) = read_file("cpdma.log")
    (sdma_d2d,  sdma_h2d,  sdma_d2h)  = read_file("sdma.log")

    collect_data("D2D", shdma_d2d, cpdma_d2d, sdma_d2d, 0)
    collect_data("H2D", shdma_h2d, cpdma_h2d, sdma_h2d, 5)
    collect_data("D2H", shdma_d2h, cpdma_d2h, sdma_d2h, 10)

    wb.close()
