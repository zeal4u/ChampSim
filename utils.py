#!/bin/python3
from distutils import core
import re
import os
import struct
import lzma
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import logging
from scipy.stats.mstats import gmean
from collections import defaultdict
from collections import Counter

logging.basicConfig(filename='utils.log', level=logging.WARNING)

def read_list(filename):
    l = []
    with open(filename, 'r') as cl:
        for line in cl:
            l.append(line.strip())
    return l


cloudsuites = read_list("./trace_list/cloudsuits_list")

# single core traces
dpc3_traces = read_list("./trace_list/dpc3_traces_list")
dpc3_4xx_traces = read_list("./trace_list/dpc3_4xx_list")
spec_total = read_list("./trace_list/spec_total_list")
spec_46 = read_list("./trace_list/spec17_46_list")
bigdata_traces = read_list("./trace_list/bigdata_traces_list")
mpki_gt5_list = read_list("./trace_list/mpki_gt5_list")
mpki_gt5_list_mini = mpki_gt5_list[::3]
iprefetch_list = read_list("./trace_list/iprefetch_list")
ipc1_list = read_list("./trace_list/ipc1_list")
total_traces = spec_total + bigdata_traces + iprefetch_list + ipc1_list

mpki_gt5_sample25 = ['ligra_PageRank.com-lj.ungraph.gcc_6.3.0_O3.drop_60750M.length_250M.champsimtrace.xz',
 '459.GemsFDTD-1320B.champsimtrace.xz',
 'ligra_BFS.com-lj.ungraph.gcc_6.3.0_O3.drop_21500M.length_250M.champsimtrace.xz',
 'ligra_Radii.com-lj.ungraph.gcc_6.3.0_O3.drop_36000M.length_250M.champsimtrace.xz',
 '433.milc-274B.champsimtrace.xz',
 'ligra_Components-Shortcut.com-lj.ungraph.gcc_6.3.0_O3.drop_17000M.length_250M.champsimtrace.xz',
 'ligra_PageRank.com-lj.ungraph.gcc_6.3.0_O3.drop_79500M.length_250M.champsimtrace.xz',
 '459.GemsFDTD-1169B.champsimtrace.xz',
 'ligra_PageRank.com-lj.ungraph.gcc_6.3.0_O3.drop_18500M.length_250M.champsimtrace.xz',
 'ligra_MIS.com-lj.ungraph.gcc_6.3.0_O3.drop_21250M.length_250M.champsimtrace.xz',
 '654.roms_s-1390B.champsimtrace.xz',
 '433.milc-127B.champsimtrace.xz',
 '605.mcf_s-665B.champsimtrace.xz',
 '410.bwaves-2097B.champsimtrace.xz',
 'ligra_Components-Shortcut.com-lj.ungraph.gcc_6.3.0_O3.drop_22000M.length_250M.champsimtrace.xz',
 'parsec_2.1.streamcluster.simlarge.prebuilt.drop_14750M.length_250M.champsimtrace.xz',
 'parsec_2.1.facesim.simlarge.prebuilt.drop_21500M.length_250M.champsimtrace.xz',
 'ligra_Components.com-lj.ungraph.gcc_6.3.0_O3.drop_15750M.length_250M.champsimtrace.xz',
 'ligra_PageRank.com-lj.ungraph.gcc_6.3.0_O3.drop_51000M.length_250M.champsimtrace.xz',
 '619.lbm_s-2677B.champsimtrace.xz',
 'ligra_PageRankDelta.com-lj.ungraph.gcc_6.3.0_O3.drop_52000M.length_250M.champsimtrace.xz',
 '605.mcf_s-484B.champsimtrace.xz',
 'ligra_MIS.com-lj.ungraph.gcc_6.3.0_O3.drop_17000M.length_250M.champsimtrace.xz',
 '459.GemsFDTD-1491B.champsimtrace.xz',
 'parsec_2.1.canneal.simlarge.prebuilt.drop_3000M.length_250M.champsimtrace.xz']

# multi core traces
mpki_gt5_list_homo_4core = [[i]*4 for i in mpki_gt5_list]
mpki_gt5_list_homo_8core = [[i]*8 for i in mpki_gt5_list]
mpki_gt5_list_homo_16core = [[i]*16 for i in mpki_gt5_list]
spec_46_homo_4core = [[i]*4 for i in spec_46]

mpki_gt5_sample25_16core = [[i]*16 for i in mpki_gt5_sample25]


# mix three types traces respectively
trace_mix1_4core = [['ligra_Triangle.com-lj.ungraph.gcc_6.3.0_O3.drop_6000M.length_250M.champsimtrace.xz',
  'ligra_BC.com-lj.ungraph.gcc_6.3.0_O3.drop_15500M.length_250M.champsimtrace.xz',
  '607.cactuBSSN_s-2421B.champsimtrace.xz',
  'ligra_CF.com-lj.ungraph.gcc_6.3.0_O3.drop_184750M.length_250M.champsimtrace.xz'],
 ['603.bwaves_s-2931B.champsimtrace.xz',
  '437.leslie3d-134B.champsimtrace.xz',
  '437.leslie3d-271B.champsimtrace.xz',
  '619.lbm_s-3766B.champsimtrace.xz'],
 ['649.fotonik3d_s-8225B.champsimtrace.xz',
  'ligra_BFS.com-lj.ungraph.gcc_6.3.0_O3.drop_17000M.length_250M.champsimtrace.xz',
  '459.GemsFDTD-1169B.champsimtrace.xz',
  '459.GemsFDTD-1169B.champsimtrace.xz'],
 ['ligra_Triangle.com-lj.ungraph.gcc_6.3.0_O3.drop_6000M.length_250M.champsimtrace.xz',
  'ligra_BFS.com-lj.ungraph.gcc_6.3.0_O3.drop_17000M.length_250M.champsimtrace.xz',
  '437.leslie3d-271B.champsimtrace.xz',
  '437.leslie3d-134B.champsimtrace.xz'],
 ['ligra_BFSCC.com-lj.ungraph.gcc_6.3.0_O3.drop_15500M.length_250M.champsimtrace.xz',
  'ligra_Components-Shortcut.com-lj.ungraph.gcc_6.3.0_O3.drop_17000M.length_250M.champsimtrace.xz',
  'parsec_2.1.canneal.simlarge.prebuilt.drop_1250M.length_250M.champsimtrace.xz',
  'ligra_MIS.com-lj.ungraph.gcc_6.3.0_O3.drop_17000M.length_250M.champsimtrace.xz'],
 ['649.fotonik3d_s-1176B.champsimtrace.xz',
  '481.wrf-816B.champsimtrace.xz',
  'ligra_MIS.com-lj.ungraph.gcc_6.3.0_O3.drop_17000M.length_250M.champsimtrace.xz',
  'ligra_CF.com-lj.ungraph.gcc_6.3.0_O3.drop_184750M.length_250M.champsimtrace.xz'],
 ['parsec_2.1.streamcluster.simlarge.prebuilt.drop_0M.length_250M.champsimtrace.xz',
  'ligra_BFS.com-lj.ungraph.gcc_6.3.0_O3.drop_17000M.length_250M.champsimtrace.xz',
  '649.fotonik3d_s-1176B.champsimtrace.xz',
  'ligra_Triangle.com-lj.ungraph.gcc_6.3.0_O3.drop_6000M.length_250M.champsimtrace.xz'],
 ['459.GemsFDTD-1169B.champsimtrace.xz',
  '459.GemsFDTD-1169B.champsimtrace.xz',
  '437.leslie3d-232B.champsimtrace.xz',
  '654.roms_s-293B.champsimtrace.xz'],
 ['459.GemsFDTD-765B.champsimtrace.xz',
  '603.bwaves_s-2931B.champsimtrace.xz',
  '603.bwaves_s-2931B.champsimtrace.xz',
  '649.fotonik3d_s-8225B.champsimtrace.xz'],
 ['470.lbm-1274B.champsimtrace.xz',
  '459.GemsFDTD-1211B.champsimtrace.xz',
  'parsec_2.1.canneal.simlarge.prebuilt.drop_1250M.length_250M.champsimtrace.xz',
  'ligra_BC.com-lj.ungraph.gcc_6.3.0_O3.drop_15500M.length_250M.champsimtrace.xz'],
 ['parsec_2.1.facesim.simlarge.prebuilt.drop_21500M.length_250M.champsimtrace.xz',
  '471.omnetpp-188B.champsimtrace.xz',
  'ligra_PageRank.com-lj.ungraph.gcc_6.3.0_O3.drop_60750M.length_250M.champsimtrace.xz',
  'ligra_Components.com-lj.ungraph.gcc_6.3.0_O3.drop_23750M.length_250M.champsimtrace.xz'],
 ['ligra_BC.com-lj.ungraph.gcc_6.3.0_O3.drop_15750M.length_250M.champsimtrace.xz',
  '433.milc-127B.champsimtrace.xz',
  '482.sphinx3-1297B.champsimtrace.xz',
  'ligra_PageRank.com-lj.ungraph.gcc_6.3.0_O3.drop_18500M.length_250M.champsimtrace.xz'],
 ['ligra_BFS.com-lj.ungraph.gcc_6.3.0_O3.drop_15750M.length_250M.champsimtrace.xz',
  '654.roms_s-1390B.champsimtrace.xz',
  'ligra_PageRank.com-lj.ungraph.gcc_6.3.0_O3.drop_18500M.length_250M.champsimtrace.xz',
  '471.omnetpp-188B.champsimtrace.xz'],
 ['605.mcf_s-994B.champsimtrace.xz',
  '471.omnetpp-188B.champsimtrace.xz',
  'ligra_CF.com-lj.ungraph.gcc_6.3.0_O3.drop_154750M.length_250M.champsimtrace.xz',
  '623.xalancbmk_s-165B.champsimtrace.xz'],
 ['410.bwaves-1963B.champsimtrace.xz',
  'parsec_2.1.canneal.simlarge.prebuilt.drop_3000M.length_250M.champsimtrace.xz',
  '602.gcc_s-1850B.champsimtrace.xz',
  'ligra_PageRank.com-lj.ungraph.gcc_6.3.0_O3.drop_51000M.length_250M.champsimtrace.xz'],
 ['ligra_BC.com-lj.ungraph.gcc_6.3.0_O3.drop_15750M.length_250M.champsimtrace.xz',
  '482.sphinx3-1395B.champsimtrace.xz',
  '654.roms_s-523B.champsimtrace.xz',
  'parsec_2.1.canneal.simlarge.prebuilt.drop_3000M.length_250M.champsimtrace.xz'],
 ['654.roms_s-294B.champsimtrace.xz',
  'ligra_PageRank.com-lj.ungraph.gcc_6.3.0_O3.drop_51000M.length_250M.champsimtrace.xz',
  '654.roms_s-523B.champsimtrace.xz',
  '482.sphinx3-417B.champsimtrace.xz'],
 ['parsec_2.1.streamcluster.simlarge.prebuilt.drop_250M.length_250M.champsimtrace.xz',
  'parsec_2.1.streamcluster.simlarge.prebuilt.drop_14750M.length_250M.champsimtrace.xz',
  'ligra_PageRank.com-lj.ungraph.gcc_6.3.0_O3.drop_60750M.length_250M.champsimtrace.xz',
  'ligra_PageRankDelta.com-lj.ungraph.gcc_6.3.0_O3.drop_24000M.length_250M.champsimtrace.xz'],
 ['parsec_2.1.streamcluster.simlarge.prebuilt.drop_4750M.length_250M.champsimtrace.xz',
  'ligra_PageRankDelta.com-lj.ungraph.gcc_6.3.0_O3.drop_52000M.length_250M.champsimtrace.xz',
  'parsec_2.1.facesim.simlarge.prebuilt.drop_21500M.length_250M.champsimtrace.xz',
  'ligra_BFSCC.com-lj.ungraph.gcc_6.3.0_O3.drop_18750M.length_250M.champsimtrace.xz'],
 ['parsec_2.1.streamcluster.simlarge.prebuilt.drop_250M.length_250M.champsimtrace.xz',
  'ligra_BellmanFord.com-lj.ungraph.gcc_6.3.0_O3.drop_33750M.length_250M.champsimtrace.xz',
  '433.milc-274B.champsimtrace.xz',
  '482.sphinx3-1395B.champsimtrace.xz'],
 ['ligra_Radii.com-lj.ungraph.gcc_6.3.0_O3.drop_25000M.length_250M.champsimtrace.xz',
  'ligra_PageRank.com-lj.ungraph.gcc_6.3.0_O3.drop_79500M.length_250M.champsimtrace.xz',
  '605.mcf_s-1554B.champsimtrace.xz',
  '605.mcf_s-472B.champsimtrace.xz'],
 ['429.mcf-192B.champsimtrace.xz',
  '429.mcf-184B.champsimtrace.xz',
  'ligra_Radii.com-lj.ungraph.gcc_6.3.0_O3.drop_32000M.length_250M.champsimtrace.xz',
  '605.mcf_s-484B.champsimtrace.xz'],
 ['483.xalancbmk-127B.champsimtrace.xz',
  'ligra_Radii.com-lj.ungraph.gcc_6.3.0_O3.drop_32000M.length_250M.champsimtrace.xz',
  'ligra_Components-Shortcut.com-lj.ungraph.gcc_6.3.0_O3.drop_22000M.length_250M.champsimtrace.xz',
  'ligra_BFSCC.com-lj.ungraph.gcc_6.3.0_O3.drop_22000M.length_250M.champsimtrace.xz'],
 ['605.mcf_s-1644B.champsimtrace.xz',
  '602.gcc_s-2226B.champsimtrace.xz',
  'ligra_Components-Shortcut.com-lj.ungraph.gcc_6.3.0_O3.drop_22000M.length_250M.champsimtrace.xz',
  '429.mcf-184B.champsimtrace.xz'],
 ['605.mcf_s-1554B.champsimtrace.xz',
  '429.mcf-192B.champsimtrace.xz',
  '605.mcf_s-1554B.champsimtrace.xz',
  '605.mcf_s-1152B.champsimtrace.xz'],
 ['ligra_Components-Shortcut.com-lj.ungraph.gcc_6.3.0_O3.drop_22000M.length_250M.champsimtrace.xz',
  '605.mcf_s-1554B.champsimtrace.xz',
  'ligra_BFSCC.com-lj.ungraph.gcc_6.3.0_O3.drop_22000M.length_250M.champsimtrace.xz',
  '483.xalancbmk-127B.champsimtrace.xz'],
 ['429.mcf-217B.champsimtrace.xz',
  'ligra_PageRank.com-lj.ungraph.gcc_6.3.0_O3.drop_21750M.length_250M.champsimtrace.xz',
  '605.mcf_s-484B.champsimtrace.xz',
  '605.mcf_s-484B.champsimtrace.xz'],
 ['ligra_BFSCC.com-lj.ungraph.gcc_6.3.0_O3.drop_22000M.length_250M.champsimtrace.xz',
  '605.mcf_s-484B.champsimtrace.xz',
  'ligra_PageRank.com-lj.ungraph.gcc_6.3.0_O3.drop_21750M.length_250M.champsimtrace.xz',
  '605.mcf_s-472B.champsimtrace.xz'],
 ['ligra_BFS.com-lj.ungraph.gcc_6.3.0_O3.drop_20250M.length_250M.champsimtrace.xz',
  '483.xalancbmk-127B.champsimtrace.xz',
  '429.mcf-51B.champsimtrace.xz',
  '623.xalancbmk_s-10B.champsimtrace.xz'],
 ['605.mcf_s-1152B.champsimtrace.xz',
  '429.mcf-22B.champsimtrace.xz',
  'ligra_PageRank.com-lj.ungraph.gcc_6.3.0_O3.drop_79500M.length_250M.champsimtrace.xz',
  'ligra_Components.com-lj.ungraph.gcc_6.3.0_O3.drop_22750M.length_250M.champsimtrace.xz']]

trace_mix2_4core = [['ligra_BFSCC.com-lj.ungraph.gcc_6.3.0_O3.drop_15500M.length_250M.champsimtrace.xz',
  '649.fotonik3d_s-1176B.champsimtrace.xz',
  '654.roms_s-1070B.champsimtrace.xz',
  'ligra_PageRankDelta.com-lj.ungraph.gcc_6.3.0_O3.drop_24000M.length_250M.champsimtrace.xz'],
 ['ligra_PageRankDelta.com-lj.ungraph.gcc_6.3.0_O3.drop_17000M.length_250M.champsimtrace.xz',
  'ligra_BFS.com-lj.ungraph.gcc_6.3.0_O3.drop_17000M.length_250M.champsimtrace.xz',
  'ligra_BFS.com-lj.ungraph.gcc_6.3.0_O3.drop_15750M.length_250M.champsimtrace.xz',
  '603.bwaves_s-2609B.champsimtrace.xz'],
 ['ligra_MIS.com-lj.ungraph.gcc_6.3.0_O3.drop_17000M.length_250M.champsimtrace.xz',
  '459.GemsFDTD-1211B.champsimtrace.xz',
  'ligra_BC.com-lj.ungraph.gcc_6.3.0_O3.drop_15750M.length_250M.champsimtrace.xz',
  '619.lbm_s-4268B.champsimtrace.xz'],
 ['459.GemsFDTD-1418B.champsimtrace.xz',
  '621.wrf_s-8065B.champsimtrace.xz',
  'parsec_2.1.facesim.simlarge.prebuilt.drop_21500M.length_250M.champsimtrace.xz',
  '654.roms_s-523B.champsimtrace.xz'],
 ['654.roms_s-293B.champsimtrace.xz',
  '459.GemsFDTD-765B.champsimtrace.xz',
  'ligra_BFS.com-lj.ungraph.gcc_6.3.0_O3.drop_21500M.length_250M.champsimtrace.xz',
  '482.sphinx3-234B.champsimtrace.xz'],
 ['ligra_BFS.com-lj.ungraph.gcc_6.3.0_O3.drop_17000M.length_250M.champsimtrace.xz',
  'ligra_BFS.com-lj.ungraph.gcc_6.3.0_O3.drop_17000M.length_250M.champsimtrace.xz',
  '459.GemsFDTD-1320B.champsimtrace.xz',
  'ligra_Components.com-lj.ungraph.gcc_6.3.0_O3.drop_15750M.length_250M.champsimtrace.xz'],
 ['ligra_PageRankDelta.com-lj.ungraph.gcc_6.3.0_O3.drop_6000M.length_250M.champsimtrace.xz',
  '607.cactuBSSN_s-2421B.champsimtrace.xz',
  'ligra_PageRank.com-lj.ungraph.gcc_6.3.0_O3.drop_60750M.length_250M.champsimtrace.xz',
  '654.roms_s-523B.champsimtrace.xz'],
 ['ligra_CF.com-lj.ungraph.gcc_6.3.0_O3.drop_184750M.length_250M.champsimtrace.xz',
  '621.wrf_s-6673B.champsimtrace.xz',
  'parsec_2.1.facesim.simlarge.prebuilt.drop_21500M.length_250M.champsimtrace.xz',
  'ligra_PageRank.com-lj.ungraph.gcc_6.3.0_O3.drop_18500M.length_250M.champsimtrace.xz'],
 ['481.wrf-455B.champsimtrace.xz',
  '459.GemsFDTD-1418B.champsimtrace.xz',
  '450.soplex-92B.champsimtrace.xz',
  'ligra_Components.com-lj.ungraph.gcc_6.3.0_O3.drop_15750M.length_250M.champsimtrace.xz'],
 ['459.GemsFDTD-1211B.champsimtrace.xz',
  '649.fotonik3d_s-1176B.champsimtrace.xz',
  'ligra_Radii.com-lj.ungraph.gcc_6.3.0_O3.drop_24500M.length_250M.champsimtrace.xz',
  '482.sphinx3-1297B.champsimtrace.xz'],
 ['ligra_MIS.com-lj.ungraph.gcc_6.3.0_O3.drop_21250M.length_250M.champsimtrace.xz',
  '437.leslie3d-134B.champsimtrace.xz',
  'ligra_Components-Shortcut.com-lj.ungraph.gcc_6.3.0_O3.drop_22000M.length_250M.champsimtrace.xz',
  'ligra_Radii.com-lj.ungraph.gcc_6.3.0_O3.drop_36000M.length_250M.champsimtrace.xz'],
 ['437.leslie3d-134B.champsimtrace.xz',
  '621.wrf_s-6673B.champsimtrace.xz',
  '459.GemsFDTD-1491B.champsimtrace.xz',
  'ligra_BFSCC.com-lj.ungraph.gcc_6.3.0_O3.drop_22000M.length_250M.champsimtrace.xz'],
 ['459.GemsFDTD-1211B.champsimtrace.xz',
  '621.wrf_s-6673B.champsimtrace.xz',
  '605.mcf_s-472B.champsimtrace.xz',
  'ligra_Radii.com-lj.ungraph.gcc_6.3.0_O3.drop_36000M.length_250M.champsimtrace.xz'],
 ['619.lbm_s-3766B.champsimtrace.xz',
  'parsec_2.1.streamcluster.simlarge.prebuilt.drop_0M.length_250M.champsimtrace.xz',
  '605.mcf_s-782B.champsimtrace.xz',
  '623.xalancbmk_s-10B.champsimtrace.xz'],
 ['ligra_PageRankDelta.com-lj.ungraph.gcc_6.3.0_O3.drop_17000M.length_250M.champsimtrace.xz',
  'ligra_CF.com-lj.ungraph.gcc_6.3.0_O3.drop_184750M.length_250M.champsimtrace.xz',
  '462.libquantum-714B.champsimtrace.xz',
  '605.mcf_s-472B.champsimtrace.xz'],
 ['481.wrf-1281B.champsimtrace.xz',
  'ligra_PageRankDelta.com-lj.ungraph.gcc_6.3.0_O3.drop_17000M.length_250M.champsimtrace.xz',
  '605.mcf_s-1644B.champsimtrace.xz',
  '605.mcf_s-782B.champsimtrace.xz'],
 ['649.fotonik3d_s-1176B.champsimtrace.xz',
  'ligra_Triangle.com-lj.ungraph.gcc_6.3.0_O3.drop_6000M.length_250M.champsimtrace.xz',
  '429.mcf-51B.champsimtrace.xz',
  '605.mcf_s-472B.champsimtrace.xz'],
 ['ligra_PageRankDelta.com-lj.ungraph.gcc_6.3.0_O3.drop_6000M.length_250M.champsimtrace.xz',
  'ligra_MIS.com-lj.ungraph.gcc_6.3.0_O3.drop_21250M.length_250M.champsimtrace.xz',
  '605.mcf_s-484B.champsimtrace.xz',
  '483.xalancbmk-127B.champsimtrace.xz'],
 ['459.GemsFDTD-1211B.champsimtrace.xz',
  '470.lbm-1274B.champsimtrace.xz',
  '459.GemsFDTD-1491B.champsimtrace.xz',
  '429.mcf-217B.champsimtrace.xz'],
 ['437.leslie3d-273B.champsimtrace.xz',
  '437.leslie3d-134B.champsimtrace.xz',
  '602.gcc_s-2226B.champsimtrace.xz',
  'ligra_Components.com-lj.ungraph.gcc_6.3.0_O3.drop_22750M.length_250M.champsimtrace.xz'],
 ['603.bwaves_s-1740B.champsimtrace.xz',
  '602.gcc_s-1850B.champsimtrace.xz',
  '605.mcf_s-782B.champsimtrace.xz',
  '429.mcf-184B.champsimtrace.xz'],
 ['ligra_BFS.com-lj.ungraph.gcc_6.3.0_O3.drop_15750M.length_250M.champsimtrace.xz',
  'ligra_BC.com-lj.ungraph.gcc_6.3.0_O3.drop_15750M.length_250M.champsimtrace.xz',
  'ligra_PageRank.com-lj.ungraph.gcc_6.3.0_O3.drop_79500M.length_250M.champsimtrace.xz',
  'ligra_Components-Shortcut.com-lj.ungraph.gcc_6.3.0_O3.drop_22000M.length_250M.champsimtrace.xz'],
 ['603.bwaves_s-1740B.champsimtrace.xz',
  '433.milc-127B.champsimtrace.xz',
  'ligra_Components.com-lj.ungraph.gcc_6.3.0_O3.drop_22750M.length_250M.champsimtrace.xz',
  '605.mcf_s-484B.champsimtrace.xz'],
 ['ligra_PageRank.com-lj.ungraph.gcc_6.3.0_O3.drop_60750M.length_250M.champsimtrace.xz',
  'parsec_2.1.streamcluster.simlarge.prebuilt.drop_4750M.length_250M.champsimtrace.xz',
  'ligra_Components-Shortcut.com-lj.ungraph.gcc_6.3.0_O3.drop_22000M.length_250M.champsimtrace.xz',
  '605.mcf_s-782B.champsimtrace.xz'],
 ['ligra_BFS.com-lj.ungraph.gcc_6.3.0_O3.drop_21500M.length_250M.champsimtrace.xz',
  '482.sphinx3-1522B.champsimtrace.xz',
  'ligra_BC.com-lj.ungraph.gcc_6.3.0_O3.drop_26750M.length_250M.champsimtrace.xz',
  'ligra_BFSCC.com-lj.ungraph.gcc_6.3.0_O3.drop_22000M.length_250M.champsimtrace.xz'],
 ['ligra_BFS.com-lj.ungraph.gcc_6.3.0_O3.drop_21500M.length_250M.champsimtrace.xz',
  'ligra_BFSCC.com-lj.ungraph.gcc_6.3.0_O3.drop_18750M.length_250M.champsimtrace.xz',
  '429.mcf-51B.champsimtrace.xz',
  '459.GemsFDTD-1491B.champsimtrace.xz'],
 ['482.sphinx3-1100B.champsimtrace.xz',
  '410.bwaves-2097B.champsimtrace.xz',
  'ligra_Components.com-lj.ungraph.gcc_6.3.0_O3.drop_22750M.length_250M.champsimtrace.xz',
  '483.xalancbmk-127B.champsimtrace.xz'],
 ['654.roms_s-1390B.champsimtrace.xz',
  'ligra_PageRankDelta.com-lj.ungraph.gcc_6.3.0_O3.drop_35250M.length_250M.champsimtrace.xz',
  '462.libquantum-714B.champsimtrace.xz',
  '602.gcc_s-2226B.champsimtrace.xz'],
 ['482.sphinx3-417B.champsimtrace.xz',
  '654.roms_s-523B.champsimtrace.xz',
  'ligra_PageRank.com-lj.ungraph.gcc_6.3.0_O3.drop_21750M.length_250M.champsimtrace.xz',
  '429.mcf-22B.champsimtrace.xz'],
 ['parsec_2.1.streamcluster.simlarge.prebuilt.drop_4750M.length_250M.champsimtrace.xz',
  '654.roms_s-1070B.champsimtrace.xz',
  '429.mcf-22B.champsimtrace.xz',
  'ligra_PageRank.com-lj.ungraph.gcc_6.3.0_O3.drop_21750M.length_250M.champsimtrace.xz']]

# We filter the traces where bingo(LLC) improves at least 5% on them.
spec_prefetch_sensitive = read_list("./trace_list/spec_prefetch_sensitive")
bigdata_prefetch_sensitive = read_list("./trace_list/bigdata_prefetch_sensitive")
prefetch_sensitive_traces = spec_prefetch_sensitive + bigdata_prefetch_sensitive



BASE_LINE_MODEL="bimodal-no_instr-no-no-no-lru-1core"

def result_file_filter(complete_model_name, filename):
    no = filename[0:filename.find('-bimodal')]
    return filename.replace('.txt.xz', '').replace('.txt', '').endswith(complete_model_name), no

def read_addresses(content):
    res = set()
    r = content.read(8)
    while r:
        res.add(struct.unpack("<Q", r)[0])
        r = content.read(8)
    return res


def read_repeated_addresses(content):
    res = {}
    r = content.read(8)
    while r:
        ad = struct.unpack("<Q", r)[0]
        if ad in res:
            res[ad] += 1
        else:
            res[ad] = 1
        r = content.read(8)
    return res


def coarse_accuracy(filename):
    count = 0
    with lzma.open("./prefetch_traces/"+filename, "rb") as prefetch_trace, \
            lzma.open("./addr_traces/"+filename, "rb") as trace:
        prefetches = read_addresses(prefetch_trace)
        r = trace.read(8)
        total = 0
        while r:
            a = (struct.unpack("<Q", r)[0])
            a &= ~(1 << 63)
            count += a in prefetches
            total += 1
            r = trace.read(8)
        return count/total


def count_bits(num, size=64):
    count = 0
    for i in range(size):
        count += 1 & (num >> i)
    return count


def analyze_miss_pattern(patterns):
    rates = {}
    for a in patterns:
        one = patterns[a]
        for o, p, m in one:
            rate = count_bits(m) / count_bits(p)
            if rate > 0:
                if a in rates:
                    rates[a].append(rate)
                else:
                    rates[a] = [rate]
    return rates


def extract_ipc(content):
    pattern = re.compile(
        r"Finished CPU [0-9]+ instructions: [0-9]+ cycles: [0-9]+ cumulative IPC: ([./0-9]+) ")
    result = pattern.findall(content)
    if result:
        ipcs = [float(r) for r in result]
        result = {"IPC": gmean(ipcs)}
    return result


def show_avg_status(models, workloads, target_dir, func):
    datas = {}
    for model in models:
        datas[model] = (transform_to_df(
            func(model, workloads, target_dir))).mean()
    return transform_to_df(datas)


def get_repeat_loc(trace):
    count = set()
    repeat_trace = []
    non_repeat_trace = []
    for i, a in trace:
        if a not in count:
            count.add(a)
            non_repeat_trace.append((i, a))
        else:
            repeat_trace.append((i, a))
    return non_repeat_trace, repeat_trace


def get_repeat(trace):
    count = set()
    repeat_trace = []
    non_repeat_trace = []
    for a in trace:
        if a not in count:
            count.add(a)
            non_repeat_trace.append(a)
        else:
            repeat_trace.append(a)
    return non_repeat_trace, repeat_trace


def draw_addr_trace(filename):
    result = []
    with lzma.open(filename, "rb") as file:
        r = file.read(8)
        while r:
            result.append(struct.unpack("<Q", r)[0])
            r = file.read(8)
    hit_trace = list(filter(lambda x: not x[1] >> 63, enumerate(result)))
    hit_trace, repeat_hit_trace = get_repeat_loc(hit_trace)
    miss_trace = list(filter(lambda x: x[1] >> 63, enumerate(result)))
    miss_trace = list(map(lambda x: (x[0], x[1] & ~(1 << 63)), miss_trace))
    miss_trace, repeat_miss_trace = get_repeat_loc(miss_trace)
    fig, axs = plt.subplots(1, 2)
    axs[0].scatter(list(map(lambda x: x[0], hit_trace)),
                   list(map(lambda x: x[1], hit_trace)), s=10, label='hit', marker=">")
    axs[0].scatter(list(map(lambda x: x[0], miss_trace)),
                   list(map(lambda x: x[1], miss_trace)), s=10, label='miss', marker="*")
    axs[0].scatter(list(map(lambda x: x[0], repeat_hit_trace)),
                   list(map(lambda x: x[1], repeat_hit_trace)), s=10, label='hit repeat', marker="+")
    axs[0].scatter(list(map(lambda x: x[0], repeat_miss_trace)),
                   list(map(lambda x: x[1], repeat_miss_trace)), s=10, label='miss repeat', marker=".")
    axs[0].set_ylabel("addr")
    axs[0].set_xlabel("time")
    axs[0].set_title(filename)
    axs[0].legend(ncol=1, bbox_to_anchor=(1, 0), loc='lower left')

    def func(pct, allvals):
        absolute = int(round(pct/100.*np.sum(allvals)))
        return "{:.1f}%\n({:d})".format(pct, absolute)

    pie_data = [len(hit_trace), len(miss_trace), len(repeat_hit_trace),
                len(repeat_miss_trace)]
    wedges, texts, autotexts = axs[1].pie(pie_data,
                                          autopct=lambda pct: func(
                                              pct, pie_data),
                                          textprops=dict(color="w"))
    classes = ['hit', 'miss', 'hit repeat', 'miss repeat']
    axs[1].legend(wedges, classes,
                  loc="center left",
                  bbox_to_anchor=(1, 0, 0.5, 1))
    fig.set_size_inches(20, 10)


def get_addr_analyze(filename):
    result = []
    with lzma.open(filename, "rb") as file:
        r = file.read(8)
        while r:
            result.append(struct.unpack("<Q", r)[0])
            r = file.read(8)
    hit_trace = list(filter(lambda x: not x >> 63, result))
    hit_trace, repeat_hit_trace = get_repeat(hit_trace)
    miss_trace = list(filter(lambda x: x >> 63, result))
    miss_trace = list(map(lambda x: x & ~(1 << 63), miss_trace))
    miss_trace, repeat_miss_trace = get_repeat(miss_trace)
    data = [result, hit_trace, miss_trace,
            repeat_hit_trace, repeat_miss_trace]
    names = [
        "total access number",
        "nonrecurrence hit number",
        "nonrecurrence miss number",
        "recurrence hit number",
        "recurrence miss number",
        "total distinct access addresses",
        "distinct recurrence hit addresses",
        "distinct recurrence miss addresses",
    ]
    res = list(map(len, data)) + \
        list(map(lambda x: len(set(x)), data[0:1]+data[3:]))
    return {names[i]: res[i] for i in range(8)}



def hash_index(key, index_len):
    if index_len == 0:
        return key
    tag = key >> index_len
    while tag > 0:
        key ^= tag & ((1 << index_len) - 1)
        tag = tag >> index_len
    return key


def draw_trigger_offsets_layout(data):
    fig, axs = plt.subplots(46)
    x = np.arange(256)
    for i, d in enumerate(spec_46):
        k = '.'.join(d.split('.')[0:2])
        axs[i].set_title(k)
        axs[i].bar(x, data[k]['Offsets'])
    fig.set_size_inches(16, 5*46)


def prefetch_match(filename):
    count = 0
    with lzma.open("./prefetch_traces/"+filename, "rb") as prefetch_trace, \
            lzma.open("./addr_traces/"+filename, "rb") as trace:
        prefetches = read_repeated_addresses(prefetch_trace)
        r = trace.read(8)
        total = 0
        prefetch_total = sum(prefetches.values())
        while r:
            a = (struct.unpack("<Q", r)[0])
            a &= ~(1 << 63)
            if a in prefetches and prefetches[a] > 0:
                count += 1
                prefetches[a] -= 1
            total += 1
            r = trace.read(8)
        return count/total, prefetch_total/total


def miss_match(filename):
    new_miss, miss, cover = 0, 0, 0
    with lzma.open("./addr_traces/"+filename, "rb") as trace,\
            lzma.open("./prefetch_miss_traces/"+filename, "rb") as new_miss_trace:
        prefetch_results = read_repeated_addresses(new_miss_trace)
        r = trace.read(8)
        origin_total = 0
        while r:
            a = (struct.unpack("<Q", r)[0])
            if a & (1 << 63):
                origin_total += 1
                if a in prefetch_results and prefetch_results[a] > 0:
                    miss += 1
                    prefetch_results[a] -= 1
            r = trace.read(8)
        cover = origin_total - miss
        new_miss = sum(
            dict(filter(lambda x: x[0] & (1 << 63), prefetch_results.items())).values())
        return {"Remained Miss": miss, "Covered Miss": cover, "New Miss": new_miss}


def extract_items(content: str, regStr: str, key: str):
    pattern = re.compile(regStr)
    regObj = pattern.findall(content)
    values = [i for i in regObj]
    if len(values) == 1:
        values = values[0]
    return {key: values} if regObj else None


def transform_to_number(data): 
    results = defaultdict(dict)
    def _num_clear(num:str):
        try:
            if '%' in num:
                return float(num.strip('% '))/100
            else:
                return float(num)
        except ValueError as e:
            return num.strip()
    for i in data:
        for k in data[i]:
            try:
                if isinstance(data[i][k], list):
                    results[i][k] = [_num_clear(j) for j in data[i][k]]
                else:
                    results[i][k] = _num_clear(data[i][k])
            except TypeError as e:
                logging.error(f"i:{i}, k:{k}, value:{data[i][k]}")
                raise e
    return results


def transform_to_df(stat_data):
    keys = list(stat_data.keys())
    table_data = {}
    for key in stat_data[keys[0]].keys():
        table_data[key] = []
    for i in stat_data:
        d = stat_data[i]
        for key in table_data:
            try:
                if '%' in d[key]:
                    table_data[key].append(float(d[key].strip('% '))/100)
                else:
                    table_data[key].append(float(d[key]))
            except Exception as e:
                logging.error(e, key, d[key])
                raise e
    df = pd.DataFrame(table_data, index=stat_data.keys())
    return df


def extracter(patternPairs: list):
    def decorator(fun):
        def func(content: str):
            result = {}
            for pair in patternPairs:
                findValue = extract_items(content, *pair)
                if findValue:
                    result.update(findValue)
                else:
                    logging.warning(f"Can't find for {pair}")
            return result
        return func
    return decorator


def convert_format_data(stats):
    result = {}
    for k in stats:
        tmp = {}
        for line in stats[k]['Stats'].splitlines():
            words = line.strip().split(':')
            try:
                tmp[words[0]] = float(words[1])
            except:
                for i, v in enumerate(words[1].strip().split()):
                    tmp['Offset ' + str(i)] = int(v)
        result[k] = tmp
    return result


# # 获取模拟器默认的所有统计数字
# def extract_default_status(content):
#     pattern = re.compile(
#         r"L1D PREFETCH  REQUESTED:([\s0-9]*)ISSUED:([\s0-9]*)USEFUL:([\s0-9]*)USELESS:([\s0-9]*)")
#     regObj = pattern.search(content)
#     pattern1 = re.compile(
#         r"L2C PREFETCH  REQUESTED:([\s0-9]*)ISSUED:([\s0-9]*)USEFUL:([\s0-9]*)USELESS:([\s0-9]*)")
#     regObj1 = pattern1.search(content)
#     pattern2 = re.compile(
#         r"LLC PREFETCH  REQUESTED:([\s0-9]*)ISSUED:([\s0-9]*)USEFUL:([\s0-9]*)USELESS:([\s0-9]*)")
#     regObj2 = pattern2.search(content)

#     pattern3 = re.compile(
#         r"L1D PREFETCH[\s]+ACCESS:[\s]+([0-9]+)[\s]+HIT:[\s]+([0-9]+)[\s]+MISS:[\s]+([0-9]+)")
#     pattern4 = re.compile(
#         r"L2C PREFETCH[\s]+ACCESS:[\s]+([0-9]+)[\s]+HIT:[\s]+([0-9]+)[\s]+MISS:[\s]+([0-9]+)")
#     pattern5 = re.compile(
#         r"LLC PREFETCH[\s]+ACCESS:[\s]+([0-9]+)[\s]+HIT:[\s]+([0-9]+)[\s]+MISS:[\s]+([0-9]+)")
#     regObj3 = pattern3.search(content)
#     regObj4 = pattern4.search(content)
#     regObj5 = pattern5.search(content)
#     result = None
#     if regObj and regObj1 and regObj2:
#         result = {
#             "L1D Prefetch Requested": int(regObj.group(1).strip()),
#             "L1D Prefetch Issued": int(regObj.group(2).strip()),
#             "L1D Prefetch Useful": int(regObj.group(3).strip()),
#             "L1D Prefetch Useless": int(regObj.group(4).strip()),

#             "L2C Prefetch Requested": int(regObj1.group(1).strip()),
#             "L2C Prefetch Issued": int(regObj1.group(2).strip()),
#             "L2C Prefetch Useful": int(regObj1.group(3).strip()),
#             "L2C Prefetch Useless": int(regObj1.group(4).strip()),

#             "LLC Prefetch Requested": int(regObj2.group(1).strip()),
#             "LLC Prefetch Issued": int(regObj2.group(2).strip()),
#             "LLC Prefetch Useful": int(regObj2.group(3).strip()),
#             "LLC Prefetch Useless": int(regObj2.group(4).strip()),

#             "L1D Prefetch Access": int(regObj3.group(1).strip()),
#             "L1D Prefetch Hit": int(regObj3.group(2).strip()),
#             "L1D Prefetch Miss": int(regObj3.group(3).strip()),

#             "L2C Prefetch Access": int(regObj4.group(1).strip()),
#             "L2C Prefetch Hit": int(regObj4.group(2).strip()),
#             "L2C Prefetch Miss": int(regObj4.group(3).strip()),

#             "LLC Prefetch Access": int(regObj5.group(1).strip()),
#             "LLC Prefetch Hit": int(regObj5.group(2).strip()),
#             "LLC Prefetch Miss": int(regObj5.group(3).strip()),
#         }
#     return result


def extract_WB_stall(content):
    pattern = re.compile(r"L1D WB Stall Cycle:([\s0-9]*)")

    regObj = pattern.search(content)
    result = None
    if regObj:
        result = {'L1D WB Stall Cycle': int(regObj.group(1).strip())}
    return result


def extract_fault_stats(content):
    pattern = re.compile(r"Major fault:([0-9\s]*)Minor fault:([0-9\s]*)")
    regObj = pattern.search(content)
    result = None
    if regObj:
        result = {
            "Major fault": int(regObj.group(1).strip()),
            "Minor fault": int(regObj.group(2).strip()),
        }
    return result


def extract_timeliness_stat(content):
    pattern = re.compile(
        r"total_filled:([0-9\s]*)\ntotal_useful:([0-9\s]*)\ntotal_late:([0-9\s]*)\n")
    regObj = pattern.search(content)
    result = None
    if regObj:
        result = {
            "total_filled": int(regObj.group(1).strip()),
            "total_useful": int(regObj.group(2).strip()),
            "total_late": int(regObj.group(3).strip())
        }
    return result


def extract_cache_miss_latency(content):
    pattern1 = re.compile(r"L1D AVERAGE MISS LATENCY:([\.0-9\s]*)cycles")
    pattern2 = re.compile(r"L1I AVERAGE MISS LATENCY:([\.0-9\s]*)cycles")
    pattern3 = re.compile(r"L2C AVERAGE MISS LATENCY:([\.0-9\s]*)cycles")
    pattern4 = re.compile(r"LLC AVERAGE MISS LATENCY:([\.0-9\s]*)cycles")

    regObj1 = pattern1.search(content)
    regObj2 = pattern2.search(content)
    regObj3 = pattern3.search(content)
    regObj4 = pattern4.search(content)

    result = None
#     print(regObj1, regObj2, regObj3, regObj4)
    if regObj1 and regObj3 and regObj4:
        result = {
            "L1D AVERAGE MISS LATENCY": float(regObj1.group(1).strip()),
            "L1I AVERAGE MISS LATENCY": float(regObj2.group(1).strip()) if regObj2 else -1,
            "L2C AVERAGE MISS LATENCY": float(regObj3.group(1).strip()),
            "LLC AVERAGE MISS LATENCY": float(regObj4.group(1).strip())
        }
    return result


appendixPatterns = [
    [r'Warmup Instructions: ([\.0-9\s]*)\n', 'Warmup Instructions'],
    [r'Simulation Instructions: ([\.0-9\s]*)\n', 'Simulation Instructions'],
    [r'Number of CPUs: ([\.0-9\s]*)\n', 'Number of CPUs'],
    [r'LLC sets: ([\.0-9\s]*)\n', 'LLC sets'],
    [r'LLC ways: ([\.0-9\s]*)\n', 'LLC ways'],
    [r'Off-chip DRAM Size: ([\.0-9\s]*) MB', 'Off-chip DRAM Size'],
    [r'Channels: ([\.0-9\s]*)', 'Channels'],
    [r'Width: ([\.0-9\s]*)-bit', 'Width'],
    [r'Data Rate: ([\.0-9\s]*) MT/s', 'Data Rate'],
]

ipcPattern = [
    [r"Finished CPU[0-9\s]*instructions:[0-9\s]*cycles:[0-9\s]*cumulative IPC:([\.0-9\s]*)", 'cumulative IPC']
]

cacheStatPatterns = [
    [r'L1D TOTAL     ACCESS:([\.0-9\s]*) HIT:[\.0-9\s]* MISS:[\.0-9\s]*',
     'L1D TOTAL ACCESS'],
    [r'L1D TOTAL     ACCESS:[\.0-9\s]* HIT:([\.0-9\s]*) MISS:[\.0-9\s]*',
     'L1D TOTAL HIT'],
    [r'L1D TOTAL     ACCESS:[\.0-9\s]* HIT:[\.0-9\s]* MISS:([\.0-9\s]*)', 'L1D TOTAL MISS'],
    [r'L1D LOAD      ACCESS:([\.0-9\s]*) HIT:[\.0-9\s]* MISS:[\.0-9\s]*',
     'L1D LOAD ACCESS'],
    [r'L1D LOAD      ACCESS:[\.0-9\s]* HIT:([\.0-9\s]*) MISS:[\.0-9\s]*',
     'L1D LOAD HIT'],
    [r'L1D LOAD      ACCESS:[\.0-9\s]* HIT:[\.0-9\s]* MISS:([\.0-9\s]*)', 'L1D LOAD MISS'],
    [r'L1D RFO       ACCESS:([\.0-9\s]*) HIT:[\.0-9\s]* MISS:[\.0-9\s]*',
     'L1D RFO ACCESS'],
    [r'L1D RFO       ACCESS:[\.0-9\s]* HIT:([\.0-9\s]*) MISS:[\.0-9\s]*',
     'L1D RFO HIT'],
    [r'L1D RFO       ACCESS:[\.0-9\s]* HIT:[\.0-9\s]* MISS:([\.0-9\s]*)', 'L1D RFO MISS'],
    [r'L1D PREFETCH  ACCESS:([\.0-9\s]*) HIT:[\.0-9\s]* MISS:[\.0-9\s]*',
     'L1D PREFETCH ACCESS'],
    [r'L1D PREFETCH  ACCESS:[\.0-9\s]* HIT:([\.0-9\s]*) MISS:[\.0-9\s]*',
     'L1D PREFETCH HIT'],
    [r'L1D PREFETCH  ACCESS:[\.0-9\s]* HIT:[\.0-9\s]* MISS:([\.0-9\s]*)',
     'L1D PREFETCH MISS'],
    [r'L1D WRITEBACK ACCESS:([\.0-9\s]*) HIT:[\.0-9\s]* MISS:[\.0-9\s]*',
     'L1D WRITEBACK ACCESS'],
    [r'L1D WRITEBACK ACCESS:[\.0-9\s]* HIT:([\.0-9\s]*) MISS:[\.0-9\s]*',
     'L1D WRITEBACK HIT'],
    [r'L1D WRITEBACK ACCESS:[\.0-9\s]* HIT:[\.0-9\s]* MISS:([\.0-9\s]*)',
     'L1D WRITEBACK MISS'],
    [r'L1D PREFETCH  REQUESTED:([\.0-9\s]*)ISSUED:[\.0-9\s]*USEFUL:[\.0-9\s]*USELESS:[\.0-9\s]*',
     'L1D PREFETCH REQUESTED'],
    [r'L1D PREFETCH  REQUESTED:[\.0-9\s]*ISSUED:([\.0-9\s]*)USEFUL:[\.0-9\s]*USELESS:[\.0-9\s]*',
     'L1D PREFETCH ISSUED'],
    [r'L1D PREFETCH  REQUESTED:[\.0-9\s]*ISSUED:[\.0-9\s]*USEFUL:([\.0-9\s]*)USELESS:[\.0-9\s]*',
     'L1D PREFETCH USEFUL'],
    [r'L1D PREFETCH  REQUESTED:[\.0-9\s]*ISSUED:[\.0-9\s]*USEFUL:[\.0-9\s]*USELESS:([\.0-9\s]*)', 'L1D PREFETCH USELESS'],
    [r'L1D AVERAGE MISS LATENCY: ([\.0-9\s]*) cycles',
     'L1D AVERAGE MISS LATENCY'],
    [r'L1D AVERAGE LOAD MISS LATENCY: ([\.0-9\s]*) cycles',
     'L1D AVERAGE LOAD MISS LATENCY'],
    [r'L1D AVERAGE RFO MISS LATENCY: ([\.0-9\s]*) cycles',
     'L1D AVERAGE RFO MISS LATENCY'],
    [r'L1D AVERAGE PREFETCH MISS LATENCY: ([\.0-9\s]*) cycles',
     'L1D AVERAGE PREFETCH MISS LATENCY'],
    [r'L1D AVERAGE WRITEBACK MISS LATENCY: ([\.0-9\s]*) cycles',
     'L1D AVERAGE WRITEBACK MISS LATENCY'],
    [r'L1I TOTAL     ACCESS:([\.0-9\s]*) HIT:[\.0-9\s]* MISS:[\.0-9\s]*',
     'L1I TOTAL ACCESS'],
    [r'L1I TOTAL     ACCESS:[\.0-9\s]* HIT:([\.0-9\s]*) MISS:[\.0-9\s]*',
     'L1I TOTAL HIT'],
    [r'L1I TOTAL     ACCESS:[\.0-9\s]* HIT:[\.0-9\s]* MISS:([\.0-9\s]*)', 'L1I TOTAL MISS'],
    [r'L1I LOAD      ACCESS:([\.0-9\s]*) HIT:[\.0-9\s]* MISS:[\.0-9\s]*',
     'L1I LOAD ACCESS'],
    [r'L1I LOAD      ACCESS:[\.0-9\s]* HIT:([\.0-9\s]*) MISS:[\.0-9\s]*',
     'L1I LOAD HIT'],
    [r'L1I LOAD      ACCESS:[\.0-9\s]* HIT:[\.0-9\s]* MISS:([\.0-9\s]*)', 'L1I LOAD MISS'],
    [r'L1I RFO       ACCESS:([\.0-9\s]*) HIT:[\.0-9\s]* MISS:[\.0-9\s]*',
     'L1I RFO ACCESS'],
    [r'L1I RFO       ACCESS:[\.0-9\s]* HIT:([\.0-9\s]*) MISS:[\.0-9\s]*',
     'L1I RFO HIT'],
    [r'L1I RFO       ACCESS:[\.0-9\s]* HIT:[\.0-9\s]* MISS:([\.0-9\s]*)', 'L1I RFO MISS'],
    [r'L1I PREFETCH  ACCESS:([\.0-9\s]*) HIT:[\.0-9\s]* MISS:[\.0-9\s]*',
     'L1I PREFETCH ACCESS'],
    [r'L1I PREFETCH  ACCESS:[\.0-9\s]* HIT:([\.0-9\s]*) MISS:[\.0-9\s]*',
     'L1I PREFETCH HIT'],
    [r'L1I PREFETCH  ACCESS:[\.0-9\s]* HIT:[\.0-9\s]* MISS:([\.0-9\s]*)',
     'L1I PREFETCH MISS'],
    [r'L1I WRITEBACK ACCESS:([\.0-9\s]*) HIT:[\.0-9\s]* MISS:[\.0-9\s]*',
     'L1I WRITEBACK ACCESS'],
    [r'L1I WRITEBACK ACCESS:[\.0-9\s]* HIT:([\.0-9\s]*) MISS:[\.0-9\s]*',
     'L1I WRITEBACK HIT'],
    [r'L1I WRITEBACK ACCESS:[\.0-9\s]* HIT:[\.0-9\s]* MISS:([\.0-9\s]*)',
     'L1I WRITEBACK MISS'],
    [r'L1I PREFETCH  REQUESTED:([\.0-9\s]*)ISSUED:[\.0-9\s]*USEFUL:[\.0-9\s]*USELESS:[\.0-9\s]*',
     'L1I PREFETCH REQUESTED'],
    [r'L1I PREFETCH  REQUESTED:[\.0-9\s]*ISSUED:([\.0-9\s]*)USEFUL:[\.0-9\s]*USELESS:[\.0-9\s]*',
     'L1I PREFETCH ISSUED'],
    [r'L1I PREFETCH  REQUESTED:[\.0-9\s]*ISSUED:[\.0-9\s]*USEFUL:([\.0-9\s]*)USELESS:[\.0-9\s]*',
     'L1I PREFETCH USEFUL'],
    [r'L1I PREFETCH  REQUESTED:[\.0-9\s]*ISSUED:[\.0-9\s]*USEFUL:[\.0-9\s]*USELESS:([\.0-9\s]*)', 'L1I PREFETCH USELESS'],
    [r'L1I AVERAGE MISS LATENCY: ([\.0-9\s]*) cycles',
     'L1I AVERAGE MISS LATENCY'],
    [r'L1I AVERAGE LOAD MISS LATENCY: ([\.0-9\s]*) cycles',
     'L1I AVERAGE LOAD MISS LATENCY'],
    [r'L1I AVERAGE RFO MISS LATENCY: ([\.0-9\s]*) cycles',
     'L1I AVERAGE RFO MISS LATENCY'],
    [r'L1I AVERAGE PREFETCH MISS LATENCY: ([\.0-9\s]*) cycles',
     'L1I AVERAGE PREFETCH MISS LATENCY'],
    [r'L1I AVERAGE WRITEBACK MISS LATENCY: ([\.0-9\s]*) cycles',
     'L1I AVERAGE WRITEBACK MISS LATENCY'],
    [r'L2C TOTAL     ACCESS:([\.0-9\s]*) HIT:[\.0-9\s]* MISS:[\.0-9\s]*',
     'L2C TOTAL ACCESS'],
    [r'L2C TOTAL     ACCESS:[\.0-9\s]* HIT:([\.0-9\s]*) MISS:[\.0-9\s]*',
     'L2C TOTAL HIT'],
    [r'L2C TOTAL     ACCESS:[\.0-9\s]* HIT:[\.0-9\s]* MISS:([\.0-9\s]*)', 'L2C TOTAL MISS'],
    [r'L2C LOAD      ACCESS:([\.0-9\s]*) HIT:[\.0-9\s]* MISS:[\.0-9\s]*',
     'L2C LOAD ACCESS'],
    [r'L2C LOAD      ACCESS:[\.0-9\s]* HIT:([\.0-9\s]*) MISS:[\.0-9\s]*',
     'L2C LOAD HIT'],
    [r'L2C LOAD      ACCESS:[\.0-9\s]* HIT:[\.0-9\s]* MISS:([\.0-9\s]*)', 'L2C LOAD MISS'],
    [r'L2C RFO       ACCESS:([\.0-9\s]*) HIT:[\.0-9\s]* MISS:[\.0-9\s]*',
     'L2C RFO ACCESS'],
    [r'L2C RFO       ACCESS:[\.0-9\s]* HIT:([\.0-9\s]*) MISS:[\.0-9\s]*',
     'L2C RFO HIT'],
    [r'L2C RFO       ACCESS:[\.0-9\s]* HIT:[\.0-9\s]* MISS:([\.0-9\s]*)', 'L2C RFO MISS'],
    [r'L2C PREFETCH  ACCESS:([\.0-9\s]*) HIT:[\.0-9\s]* MISS:[\.0-9\s]*',
     'L2C PREFETCH ACCESS'],
    [r'L2C PREFETCH  ACCESS:[\.0-9\s]* HIT:([\.0-9\s]*) MISS:[\.0-9\s]*',
     'L2C PREFETCH HIT'],
    [r'L2C PREFETCH  ACCESS:[\.0-9\s]* HIT:[\.0-9\s]* MISS:([\.0-9\s]*)',
     'L2C PREFETCH MISS'],
    [r'L2C WRITEBACK ACCESS:([\.0-9\s]*) HIT:[\.0-9\s]* MISS:[\.0-9\s]*',
     'L2C WRITEBACK ACCESS'],
    [r'L2C WRITEBACK ACCESS:[\.0-9\s]* HIT:([\.0-9\s]*) MISS:[\.0-9\s]*',
     'L2C WRITEBACK HIT'],
    [r'L2C WRITEBACK ACCESS:[\.0-9\s]* HIT:[\.0-9\s]* MISS:([\.0-9\s]*)',
     'L2C WRITEBACK MISS'],
    [r'L2C PREFETCH  REQUESTED:([\.0-9\s]*)ISSUED:[\.0-9\s]*USEFUL:[\.0-9\s]*USELESS:[\.0-9\s]*',
     'L2C PREFETCH REQUESTED'],
    [r'L2C PREFETCH  REQUESTED:[\.0-9\s]*ISSUED:([\.0-9\s]*)USEFUL:[\.0-9\s]*USELESS:[\.0-9\s]*',
     'L2C PREFETCH ISSUED'],
    [r'L2C PREFETCH  REQUESTED:[\.0-9\s]*ISSUED:[\.0-9\s]*USEFUL:([\.0-9\s]*)USELESS:[\.0-9\s]*',
     'L2C PREFETCH USEFUL'],
    [r'L2C PREFETCH  REQUESTED:[\.0-9\s]*ISSUED:[\.0-9\s]*USEFUL:[\.0-9\s]*USELESS:([\.0-9\s]*)', 'L2C PREFETCH USELESS'],
    [r'L2C AVERAGE MISS LATENCY: ([\.0-9\s]*) cycles',
     'L2C AVERAGE MISS LATENCY'],
    [r'L2C AVERAGE LOAD MISS LATENCY: ([\.0-9\s]*) cycles',
     'L2C AVERAGE LOAD MISS LATENCY'],
    [r'L2C AVERAGE RFO MISS LATENCY: ([\.0-9\s]*) cycles',
     'L2C AVERAGE RFO MISS LATENCY'],
    [r'L2C AVERAGE PREFETCH MISS LATENCY: ([\.0-9\s]*) cycles',
     'L2C AVERAGE PREFETCH MISS LATENCY'],
    [r'L2C AVERAGE WRITEBACK MISS LATENCY: ([\.0-9\s]*) cycles',
     'L2C AVERAGE WRITEBACK MISS LATENCY'],
    [r'LLC TOTAL     ACCESS:([\.0-9\s]*) HIT:[\.0-9\s]* MISS:[\.0-9\s]*',
     'LLC TOTAL ACCESS'],
    [r'LLC TOTAL     ACCESS:[\.0-9\s]* HIT:([\.0-9\s]*) MISS:[\.0-9\s]*',
     'LLC TOTAL HIT'],
    [r'LLC TOTAL     ACCESS:[\.0-9\s]* HIT:[\.0-9\s]* MISS:([\.0-9\s]*)', 'LLC TOTAL MISS'],
    [r'LLC LOAD      ACCESS:([\.0-9\s]*) HIT:[\.0-9\s]* MISS:[\.0-9\s]*',
     'LLC LOAD ACCESS'],
    [r'LLC LOAD      ACCESS:[\.0-9\s]* HIT:([\.0-9\s]*) MISS:[\.0-9\s]*',
     'LLC LOAD HIT'],
    [r'LLC LOAD      ACCESS:[\.0-9\s]* HIT:[\.0-9\s]* MISS:([\.0-9\s]*)', 'LLC LOAD MISS'],
    [r'LLC RFO       ACCESS:([\.0-9\s]*) HIT:[\.0-9\s]* MISS:[\.0-9\s]*',
     'LLC RFO ACCESS'],
    [r'LLC RFO       ACCESS:[\.0-9\s]* HIT:([\.0-9\s]*) MISS:[\.0-9\s]*',
     'LLC RFO HIT'],
    [r'LLC RFO       ACCESS:[\.0-9\s]* HIT:[\.0-9\s]* MISS:([\.0-9\s]*)', 'LLC RFO MISS'],
    [r'LLC PREFETCH  ACCESS:([\.0-9\s]*) HIT:[\.0-9\s]* MISS:[\.0-9\s]*',
     'LLC PREFETCH ACCESS'],
    [r'LLC PREFETCH  ACCESS:[\.0-9\s]* HIT:([\.0-9\s]*) MISS:[\.0-9\s]*',
     'LLC PREFETCH HIT'],
    [r'LLC PREFETCH  ACCESS:[\.0-9\s]* HIT:[\.0-9\s]* MISS:([\.0-9\s]*)',
     'LLC PREFETCH MISS'],
    [r'LLC WRITEBACK ACCESS:([\.0-9\s]*) HIT:[\.0-9\s]* MISS:[\.0-9\s]*',
     'LLC WRITEBACK ACCESS'],
    [r'LLC WRITEBACK ACCESS:[\.0-9\s]* HIT:([\.0-9\s]*) MISS:[\.0-9\s]*',
     'LLC WRITEBACK HIT'],
    [r'LLC WRITEBACK ACCESS:[\.0-9\s]* HIT:[\.0-9\s]* MISS:([\.0-9\s]*)',
     'LLC WRITEBACK MISS'],
    [r'LLC PREFETCH  REQUESTED:([\.0-9\s]*)ISSUED:[\.0-9\s]*USEFUL:[\.0-9\s]*USELESS:[\.0-9\s]*',
     'LLC PREFETCH REQUESTED'],
    [r'LLC PREFETCH  REQUESTED:[\.0-9\s]*ISSUED:([\.0-9\s]*)USEFUL:[\.0-9\s]*USELESS:[\.0-9\s]*',
     'LLC PREFETCH ISSUED'],
    [r'LLC PREFETCH  REQUESTED:[\.0-9\s]*ISSUED:[\.0-9\s]*USEFUL:([\.0-9\s]*)USELESS:[\.0-9\s]*',
     'LLC PREFETCH USEFUL'],
    [r'LLC PREFETCH  REQUESTED:[\.0-9\s]*ISSUED:[\.0-9\s]*USEFUL:[\.0-9\s]*USELESS:([\.0-9\s]*)', 'LLC PREFETCH USELESS'],
    [r'LLC AVERAGE MISS LATENCY: ([\.0-9\s]*) cycles',
     'LLC AVERAGE MISS LATENCY'],
    [r'LLC AVERAGE LOAD MISS LATENCY: ([\.0-9\s]*) cycles',
     'LLC AVERAGE LOAD MISS LATENCY'],
    [r'LLC AVERAGE RFO MISS LATENCY: ([\.0-9\s]*) cycles',
     'LLC AVERAGE RFO MISS LATENCY'],
    [r'LLC AVERAGE PREFETCH MISS LATENCY: ([\.0-9\s]*) cycles',
     'LLC AVERAGE PREFETCH MISS LATENCY'],
    [r'LLC AVERAGE WRITEBACK MISS LATENCY: ([\.0-9\s]*) cycles',
     'LLC AVERAGE WRITEBACK MISS LATENCY'],
]

faultsPatterns = [
    [r'Major fault:([\.0-9\s]*)', 'Major fault'],
    [r'Minor fault:([\.0-9\s]*)', 'Minor fault'],
    [r'L1D WB Stall Cycle:([\.0-9\s]*)', 'L1D WB Stall Cycle']
]

DRAMStatPatterns = [
    [r'DBUS_CONGESTED:([\.0-9\s]*)', 'DBUS_CONGESTED'],
    [r'AVG_CONGESTED_CYCLE:([\.0-9\s]*)\n', 'AVG_CONGESTED_CYCLE'],
    [r'RQ ROW_BUFFER_HIT:([\.0-9\s]*)ROW_BUFFER_MISS:[\.0-9\s]*',
     'RQ ROW_BUFFER_HIT'],
    [r'RQ ROW_BUFFER_HIT:[\.0-9\s]*ROW_BUFFER_MISS:([\.0-9\s]*)',
     'RQ ROW_BUFFER_MISS'],
    [r'WQ ROW_BUFFER_HIT:([\.0-9\s]*)ROW_BUFFER_MISS:[\.0-9\s]*FULL:[\.0-9\s]*',
     'WQ ROW_BUFFER_HIT'],
    [r'WQ ROW_BUFFER_HIT:[\.0-9\s]*ROW_BUFFER_MISS:([\.0-9\s]*)FULL:[\.0-9\s]*',
     'WQ ROW_BUFFER_MISS'],
    [r'WQ ROW_BUFFER_HIT:[\.0-9\s]*ROW_BUFFER_MISS:[\.0-9\s]*FULL:([\.0-9\s]*)', 'WQ FULL'],
]

branchStatPatterns = [
    [r'Branch Prediction Accuracy: ([\.0-9\s\%]*)',
     'Branch Prediction Accuracy'],
    [r'MPKI: ([\.0-9\s]*)', 'MPKI'],
    [r'Average ROB Occupancy at Mispredict: ([\.0-9\s]*)',
     'Average ROB Occupancy at Mispredict'],
    [r'NOT_BRANCH: ([\.0-9]*)', 'NOT_BRANCH'],
    [r'BRANCH_DIRECT_JUMP: ([\.0-9]*)', 'BRANCH_DIRECT_JUMP'],
    [r'BRANCH_INDIRECT: ([\.0-9]*)', 'BRANCH_INDIRECT'],
    [r'BRANCH_CONDITIONAL: ([\.0-9]*)', 'BRANCH_CONDITIONAL'],
    [r'BRANCH_DIRECT_CALL: ([\.0-9]*)', 'BRANCH_DIRECT_CALL'],
    [r'BRANCH_INDIRECT_CALL: ([\.0-9]*)', 'BRANCH_INDIRECT_CALL'],
    [r'BRANCH_RETURN: ([\.0-9]*)', 'BRANCH_RETURN'],
    [r'BRANCH_OTHER: ([\.0-9]*)', 'BRANCH_OTHER']
]

tracePattern = [[r"CPU [0-9]+ runs [/\.0-9A-Za-z\_]+/([0-9a-zA-Z\.\-\_]+)\n", "Trace Name"]]

@extracter(tracePattern)
def get_traces(content:str):
    pass

@extracter(tracePattern+appendixPatterns+ipcPattern+cacheStatPatterns+faultsPatterns+DRAMStatPatterns+branchStatPatterns)
def get_all_results(content: str):
    pass


def extract_load_cache_miss_latency(content):
    pattern1 = re.compile(r"L1D AVERAGE LOAD MISS LATENCY:([\.0-9\s]*)cycles")
#     pattern2 = re.compile(r"L1I AVERAGE LOAD MISS LATENCY:([\.0-9\s]*)cycles")
    pattern3 = re.compile(r"L2C AVERAGE LOAD MISS LATENCY:([\.0-9\s]*)cycles")
    pattern4 = re.compile(r"LLC AVERAGE LOAD MISS LATENCY:([\.0-9\s]*)cycles")

    regObj1 = pattern1.search(content)
#     regObj2 = pattern2.search(content)
    regObj3 = pattern3.search(content)
    regObj4 = pattern4.search(content)

    result = None
#     print(regObj1, regObj2, regObj3, regObj4)
    if regObj1 and regObj3 and regObj4:
        result = {
            "L1D AVERAGE LOAD MISS LATENCY": float(regObj1.group(1).strip()),
            #             "L1I AVERAGE LOAD MISS LATENCY": float(regObj2.group(1).strip()) if regObj2 else -1,
            "L2C AVERAGE LOAD MISS LATENCY": float(regObj3.group(1).strip()),
            "LLC AVERAGE LOAD MISS LATENCY": float(regObj4.group(1).strip())
        }
    return result


def extract_prefetch_miss_latency(content):
    pattern1 = re.compile(
        r"L1D AVERAGE PREFETCH MISS LATENCY:([\.0-9\s]*)cycles")
    pattern3 = re.compile(
        r"L2C AVERAGE PREFETCH MISS LATENCY:([\.0-9\s]*)cycles")
    pattern4 = re.compile(
        r"LLC AVERAGE PREFETCH MISS LATENCY:([\.0-9\s]*)cycles")

    regObj1 = pattern1.search(content)
    regObj3 = pattern3.search(content)
    regObj4 = pattern4.search(content)

    result = None
    if regObj1 and regObj3 and regObj4:
        result = {
            "L1D AVERAGE PREFECH MISS LATENCY": float(regObj1.group(1).strip()),
            "L2C AVERAGE PREFECH MISS LATENCY": float(regObj3.group(1).strip()),
            "LLC AVERAGE PREFECH MISS LATENCY": float(regObj4.group(1).strip())
        }
    return result


delta_seq_stats_pattern = [
    ["\[Delta Seq\] Prefetch Vote Turns Count:([\.0-9+\-a-z]*)",
     "Prefetch Vote Turns Count"],
    ["\[Delta Seq\] Prefetch Voters Mean:([\.0-9+\-a-z]*)",
     "Prefetch Voters Mean"],
    ["\[Delta Seq\] Prefetch Voters SD:([\.0-9+\-a-z]*)",
     "Prefetch Voters SD"],
    ["\[Delta Seq\] 3 Deltas Hit Mean:([\.0-9+\-a-z]*)", "3 Deltas Hit Mean"],
    ["\[Delta Seq\] 2 Deltas Hit Mean:([\.0-9+\-a-z]*)", "2 Deltas Hit Mean"],
    ["\[Delta Seq\] 1 Deltas Hit Mean:([\.0-9+\-a-z]*)", "1 Deltas Hit Mean"],
    ["\[Delta Seq\] Delta Miss Mean:([\.0-9+\-a-z]*)", "Delta Miss Mean"],
    ["\[Delta Seq\] Table Update Count:([\.0-9+\-a-z]*)",
     "Table Update Count"],
    ["\[Delta Seq\] Table Eviction Count:([\.0-9+\-a-z]*)",
     "Table Eviction Count"],
    ["\[Delta Seq\] Table Eviction Conf Mean:([\.0-9+\-a-z]*)",
     "Table Eviction Conf Mean"],
    ["\[Delta Seq\] Prefetch L1 Mean:([\.0-9+\-a-z]*)", "Prefetch L1 Mean"],
    ["\[Delta Seq\] Prefetch L2 Mean:([\.0-9+\-a-z]*)", "Prefetch L2 Mean"],
    ["\[Delta Seq\] Prefetch LLC Mean:([\.0-9+\-a-z]*)", "Prefetch LLC Mean"],
    ["\[Delta Seq\] Real Prefetch Mean:([\.0-9+\-a-z]*)",
     "Real Prefetch Mean"],
    ["\[Delta Seq\] Prefetch Degree Mean:([\.0-9+\-a-z]*)",
     "Prefetch Degree Mean"],
    ["\[Delta Seq\] Useful Prefetch Rate:([\.0-9+\-a-z]*)",
     "Useful Prefetch Rate"],
    ["\[Delta Seq\] Useful Recent Prefetch Rate:([\.0-9+\-a-z]*)",
     "Useful Recent Prefetch Rate"],
    ["\[Delta Seq\] Useful Recent Prefetch Score Mean:([\.0-9+\-a-z]*)",
     "Useful Recent Prefetch Score Mean"],
    ["\[Delta Seq\] Table Conf Saturated Mean:([\.0-9+\-a-z]*)",
     "Table Conf Saturated Mean"],
    ["\[Delta Seq\] His Table Miss Mean:([\.0-9+\-a-z]*)",
     "His Table Miss Mean"],
    ["\[Delta Seq\] Delta Index Table Miss Mean:([\.0-9+\-a-z]*)",
     "Delta Index Table Miss Mean"],
    ["\[Delta Seq\] Delta Seq Table Expand Mean:([\.0-9+\-a-z]*)",
     "Delta Seq Table Expand Mean"],
    ["\[Delta Seq\] Constant Stride Prefetch Rate:([\.0-9+\-a-z]*)",
     "Constant Stride Prefetch Rate"],
    ["\[Delta Seq\] Fast Learning Rate:([\.0-9+\-a-z]*)",
     "Fast Learning Rate"],
    ["\[Delta Seq\] Constant Stride Real Prefetch Mean:([\.0-9+\-a-z]*)",
     "Constant Stride Real Prefetch Mean"],
    ["\[Delta Seq\] Constant Stride Prefetch Degree Mean:([\.0-9+\-a-z]*)",
     "Constant Stride Prefetch Degree Mean"],
    ["\[Delta Seq\] Complex Pattern Real Prefetch Mean:([\.0-9+\-a-z]*)",
     "Complex Pattern Real Prefetch Mean"],
    ["\[Delta Seq\] Complex Pattern Prefetch Degree Mean:([\.0-9+\-a-z]*)",
     "Complex Pattern Prefetch Degree Mean"],
    ["\[Delta Seq\] Complex Pattern Prefetch L1 Mean:([\.0-9+\-a-z]*)",
     "Complex Pattern Prefetch L1 Mean"],
    ["\[Delta Seq\] Complex Pattern Prefetch L2 Mean:([\.0-9+\-a-z]*)",
     "Complex Pattern Prefetch L2 Mean"],
    ["\[Delta Seq\] Prefetch Count:([\.0-9+\-a-z]*)", "Prefetch Count"],
    ["\[Delta Seq\] Shoot Prefetch Sequence Rate:([\.0-9+\-a-z]*)",
     "Shoot Prefetch Sequence Rate"]
]


@extracter(delta_seq_stats_pattern)
def status_analyze_delta_seq(content: str):
    pass


def status_analyze_l2c(content):
    pattern = re.compile(
        "\[L2C\] Constant Stride Prefetch Rate:([\.0-9+\-a-z]*)\n\[L2C\] NL Prefetch Rate:([\.0-9+\-a-z]*)\n\[L2C\] Complex Prefetch Rate:([\.0-9+\-a-z]*)\n")
    regObj = pattern.search(content)
    result = None
    if regObj:
        result = {
            "Constant Stride Prefetch Rate": float(regObj.group(1)),
            "NL Prefetch Rate": float(regObj.group(2)),
            "Complex Prefetch Rate": float(regObj.group(3))
        }
    return result


def get_mpki(content: str):
    pattern_l1d = re.compile(
        r"L1D LOAD[\s]*ACCESS:([0-9\s]+)HIT:([0-9\s]+)MISS:([0-9\s]+)\n")
    #pattern_l1i = re.compile(r"L1I LOAD[\s]*ACCESS:([0-9\s]+)HIT:([0-9\s]+)MISS:([0-9\s]+)\n")
    pattern_l2 = re.compile(
        r"L2C LOAD[\s]*ACCESS:([0-9\s]+)HIT:([0-9\s]+)MISS:([0-9\s]+)\n")
    pattern_llc = re.compile(
        r"LLC LOAD[\s]*ACCESS:([0-9\s]+)HIT:([0-9\s]+)MISS:([0-9\s]+)\n")
    regObj1 = pattern_l1d.search(content)
    #regObj2 = pattern_l1i.search(content)
    regObj3 = pattern_l2.search(content)
    regObj4 = pattern_llc.search(content)
    result = None
    if regObj1 and regObj3 and regObj4:
        result = {
            "MPKI L1": int(regObj1.group(3).strip()) * 1000 / 20000000,
            "MPKI L2": int(regObj3.group(3).strip()) * 1000 / 20000000,
            "MPKI LLC": int(regObj4.group(3).strip()) * 1000 / 20000000
        }
    return result


def get_dbus_congested(content: str):
    pattern = re.compile(r"DBUS_CONGESTED:([\s0-9]+)\n")
    regObj = pattern.search(content)
    result = None
    if regObj:
        result = {"DBUS_CONGESTED": int(regObj.group(1).strip())}
    return result


def get_bandwidth_cost(content: str):
    pattern = re.compile(
        r"RQ ROW_BUFFER_HIT:([\s0-9]+)ROW_BUFFER_MISS:([\s0-9]+)")
    pattern1 = re.compile(r"DBUS_CONGESTED:([\s0-9]+)")
    pattern2 = re.compile(
        r"WQ ROW_BUFFER_HIT:([\s0-9]+)ROW_BUFFER_MISS:([\s0-9]+)")
    regObj = pattern.search(content)
    regObj1 = pattern1.search(content)
    regObj2 = pattern2.search(content)
    result = None
    if regObj and regObj1 and regObj2:
        result = {"bandwidth":
                  int(regObj.group(1).strip()) +
                  int(regObj.group(2).strip()) +
                  int(regObj1.group(1).strip()) +
                  int(regObj2.group(1).strip()) +
                  int(regObj2.group(2).strip())}
    return result


def get_l1d_prefetch_stat(content: str):
    pattern = re.compile(
        r"L1D PREFETCH  REQUESTED:[\s0-9]+ISSUED:[\s0-9]+USEFUL:([\s0-9]+)USELESS:([\s0-9]+)")
    regObj = pattern.search(content)
    result = None
    if regObj:
        result = {"useful": int(regObj.group(1).strip()),
                  "useless": int(regObj.group(2).strip())}
    return result


def get_l1d_miss(content: str):
    pattern = re.compile(
        r"L1D LOAD[\s]+ACCESS:[\s0-9]+HIT:[\s0-9]+MISS:([\s0-9]+)")
    regObj = pattern.search(content)
    result = None
    if regObj:
        result = {"l1d_miss": int(regObj.group(1).strip())}
    return result


def get_load_access(content: str):
    pattern = re.compile(r"L1D LOAD[\s]+ACCESS:[\s]+([0-9]+)")
    regObj = pattern.search(content)
    result = None
    if regObj:
        result = {"access": int(regObj.group(1).strip())}
    return result


def get_all_load_access(content: str):
    pattern = re.compile(r"L1D LOAD[\s]+ACCESS:[\s]+([0-9]+)")
    pattern1 = re.compile(r"L2C LOAD[\s]+ACCESS:[\s]+([0-9]+)")
    pattern2 = re.compile(r"LLC LOAD[\s]+ACCESS:[\s]+([0-9]+)")
    regObj = pattern.search(content)
    regObj1 = pattern1.search(content)
    regObj2 = pattern2.search(content)
    result = None
    if regObj and regObj1 and regObj2:
        result = {
            "L1D LOAD ACCESS": int(regObj.group(1).strip()),
            "L2C LOAD ACCESS": int(regObj1.group(1).strip()),
            "LLC LOAD ACCESS": int(regObj2.group(1).strip())
        }
    return result


def get_all_rfo_access(content: str):
    pattern = re.compile(r"L1D RFO[\s]+ACCESS:[\s]+([0-9]+)")
    pattern1 = re.compile(r"L2C RFO[\s]+ACCESS:[\s]+([0-9]+)")
    pattern2 = re.compile(r"LLC RFO[\s]+ACCESS:[\s]+([0-9]+)")
    regObj = pattern.search(content)
    regObj1 = pattern1.search(content)
    regObj2 = pattern2.search(content)
    result = None
    if regObj and regObj1 and regObj2:
        result = {
            "L1D RFO ACCESS": int(regObj.group(1).strip()),
            "L2C RFO ACCESS": int(regObj1.group(1).strip()),
            "LLC RFO ACCESS": int(regObj2.group(1).strip())
        }
    return result


def get_all_load_miss(content: str):
    pattern = re.compile(
        r"L1D LOAD[\s]+ACCESS:[\s0-9]+HIT:[\s0-9]+MISS:([\s0-9]+)")
    pattern1 = re.compile(
        r"L2C LOAD[\s]+ACCESS:[\s0-9]+HIT:[\s0-9]+MISS:([\s0-9]+)")
    pattern2 = re.compile(
        r"LLC LOAD[\s]+ACCESS:[\s0-9]+HIT:[\s0-9]+MISS:([\s0-9]+)")
    regObj = pattern.search(content)
    regObj1 = pattern1.search(content)
    regObj2 = pattern2.search(content)
    result = None
    if regObj and regObj1 and regObj2:
        result = {
            "L1D LOAD MISS": int(regObj.group(1).strip()),
            "L2C LOAD MISS": int(regObj1.group(1).strip()),
            "LLC LOAD MISS": int(regObj2.group(1).strip())
        }
    return result


def get_all_rfo_miss(content: str):
    pattern = re.compile(
        r"L1D RFO[\s]+ACCESS:[\s0-9]+HIT:[\s0-9]+MISS:([\s0-9]+)")
    pattern1 = re.compile(
        r"L2C RFO[\s]+ACCESS:[\s0-9]+HIT:[\s0-9]+MISS:([\s0-9]+)")
    pattern2 = re.compile(
        r"LLC RFO[\s]+ACCESS:[\s0-9]+HIT:[\s0-9]+MISS:([\s0-9]+)")
    regObj = pattern.search(content)
    regObj1 = pattern1.search(content)
    regObj2 = pattern2.search(content)
    result = None
    if regObj and regObj1 and regObj2:
        result = {
            "L1D RFO MISS": int(regObj.group(1).strip()),
            "L2C RFO MISS": int(regObj1.group(1).strip()),
            "LLC RFO MISS": int(regObj2.group(1).strip())
        }
    return result


def get_access_excl_pref(content: str):
    pattern = re.compile(
        r"L1D LOAD[\s]+ACCESS:[\s]+([0-9]+)[\S\s]+L1D RFO       ACCESS:[\s]+([0-9]+)")
    regObj = pattern.search(content)
    result = None
    if regObj:
        result = {"access": int(regObj.group(1).strip()) +
                  int(regObj.group(2).strip())}
    return result


def guess_model_from_results(keywords):
    res = os.popen(
        f"ls ./results_200M/602.gcc_s-1850B.champsimtrace.xz*{keywords}*")
    return [s[s.find("champsimtrace.xz-")+17:].replace(".txt.xz\n", '').replace(".txt\n", '') for s in res.readlines()]


def get_all_models(keywords):
    for root, dirs, files in os.walk("./bin"):
        if not keywords:
            return files
        else:
            res = []
            for f in files:
                if keywords in f:
                    res.append(f)
            return res


def clear_str(s: str):
    return s.replace('.', '_')


def analyze_output(directory, partial_model_name, workload_list:list, statu_analyze_func, wf=result_file_filter):
    stat_data = {}
    multi_core = isinstance(workload_list[0], list)
    for root, dirs, files in os.walk(directory, topdown=False):
        for name in files:
            filepath = os.path.join(root, name)
            p, no = wf(partial_model_name, name)
            if not p or not multi_core and not no in workload_list:
                continue
            if filepath.endswith('.xz'):
                openFunc, flags = lzma.open, 'rt'
            else:
                openFunc, flags = open, 'r'
            with openFunc(filepath, flags) as res:
                content = res.read()
                # get status
                if multi_core:
                    try:
                        traces = get_traces(content)['Trace Name']
                        if not traces in workload_list:
                            continue
                    except Exception as e:
                        print(name)
                        raise e
                stat_data[no] = statu_analyze_func(content) 
                if not stat_data[no]:
                    logging.warning(f"No stat for {filepath}")
    return stat_data


def get_normalized_ipc(data, core_n=1, raise_err=True):
    baseline_model = BASE_LINE_MODEL.replace("1core", f"{core_n}core")
    assert(baseline_model in data)
    result = {}
    baseline = data[baseline_model]
    for k in data:
        if k == baseline_model:
            continue
        one = {}
        for trace in data[k]:
            try:
                if core_n > 1:
                    nipc = gmean(data[k][trace]['cumulative IPC']) / gmean(baseline[trace]['cumulative IPC'])
                else:
                    nipc = data[k][trace]['cumulative IPC'] / baseline[trace]['cumulative IPC']
                one[trace] = {'Normalized IPC': nipc} 
            except Exception as e:
                print(k, trace)
                if raise_err:
                    raise e
        result[k] = one
    return result

# compute the relative results depend on the env parameters (LLC Size or Bandwidth)
def get_results_by_env(func, data, core_n=1, raise_err=True):
    result = {}
    llc_pattern = re.compile(r"(LLC_SET-[0-9]+)")
    band_pattern = re.compile(r"(DRAM_IO_FREQ-[0-9]+)")
    for key in data:
        llc_res = llc_pattern.search(key)
        band_res = band_pattern.search(key)
        baseline_model = BASE_LINE_MODEL.replace("1core", f"{core_n}core")
        if llc_res:
            baseline_model = f"bimodal-no-no-{llc_res.group(1)}-no-no-lru-{core_n}core" 
        if band_res:
            baseline_model = f"bimodal-no-no-{band_res.group(1)}-no-no-lru-{core_n}core" 
        if key == baseline_model:
            continue
        baseline = data[baseline_model] 
        one = {}
        for trace in data[key]:
            try:
                one[trace] = func(data[key][trace], baseline[trace])
            except Exception as e:
                print(key, trace)
                if raise_err:
                    raise e
        result[key] = one
    return result

def get_nipc_by_env(data, core_n=1, raise_err=True):
    def get_nipc(data, baseline):
        if core_n > 1:
            nipc = gmean(data['cumulative IPC']) / gmean(baseline['cumulative IPC'])
        else:
            nipc = data['cumulative IPC'] / baseline['cumulative IPC']
        return {'Normalized IPC': nipc} 
    return get_results_by_env(get_nipc, data, core_n, raise_err) 

def get_coverage_by_env(data, core_n=1, raise_err=True):
    def get_coverage(data, baseline):
        cache = ['L1D', 'L2C', 'LLC']
        coverages = {}
        for c in cache:
                coverage = 1 - data[c+" LOAD MISS"] / baseline[c+" LOAD MISS"] if baseline[c+" LOAD MISS"] else 0
                coverages[c + " Coverage"] = coverage
        return coverages
    return get_results_by_env(get_coverage, data, core_n, raise_err)

def get_coverage(data):
    assert(BASE_LINE_MODEL in data)
    cache = ['L1D', 'L2C', 'LLC']
    result = {}
    baseline = data[BASE_LINE_MODEL]
    for k in data:
        if k == BASE_LINE_MODEL:
            continue
        one = {}
        for trace in data[k]:
            coverages = {}
            for c in cache:
                try:
                    # 对于某些负载，经过预热后就足以让其消除所有MISS。理论上这类负载不是Prefetch Sensitive的
                    # scale_conf = baseline[trace][c+" LOAD ACCESS"] / data[k][trace][c+" LOAD ACCESS"] if data[k][trace][c+" LOAD ACCESS"] else 1
                    coverage = 1 - data[k][trace][c+" LOAD MISS"] / baseline[trace][c+" LOAD MISS"] if baseline[trace][c+" LOAD MISS"] else 0
                    coverages[c + " Coverage"] = coverage
                except Exception as e:
                    print(k, trace, c)
                    raise e
            one[trace] = coverages
        result[k] = one
    return result

def get_miss_per_kiloinstr(data):
    cache = ['L1D', 'L2C', 'LLC']
    result = {}
    for k in data:
        one = {}
        for trace in data[k]:
            mpkis = {}
            for c in cache:
                try:
                    mpki = 1000 * data[k][trace][c + " LOAD MISS"] / data[k][trace]["Simulation Instructions"]
                    mpkis[c + " MPKI"] = mpki
                except Exception as e:
                    print(k, trace, c)
                    raise e
            one[trace] = mpkis
        result[k] = one
    return result


def get_overprediction(data):
    assert(BASE_LINE_MODEL in data)
    cache = ['L1D', 'L2C', 'LLC']
    result = {}
    baseline = data[BASE_LINE_MODEL]
    for k in data:
        if k == BASE_LINE_MODEL:
            continue
        one = {}
        for trace in data[k]:
            overps = {}
            for c in cache:
                try:
                    # 对于某些负载，经过预热后就足以让其消除所有MISS。理论上这类负载不是Prefetch Sensitive的
                    # scale_conf = baseline[trace][c+" LOAD ACCESS"] / data[k][trace][c +
                    #                                                                 " LOAD ACCESS"] if data[k][trace][c + " LOAD ACCESS"] else 1
                    overp = data[k][trace][c+" PREFETCH USELESS"] / \
                        baseline[trace][c + " LOAD MISS"] if baseline[trace][c + " LOAD MISS"] else 0
                    overps[c + " Overprediction Rate"] = overp
                except Exception as e:
                    print(k, trace, c)
                    raise e
            one[trace] = overps
        result[k] = one
    return result


def get_accuracy(data, method=1):
    assert(BASE_LINE_MODEL in data)
    cache = ['L1D', 'L2C', 'LLC']
    result = {}
    baseline = data[BASE_LINE_MODEL]
    for k in data:
        if k == BASE_LINE_MODEL:
            continue
        one = {}
        for trace in data[k]:
            accuracies = {}
            for c in cache:
                try:
                    # 对于某些负载，经过预热后就足以让其消除所有MISS。理论上这类负载不是Prefetch Sensitive的
                    if method == 1:
                        accuracy = data[k][trace][c+" PREFETCH USEFUL"] / data[k][trace][c + " PREFETCH ISSUED"] \
                            if data[k][trace][c + " PREFETCH ISSUED"] else 0
                    elif method == 2:
                        accuracy = data[k][trace][c+" PREFETCH USEFUL"]/(data[k][trace][c+" PREFETCH USEFUL"]+data[k][trace][c+" PREFETCH USELESS"]) \
                            if data[k][trace][c + " PREFETCH USEFUL"] else 0
                    accuracies[c + " Accuracy"] = accuracy
                except Exception as e:
                    print(k, trace, c)
                    raise e
            one[trace] = accuracies
        result[k] = one
    return result


def convert2df(data):
    dfs = {}
    for model in data:
        df = {}
        for trace in data[model]:
            df[trace] = data[model][trace]
        dfs[model] = pd.DataFrame(df).T
    return dfs


def display_coverage_and_accuracy(model, workloads):
    baseline_access = analyze_output(
        "./results_200M/", BASE_LINE_MODEL, workloads, get_load_access, result_file_filter)
    access = analyze_output("./results_200M/", model,
                            workloads, get_load_access, result_file_filter)
    baseline_l1d_miss_stand = analyze_output(
        "./results_200M/", BASE_LINE_MODEL, workloads, get_l1d_miss, result_file_filter)
    l1d_miss_stand = analyze_output(
        "./results_200M/", model, workloads, get_l1d_miss, result_file_filter)
    l1d_prfetch_stat = analyze_output(
        "./results_200M/", model, workloads, get_l1d_prefetch_stat, result_file_filter)
    overpredicted = []
    coverage_data = []
    uncoverage_data = []
    for key in workloads:
        scale_conf = baseline_access[key]['access'] / access[key]['access']
        overpredicted += [scale_conf * l1d_prfetch_stat[key]['useless'] /
                          baseline_l1d_miss_stand[key]['l1d_miss'] if l1d_miss_stand[key]['l1d_miss'] else 1]
#         overpredicted += [l1d_prfetch_stat[key]['useless']/(l1d_prfetch_stat[key]['useless']+l1d_prfetch_stat[key]['useful'])]
        coverage = 1 - scale_conf * \
            l1d_miss_stand[key]['l1d_miss'] / \
            baseline_l1d_miss_stand[key]['l1d_miss']
        coverage_data += [coverage]
        uncoverage_data += [1 - coverage]
#         print(key, scale_conf, coverage, l1d_miss_stand[key]['l1d_miss'], baseline_l1d_miss_stand[key]['l1d_miss'])
    fig, ax = plt.subplots()
    x = np.arange(len(workloads)+1)
    width = 0.4
    cover_gmean = sum(coverage_data)/len(coverage_data)
    coverage_data += [cover_gmean]
    uncoverage_data += [1 - cover_gmean]
    ax.bar(x, coverage_data, width, label='covered')
    ax.bar(x, uncoverage_data, width, bottom=coverage_data, label='uncovered')
    o_mean = sum(overpredicted)/len(overpredicted)
    ax.bar(x, overpredicted+[o_mean], width, bottom=[1]
           * (len(workloads)+1), label='overpredicted')
    ax.annotate(
        "%.02f" % cover_gmean,
        xy=(x[-1], cover_gmean), xycoords='data',
        xytext=(20, 30), textcoords='offset points',
        arrowprops=dict(arrowstyle="->"),
        horizontalalignment='right',
        verticalalignment='top')
    ax.annotate(
        "%.02f" % (1 - cover_gmean),
        xy=(x[-1], 1), xycoords='data',
        xytext=(-20, 30), textcoords='offset points',
        arrowprops=dict(arrowstyle="->"),
        horizontalalignment='right',
        verticalalignment='top')
    ax.annotate(
        "%.02f" % o_mean,
        xy=(x[-1], o_mean + 1), xycoords='data',
        xytext=(20, 30), textcoords='offset points',
        arrowprops=dict(arrowstyle="->"),
        horizontalalignment='right',
        verticalalignment='top')
    ax.set_xticks(x)
    labels = [k.replace(".champsimtrace.xz", "")
              for k in workloads] + ['amean']
    ax.set_xticklabels(labels)
    font = {"weight": "normal", "size": 16}
    ax.set_xlabel("Workloads", font)
    ax.set_ylabel("Ratio", font)
    ax.legend(ncol=3, bbox_to_anchor=(0, 1),
              loc='lower left', fontsize='small')
    plt.setp(ax.get_xticklabels(), rotation=90)
    fig.set_size_inches(16, 4)
    return overpredicted


def display_bandwidth(models, name_workloads, workloads, width, up_limit, *fig_size):
    baseline = analyze_output("./results_200M/", "no-no-no-no-lru",
                              workloads, get_bandwidth_cost, result_file_filter)
    fig, ax = plt.subplots()
    addition = []
    for i, _ in enumerate(models):
        addition.append(i * width - ((len(models)-1)*width)/2)
    x = np.arange(len(workloads)+1)
    result = {}
    for i, model in enumerate(models):
        cur_dbus = analyze_output(
            "./results_200M/", model, workloads, get_bandwidth_cost, result_file_filter)
        dbus_data = []
#         print(cur_dbus)
        for key in workloads:
            dbus_data.append(
                cur_dbus[key]['bandwidth'] / baseline[key]['bandwidth'])
        avg_bandwidth = gmean(dbus_data)
        result[model] = avg_bandwidth
#         avg_bandwidth = sum(dbus_data)/len(dbus_data)
        ax.bar(x + addition[i], dbus_data + [avg_bandwidth],
               width, label=name_workloads[i])
        ax.annotate(
            "%.02f" % avg_bandwidth,
            xy=(x[-1] + addition[i], gmean(dbus_data)), xycoords='data',
            xytext=(i*20-20, 40+i*(-10)), textcoords='offset points',
            arrowprops=dict(arrowstyle="->"),
            horizontalalignment='right',
            verticalalignment='top')
        for j, high in enumerate(dbus_data):
            if high >= up_limit:
                ax.annotate(
                    "%.02f" % dbus_data[j],
                    xy=(x[j]+addition[i], up_limit), xycoords='data',
                    xytext=(+10, -10), textcoords='offset points',
                    arrowprops=dict(arrowstyle="->"),
                    horizontalalignment='right',
                    verticalalignment='top')
    labels = [k.replace(".champsimtrace.xz", "")
              for k in workloads] + ['gmean']
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    font = {"weight": "normal", "size": 16}
    ax.set_xlabel("Workloads", font)
    ax.set_ylabel("Ratio", font)
    ax.legend(ncol=len(models), bbox_to_anchor=(0, 1),
              loc='lower left', fontsize='small')
    plt.setp(ax.get_xticklabels(), rotation=90)
    fig.set_size_inches(fig_size)
    return result


def default_classify(raw_data):
    content = {}
    for key in raw_data:
        words = key.split(".")
        token = words[3]
        if token not in content:
            content[token] = [[words[0]+'.'+words[1], raw_data[key]]]
        else:
            content[token].append([words[0]+'.'+words[1], raw_data[key]])
    return content


def default_filter(workload_name, workloads):
    words = workload_name.split('.')
    no = words[0]+'.'+words[1]+".champsimtrace.xz"
    return no, no in workloads


def cloudsuits_classify(raw_data):
    content = {}
    for key in raw_data:
        words = key.split("-")
        token = "xz-bimodal-no-" + '-'.join(words[4:7]) + "-lru-4core"
        if token not in content:
            content[token] = []
            content[token].append([words[1], raw_data[key]])
        else:
            content[token].append([words[1], raw_data[key]])
    return content


def cloudsuits_filter(workload_name, workloads):
    no = workload_name.split('-')[1]
    return no, no in workloads


def mix_filter(workload_name, workloads):
    no = workload_name.split('-')[1] + "-"+workload_name.split('-')[2]
    return no, no in workloads


def mix_classify(raw_data):
    content = {}
    for key in raw_data:
        words = key.split(".")
        token = words[3]
        if token not in content:
            content[token] = []
            content[token].append([words[0]+"."+words[1], raw_data[key]])
        else:
            content[token].append([words[0]+"."+words[1], raw_data[key]])
    return content


def mix_het_filter(workload_name: str, workloads: list):
    no = "mix" + workload_name.split('-')[1]
    return no, workload_name.split('-')[1].isdecimal() and int(workload_name.split('-')[1]) <= 100


def mix_het_classify(raw_data):
    content = {}
    for key in raw_data:
        words = key.split(".")[0].split("-")
        token = "xz-" + '-'.join(words[2:9])
        if token not in content:
            content[token] = []
            content[token].append(["mix" + words[1], raw_data[key]])
        else:
            content[token].append(["mix" + words[1], raw_data[key]])
    return content

def show_performance(dirctory, model_names, workloads, name_labels=None,
                     workload_filter=default_filter,
                     data_classify=default_classify, standard="1core",
                     orderby='xz-bimodal-no-ipcp-ipcp-ipcp-lru-1core', size=[10, 3.5]):
    data = {}
    for root, dirs, files in os.walk(dirctory, topdown=False):
        for name in files:
            filepath = os.path.join(root, name)
            # if not "xz-bimodal-no-no-spp-no-lru-1core" in name and not "xz-bimodal-no-no-no-no-lru-1core" in name:
            #    continue
            no, filter_res = workload_filter(name, workloads)
            if not (filter_res and name.endswith(standard+".txt")):
                continue
            with open(filepath, "r") as res:
                for model in model_names:
                    if model in name:
                        content = res.read()
                        try:
                            pattern = re.compile(r"Finished CPU [0-9]+ instructions: "
                                                 "[0-9]+ cycles: [0-9]+ cumulative IPC:"
                                                 " ([./0-9]+) ")
                            result = pattern.findall(content)
                            if result:
                                ipcs = [float(r) for r in result]
                                data[name] = gmean(ipcs)
#                                 print(name, data[name])
                            else:
                                print("Cannot find IPC", name)
                        except (AttributeError, ZeroDivisionError) as e:
                            print(e, name)
    content = data_classify(data)
    labels = list(
        map(lambda x: x[0], content['xz-bimodal-no-no-no-no-lru-'+standard]))
    index = {l: i for i, l in enumerate(labels)}
    for key in content:
        if not "no-no-no-no-lru" in key:
            content[key].sort(key=lambda i: index[i[0]])
            for j in range(len(labels)):
                try:
                    content[key][j][1] /= content['xz-bimodal-no-no-no-no-lru-' +
                                                  standard][j][1]
                except BaseException as e:
                    print(e, key)

    labels = content[orderby]
    labels.sort(key=lambda x: x[1])
    labels = list(map(lambda x: x[0], labels))
    index = {l: i for i, l in enumerate(labels)}
    x = np.arange(len(labels))
    width = 0.1
    fig, ax = plt.subplots()
    linestyles = ['-', '--', ':']
    markers = ['o', '>', '*', '.']
    del content['xz-bimodal-no-no-no-no-lru-'+standard]

    perf = [(k, gmean(list(map(lambda i: i[1], content[k])))) for k in content]
    perf.sort(key=lambda x: x[1])
    for i, item in enumerate(perf):
        key = item[0]
        content[key].sort(key=lambda i: index[i[0]])
        label = key.replace('xz-bimodal-', '')
        if name_labels:
            label = name_labels[i]
        ax.plot(x, list(map(lambda i: i[1], content[key])), label=label,
                #                 color='%.01f' % (.99 - i/len(perf)),
                linestyle=linestyles[i % len(linestyles)])

    font = {"weight": "normal", "size": 15}
    ax.set_ylabel('Normalized IPC', font)
    ax.set_xlabel('Workloads', font)
    #ax.set_title('IPC Performance per Workloads')
#     ax.set_xticks(x)
#     ax.set_xticklabels(labels)
    ax.legend(prop=font)
    plt.setp(ax.get_xticklabels(), rotation=90)
    fig.set_size_inches(size[0], size[1])
    return content

# compute the avg IPC of every prefetchers


def draw_plot(datas, x_labels: list, name_x_labels: list, y_label: str, order_labels: list,
              name_labels: list, up_limit, *fig_size):
    fig, ax = plt.subplots()

    x = np.arange(len(x_labels))
    width = 0.9 / len(datas)
    addition = []
    for i in range(len(datas)):
        addition.append(i * width - ((len(datas)-1)*width)/2)
    for i, k in enumerate(order_labels):
        label = k
        if name_labels:
            label = name_labels[i]
        ax.bar(x + addition[i], datas[k], width, label=label,
               color='%.01f' % ((len(datas)-i-1)/len(datas)))
        for j, d in enumerate(datas[k]):
            if d >= up_limit:
                ax.annotate(
                    "%.02f" % d,
                    xy=(x[j]+addition[i], up_limit), xycoords='data',
                    xytext=(40, -5), textcoords='offset points',
                    arrowprops=dict(arrowstyle="->"),
                    horizontalalignment='right',
                    verticalalignment='top')
    font1 = {
        'family': 'Times New Roman',
        'weight': 'normal',
        'size': 12,
    }
    ax.set_ylabel(y_label, font1)
    ax.set_xticks(x)
    if name_x_labels:
        ax.set_xticklabels(name_x_labels)
    else:
        ax.set_xticklabels(x_labels)
#     if len(name_labels) > 1:
    ax.legend(ncol=len(datas), bbox_to_anchor=(
        0, 1), loc='lower left', prop=font1)

    plt.setp(ax.get_xticklabels(), rotation=90)
    fig.set_size_inches(*fig_size)


def show_mean_performance_list(content):
    GM = {}
    for key in content:
        GM[key] = gmean(list(map(lambda x: x[1], content[key])))
    keys = list(content.keys())
    keys.sort(key=lambda x: GM[x], reverse=True)
    result = []
    for k in keys:
        result.append((k, GM[k]))
    return result


def get_mpki_data(model, workloads):
    return analyze_output("./results_200M/", model, workloads, get_mpki, result_file_filter)


# def get_prefetch_data(model, workloads):
    # return analyze_output("./results_200M", model, workloads, extract_default_status, workload_filter)


def get_prefetch_latency(model, workloads):
    return analyze_output("./results_200M", model, workloads, extract_prefetch_miss_latency, result_file_filter)


# def get_prefetch_rate(model, workloads):
#     access = analyze_output("./results_200M/", model,
#                             workloads, get_all_load_access, workload_filter)
#     prefetch = analyze_output(
#         "./results_200M", model, workloads, extract_default_status, workload_filter)
#     result = {}
#     caches = ["L1D", "L2C", "LLC"]
#     for key in workloads:
#         tmp_result = {}
#         for cache in caches:
#             tmp_result[cache + " Prefetch Rate"] = prefetch[key][cache +
#                                                                  " Prefetch Issued"] / access[key][cache + " LOAD ACCESS"]
#         result[key] = tmp_result
#     return result


def get_timeliness(model, workloads):
    stat_data = analyze_output(
        "./results_200M", model, workloads, extract_timeliness_stat, result_file_filter)
    result = {}
    for key in stat_data:
        result[key] = {
            "In Time Rate":
            1-(stat_data[key]['total_late'] / (stat_data[key]['total_late']
                                               + stat_data[key]['total_useful']) if (stat_data[key]['total_late']
                                                                                     + stat_data[key]['total_useful']) else 1)
        }
    return result


def get_memory_traffic(model, workloads):
    baseline = analyze_output("./results_200M/", "no-no-no-no-lru",
                              workloads, get_bandwidth_cost, result_file_filter)
    dbus = analyze_output("./results_200M/", model, workloads,
                          get_bandwidth_cost, result_file_filter)
    result = {}
    for key in workloads:
        result[key] = {"Addtional Memory Traffic": dbus[key]
                       ['bandwidth']/baseline[key]['bandwidth'] - 1}
    return result


def order_by_label(data, ordered_labels: list):
    result = {}
    for k in data:
        result[k] = [data[k][label] for label in ordered_labels]
    return result


def draw_measurements(models, workloads, shown_labels, feature, func,
                      limit, withx, *figure_size):
    # show_avg_status(models, workloads, func).sort_values(by=feature, ascending=reverse).index
    ordered_models = models
    datas = {}
    for model in ordered_models:
        datas[model] = transform_to_df(func(model, workloads))[feature]
        datas[model]['mean'] = datas[model].mean()

    x_labels = list(map(lambda x: x.replace(
        ".champsimtrace.xz", ""), workloads)) + ['mean']
    if withx:
        draw_plot(order_by_label(datas, workloads+['mean']), x_labels, [], "Rate",
                  ordered_models,
                  shown_labels,
                  limit, *figure_size)
    else:
        draw_plot(order_by_label(datas, workloads+['mean']), [""]*46, [], "Rate",
                  ordered_models,
                  shown_labels,
                  limit, *figure_size)


def get_performance(model, workloads, target_dir):
    # default filter
    res = analyze_output(target_dir, model, workloads,
                         extract_ipc, result_file_filter)
    return res


def get_normalized_ipc_from_file(model, workloads, target_dir, enlarge=False):
    data = get_performance(model, workloads, target_dir)
    base = get_performance(
        BASE_LINE_MODEL, workloads, target_dir)
    enlarge_base = get_performance(
        "bimodal-no-L1DEnlarge-no-no-lru-1core", workloads, target_dir)
    norm_ipc = {}
    if enlarge:
        for key in data:
            norm_ipc[key] = {'Normalized IPC': data[key]
                             ['IPC']/enlarge_base[key]['IPC']}
    else:
        for key in data:
            norm_ipc[key] = {'Normalized IPC': data[key]
                             ['IPC']/base[key]['IPC']}
    return norm_ipc


def yield_index(data, reverse, keyword=None):
    pairs = []
    for k in data:
        if keyword:
            pairs.append([k, data[k][keyword]])
        else:
            pairs.append([k, data[k]])
    pairs.sort(key=lambda x: x[1], reverse=reverse)
    return {x[0]: i for i, x in enumerate(pairs)}

# def draw_normalized_ipcs(reformat_ipcs, index_by='bimodal-no-no-no-no-lru-1core',
#                          reverse=False, fixed_index=None, size=[10, 3.5]):
#     reformat_ipcs = reformat_ipcs
#     if not fixed_index:
#         fixed_index = yield_index(reformat_ipcs[index_by], reverse)
#     fig, ax = plt.subplots()
#     linestyles = ['-', '--', ':']
#     x = np.arange(len(fixed_index))
#     for i, model in enumerate(reformat_ipcs):
#         if model == 'bimodal-no-no-no-no-lru-1core':
#             continue
#         pairs = []
#         for k in fixed_index:
#             pairs.append([k, reformat_ipcs[model][k]])
#         pairs.sort(key=lambda x: fixed_index[x[0]])
#         ipcs = [x[1] for x in pairs]
#         ax.plot(x, ipcs, label=model,
#                 linestyle=linestyles[i % len(linestyles)])
#     font = {"weight": "normal", "size": 15}
#     ax.set_ylabel('Normalized IPC', font)
#     ax.set_xlabel('Workloads', font)
#     ax.legend(prop=font)
#     plt.setp(ax.get_xticklabels(), rotation=90)
#     fig.set_size_inches(size[0], size[1])


# def rank(df):
#     scores = []
#     for model in df.T.index:
#         if model != 'bimodal-no-no-no-no-lru-1core':
#             scores.append((model, gmean(df[model])))
#     scores.sort(key=lambda x: x[1])
#     return scores


def jaccard_simularity(pattern1, pattern2):
    return count_bits(pattern1 & pattern2, 32)/count_bits(pattern1 | pattern2, 32)


# def draw_normalized_ipcs_bars(reformat_ipcs, index_by='bimodal-no-no-no-no-lru-1core',
#                               reverse=False, fixed_index=None, upLimit=3.2, size=[16, 2]):
#     reformat_ipcs = reformat_ipcs
#     if not fixed_index:
#         fixed_index = yield_index(reformat_ipcs[index_by], reverse)

#     fig, ax = plt.subplots()
#     width = 0.9 / len(reformat_ipcs)
#     addition = []
#     for i in range(len(reformat_ipcs)):
#         addition.append(i * width - ((len(reformat_ipcs)-1)*width)/2)
#     x = np.arange(len(fixed_index))
#     for i, model in enumerate(reformat_ipcs):
#         if model == 'bimodal-no-no-no-no-lru-1core':
#             continue
#         pairs = []
#         for k in fixed_index:
#             pairs.append([k, reformat_ipcs[model][k]])
#         pairs.sort(key=lambda x: fixed_index[x[0]])
#         ipcs = [x[1] for x in pairs]
#         ax.bar(x + addition[i], ipcs, width, label=model, color='%.01f' %
#                ((len(reformat_ipcs)-i-1)/len(reformat_ipcs)))
#         for j, d in enumerate(ipcs):
#             if d >= upLimit:
#                 ax.annotate(
#                     "%.02f" % d,
#                     xy=(x[j]+addition[i], upLimit), xycoords='data',
#                     xytext=(40, -5), textcoords='offset points',
#                     arrowprops=dict(arrowstyle="->"),
#                     horizontalalignment='right',
#                     verticalalignment='top')

#     font = {"weight": "normal", "size": 15}
#     ax.set_ylabel('Normalized IPC', font)
#     ax.set_xlabel('Workloads', font)
#     ax.set_xticks(x)
#     x_labels = list(fixed_index.keys())
#     x_labels.sort(key=lambda x: fixed_index[x])
#     ax.set_xticklabels([x.replace('_champsimtrace_xz', '') for x in x_labels])
#     ax.legend(ncol=3, bbox_to_anchor=(0, 1), loc='lower left', prop=font)
#     plt.setp(ax.get_xticklabels(), rotation=90)
#     plt.ylim(.9, 3.2)
#     fig.set_size_inches(size[0], size[1])

def draw_data(format_data, keyword, *size, index_by=None, reverse=False, fixed_index=None):
    dfs = convert2df(format_data)
    fig, ax = plt.subplots()
    linestyles = ['-.', '-', '--', ':']
    if index_by and not fixed_index:
        fixed_index = yield_index(format_data[index_by], reverse, keyword)
    for i, model in enumerate(dfs):
        x = np.arange(len(dfs[model][keyword]))
        if not fixed_index:
            ax.plot(np.arange(len(dfs[model][keyword])), dfs[model][keyword].sort_index(), 
            label=model, linestyle=linestyles[i % len(linestyles)])
        else:
            x_labels = dfs[model][keyword].index.to_list()
            x_labels.sort(key=lambda x: fixed_index[x])
            ax.plot(np.arange(len(dfs[model][keyword])), [dfs[model][keyword][l] for l in x_labels], 
            label=model, linestyle=linestyles[i % len(linestyles)])

    ax.set_xticks(x)
    ax.legend()
    ax.set_xlabel('Workloads')
    ax.set_ylabel(keyword)
    plt.setp(ax.get_xticklabels(), rotation=90)
    fig.set_size_inches(*size)


def draw_data_bars(format_data, keyword, *size, index_by=None, reverse=False, fixed_index=None):
    dfs = convert2df(format_data)
    fig, ax = plt.subplots()
    addition = []
    width = 0.9/len(format_data)
    for i in range(len(format_data)):
        addition.append(i * width - ((len(format_data)-1)*width)/2)
    if index_by and not fixed_index:
        fixed_index = yield_index(format_data[index_by], reverse, keyword)
    for i, model in enumerate(dfs):
        x = np.arange(len(dfs[model][keyword]))
        if not fixed_index:
            ax.bar(np.arange(len(dfs[model][keyword])) + addition[i],
                   dfs[model][keyword].sort_index(), width, label=model)
        else:
            x_labels = dfs[model][keyword].index.to_list()
            x_labels.sort(key=lambda x: fixed_index[x])
            ax.bar(np.arange(len(dfs[model][keyword])) + addition[i], [dfs[model][keyword][l] for l in x_labels], width,
            label=model)

    font = {"weight": "normal", "size": 15}
    ax.set_xlabel('Workloads', font)
    ax.set_ylabel(keyword, font)
    ax.set_xticks(x)
    ax.set_xticklabels([x.replace('_champsimtrace_xz', '')
                       for x in dfs[model][keyword].sort_index().index])
    ax.legend(ncol=2, bbox_to_anchor=(0, 1), loc='lower left', prop=font)
    plt.setp(ax.get_xticklabels(), rotation=90)
    fig.set_size_inches(*size)


@extracter([
    ["DeadRegionCollector.is_dead.yes:([0-9]*)", "is_dead.yes"],
    ["DeadRegionCollector.is_dead.no:([0-9]*)", "is_dead.no"],
    ["DeadRegionCollector.update.insert:([0-9]*)", "update.insert"],
    ["DeadRegionCollector.update.update:([0-9]*)", "update.update"],
    ["DeadRegionCollector.update.replace_dead:([0-9]*)", "update.replace_dead"],
    ["DeadRegionCollector.is_bad.yes:([0-9]*)", "is_bad.yes"],
    ["DeadRegionCollector.is_bad.no:([0-9]*)", "is_bad.no"], 
    ["DeadRegionCollector.is_nice.yes:([0-9]*)", "is_nice.yes"],
    ["DeadRegionCollector.is_nice.no:([0-9]*)", "is_nise.no"],
])
def extract_dead_region_stats(content: str):
    pass

def get_bandwidth_increment_by_env(data, core_n=1, raise_err=False):
    def get_bandwidth_cost(d):
        return d['DBUS_CONGESTED'] + d['RQ ROW_BUFFER_HIT'] + d['RQ ROW_BUFFER_MISS'] \
               + d['WQ ROW_BUFFER_HIT'] + d['WQ ROW_BUFFER_MISS']
    def get_bandwidth_increment(data, baseline):
        return {"Bandwidth Increment":get_bandwidth_cost(data)/get_bandwidth_cost(baseline)}
    
    return get_results_by_env(get_bandwidth_increment, data, core_n, raise_err)


def get_bandwidth_increment(data):
    assert(BASE_LINE_MODEL in data)
    result = {}
    baseline = data[BASE_LINE_MODEL]
    def get_bandwidth_cost(d):
        return d['DBUS_CONGESTED'] + d['RQ ROW_BUFFER_HIT'] + d['RQ ROW_BUFFER_MISS'] \
               + d['WQ ROW_BUFFER_HIT'] + d['WQ ROW_BUFFER_MISS']
    
    for k in data:
        if k == BASE_LINE_MODEL:
            continue
        one = {}
        for trace in data[k]:
            coverages = {}
            try:
                increament = get_bandwidth_cost(data[k][trace])/get_bandwidth_cost(baseline[trace])
            except Exception as e:
                print(k, trace, e)
                raise e
            one[trace] = {"Bandwidth Increment":increament}
        result[k] = one
    return result

def get_mean_df(data):
    dfs = convert2df(data)
    coverage_df = {}
    for model in dfs:
        coverage_df[model] = dfs[model].mean()
    return pd.DataFrame(coverage_df)
