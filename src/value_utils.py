dict_age = {
    -0.95197: '18-24',
    -0.07854: '25-34',
    0.49788: '35-44',
    1.09449: '45-54',
    1.82213: '55-64',
    2.59171: '65+'
}

dict_gender = {
    0.48246: 'Female',
    -0.48246: 'Male'
}

dict_education = {
    -2.43591: 'Left school before 16 years',
    -1.73790: 'Left school at 16 years',
    -1.43719: 'Left school at 17 years',
    -1.22751: 'Left school at 18 years',
    -0.61113: 'Some college or university, no certificate or degree',
    -0.05921: 'Professional certificate/ diploma',
    0.45468: 'University degree',
    1.16365: 'Masters degree',
    1.98437: 'Doctorate degree'
}

dict_country = {
    -0.09765: 'Australia',
    0.24923: 'Canada',
    -0.46841: 'New Zealand',
    -0.28519: 'Other',
    0.21128: 'Republic of Ireland',
    0.96082: 'UK',
    -0.57009: 'USA'
}

dict_ethnicity = {
    -0.50212: 'Asian',
    -1.10702: 'Black',
    1.90725: 'Mixed-Black/Asian',
    0.12600: 'Mixed-White/Asian',
    -0.22166: 'Mixed-White/Black',
    0.11440: 'Other',
    -0.31685: 'White'
}

dict_nscore = {
    12: -3.46436,
    13: -3.15735,
    14: -2.75696,
    15: -2.52197,
    16: -2.42317,
    17: -2.34360,
    18: -2.21844,
    19: -2.05048,
    20: -1.86962,
    21: -1.69163,
    22: -1.55078,
    23: -1.43907,
    24: -1.32828,
    25: -1.19430,
    26: -1.05308,
    27: -0.92104,
    28: -0.79151,
    29: -0.67825,
    30: -0.58016,
    31: -0.46725,
    32: -0.34799,
    33: -0.24649,
    34: -0.14882,
    35: -0.05188,
    36: 0.04257,
    37: 0.13606,
    38: 0.22393,
    39: 0.31287,
    40: 0.41667,
    41: 0.52135,
    42: 0.62967,
    43: 0.73545,
    44: 0.82562,
    45: 0.91093,
    46: 1.02119,
    47: 1.13281,
    48: 1.23461,
    49: 1.37297,
    50: 1.49158,
    51: 1.60383,
    52: 1.72012,
    53: 1.83990,
    54: 1.98437,
    55: 2.12700,
    56: 2.28554,
    57: 2.46262,
    58: 2.61139,
    59: 2.82196,
    60: 3.27393}
dict_nscore = {y: x for x, y in dict_nscore.items()}

dict_escore = {
    -3.27393: 16,
    -3.00537: 18,
    -2.72827: 19,
    -2.53830: 20,
    -2.44904: 21,
    -2.32338: 22,
    -2.21069: 23,
    -2.11437: 24,
    -2.03972: 25,
    -1.92173: 26,
    -1.76250: 27,
    -1.63340: 28,
    -1.50796: 29,
    -1.37639: 30,
    -1.23177: 31,
    -1.09207: 32,
    -0.94779: 33,
    -0.80615: 34,
    -0.69509: 35,
    -0.57545: 36,
    -0.43999: 37,
    -0.30033: 38,
    -0.15487: 39,
    0.00332: 40,
    0.16767: 41,
    0.32197: 42,
    0.47617: 43,
    0.63779: 44,
    0.80523: 45,
    0.96248: 46,
    1.11406: 47,
    1.28610: 48,
    1.45421: 49,
    1.58487: 50,
    1.74091: 51,
    1.93886: 52,
    2.12700: 53,
    2.32338: 54,
    2.57309: 55,
    2.85950: 56,
    3.00537: 58,
    3.27393: 59}

dict_oscore = {
    24: -3.27393,
    38: -1.11902,
    50: 0.58331,
    26: -2.85950,
    39: -0.97631,
    51: 0.72330,
    28: -2.63199,
    40: -0.84732,
    52: 0.88309,
    29: -2.39883,
    41: -0.71727,
    53: 1.06238,
    30: -2.21069,
    42: -0.58331,
    54: 1.24033,
    31: -2.09015,
    43: -0.45174,
    55: 1.43533,
    32: -1.97495,
    44: -0.31776,
    56: 1.65653,
    33: -1.82919,
    45: -0.17779,
    57: 1.88511,
    34: -1.68062,
    46: -0.01928,
    58: 2.15324,
    35: -1.55521,
    47: 0.14143,
    59: 2.44904,
    36: -1.42424,
    48: 0.29338,
    60: 2.90161,
    37: -1.27553,
    49: 0.44585}
dict_oscore = {y: x for x, y in dict_oscore.items()}

dict_ascore = {
    12: -3.46436,
    34: -1.34289,
    48: 0.76096,
    16: -3.15735,
    35: -1.21213,
    49: 0.94156,
    18: -3.00537,
    36: -1.07533,
    50: 1.11406,
    23: -2.90161,
    37: -0.91699,
    51: 1.2861,
    24: -2.78793,
    38: -0.76096,
    52: 1.45039,
    25: -2.70172,
    39: -0.60633,
    53: 1.61108,
    26: -2.53830,
    40: -0.45321,
    54: 1.81866,
    27: -2.35413,
    41: -0.30172,
    55: 2.03972,
    28: -2.21844,
    42: -0.15487,
    66: 2.23427,
    29: -2.07848,
    43: -0.01729,
    57: 2.46262,
    30: -1.92595,
    44: 0.13136,
    58: 2.75696,
    31: -1.77200,
    45: 0.28783,
    59: 3.15735,
    32: -1.62090,
    46: 0.43852,
    60: 3.46436,
    33: -1.47955,
    47: 0.59042}
dict_ascore = {y: x for x, y in dict_ascore.items()}

dict_cscore = {
    17: -3.46436,
    32: -1.25773,
    46: 0.58489,
    19: -3.15735,
    33: -1.13788,
    47: 0.7583,
    20: -2.90161,
    34: -1.01450,
    48: 0.93949,
    21: -2.72827,
    35: -0.89891,
    49: 1.13407,
    22: -2.57309,
    36: -0.78155,
    50: 1.30612,
    23: -2.42317,
    37: -0.65253,
    51: 1.46191,
    24: -2.30408,
    38: -0.52745,
    52: 1.63088,
    25: -2.18109,
    39: -0.40581,
    53: 1.81175,
    26: -2.04506,
    40: -0.27607,
    54: 2.04506,
    27: -1.92173,
    41: -0.14277,
    55: 2.33337,
    28: -1.78169,
    42: -0.00665,
    56: 2.63199,
    29: -1.64101,
    43: 0.12331,
    57: 3.00537,
    30: -1.51840,
    44: 0.25953,
    59: 3.46436,
    31: -1.38502,
    45: 0.41594}
dict_cscore = {y: x for x, y in dict_cscore.items()}

# Dictionary for the Values of the consumptions of
# alcohol, amphetamines, amyl nitrite, benzodiazepine, caffeine, cannabis,
# chocolate, cocaine, crack, ecstasy, heroin, ketamine, LSD, methadone,
# magic, nicotine, Semeron (fictional) and volatile substances.
dict_consumption = {
    'CL0': 'Never Used',
    'CL1': 'Used over a Decade Ago',
    'CL2': 'Used in Last Decade',
    'CL3': 'Used in Last Year',
    'CL4': 'Used in Last Month',
    'CL5': 'Used in Last Week',
    'CL6': 'Used in Last Day'
}

