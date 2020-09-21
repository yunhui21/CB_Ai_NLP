# 041-050_strMethod.py

# 041
ticker = 'btc_krw'
Ticker = ticker.upper()
print('041:', Ticker)

# 042
Ticker = 'BTC_KRW'
ticker = Ticker.lower()
print('042:',ticker)

# 043
data = 'hello'
capi = data.capitalize()
print('043:', capi)

# 044
# 045
file_name = '보고서.xlsx'
file_chan = file_name.endswith('xlsx')
print('045:', file_chan)

# 045
file_name = '보고서.xlsx'
file_chan = file_name.endswith(('xlsx', 'xls'))
print('045:', file_chan)

# 046
file_name = '2020_보고서.xlsx'
file_chan = file_name.startswith('2020')
print('046:', file_chan)

# 047
a = 'hello world'
a_in = a.split()
print('047:', a_in)

# 048
ticker = 'btc_krw'
tikcer_1 = ticker.split('_')
print('048:', tikcer_1)

# 049
date = '2020-05-01'
data_1 = date.split('-')
print('049:', data_1)

# 050
data = '039490           '
data_2 = data.rstrip()
print('050:', data_2)