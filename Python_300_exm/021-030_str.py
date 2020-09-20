# 021-030_str.py

# 021
letters = 'python'
print('021:', letters[0], letters[2])

# 022
license_plate = '24가 2210'
print('022:', license_plate[-4:])

# 023
string = '훌쩍훌쩍훌쩍'
print('023:', string[::2])

# 024
py_str = 'python'
print('024:', py_str[::-1])

# 025
phone_number = '010-1111-2222'
# print('025:', phone_number[:3], phone_number[4:8], phone_number[9:])
phone_number = phone_number.replace('-', ' ')
print('025:', phone_number)

# 026
phone = '010-1111-2222'
phone_number1 = phone.replace('-','')
print('026:', phone_number1)

# 027
url = "http://shareeboo.kr"
# print('027:', url[-2:]) # kr
url_split = url.split('.')
# print(url_split)
print('027:',url_split[-1])

# 28
lang = 'python'
# lang[0] = 'i'
# print(lang) # typeError
print('028:','i'+lang[1:])

# 029
string = 'abcde2a354a32a'
re_string = string.replace('a', 'A')
print('029:', re_string)

# 030
string = 'abcd'
string.replace('b', 'B')
print('030:', string)