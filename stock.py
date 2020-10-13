def cal_upper(price):
    increment = price * 0.3
    upper_price = price + increment
    return upper_price

def cal_lower(price):
    decrement = price * 0.3
    lower_price = price - decrement
    return lower_price

author = 'paystock'

print(cal_upper(10000))
print(cal_lower(10000))
print(__name__)