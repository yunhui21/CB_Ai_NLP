# Day_29_01_BackPropagation.py


class MulLayer:
    def forward(self, x, y):
        self.x = x
        self.y = y
        return x * y

    def backward(self, dout):
        dx = dout * self.y
        dy = dout * self.x
        return dx, dy


class AddLayer:
    def forward(self, x, y):
        return x + y

    def backward(self, dout):
        return dout, dout


def apple_network():
    apple_price = 100
    apple_count = 2
    tax = 1.1

    layer_apple = MulLayer()
    layer_tax = MulLayer()

    # 순전파
    apple_total = layer_apple.forward(apple_price, apple_count)
    price = layer_tax.forward(apple_total, tax)

    # 역전파
    d_price = 1
    d_apple_total, d_tax = layer_tax.backward(d_price)
    d_apple_price, d_apple_count = layer_apple.backward(d_apple_total)

    print('가격 :', price)
    print('미분 :', d_apple_price, d_apple_count, d_tax)


def fruit_network():
    apple_price = 100
    apple_count = 2
    mango_price = 150
    mango_count = 3
    tax = 1.1

    layer_apple = MulLayer()
    layer_mango = MulLayer()
    layer_fruit = AddLayer()
    layer_tax = MulLayer()

    # 순전파
    apple_total = layer_apple.forward(apple_price, apple_count)
    mango_total = layer_mango.forward(mango_price, mango_count)
    fruit_total = layer_fruit.forward(apple_total, mango_total)
    price = layer_tax.forward(fruit_total, tax)

    # 역전파
    d_price = 1
    d_fruit_total, d_tax = layer_tax.backward(d_price)
    d_apple_total, d_mango_total = layer_fruit.backward(d_fruit_total)
    d_mango_price, d_mango_count = layer_mango.backward(d_mango_total)
    d_apple_price, d_apple_count = layer_apple.backward(d_apple_total)

    print('가격 :', price)
    print('미분 :', d_apple_price, d_apple_count, d_tax)
    print('미분 :', d_mango_price, d_mango_count)


def back_propagation(apple_price, mango_price):
    apple_count = 2
    mango_count = 3
    tax = 1.1
    target = 715

    # apple_price = 10
    # mango_price = 15

    layer_apple = MulLayer()
    layer_mango = MulLayer()
    layer_fruit = AddLayer()
    layer_tax = MulLayer()

    for i in range(1000):
        # 순전파
        apple_total = layer_apple.forward(apple_price, apple_count)
        mango_total = layer_mango.forward(mango_price, mango_count)
        fruit_total = layer_fruit.forward(apple_total, mango_total)
        price = layer_tax.forward(fruit_total, tax)
        hx = price

        loss = (hx - target) ** 2
        # print('loss :', loss)

        # 역전파
        # gradient = (hx - target) * x
        d_price = hx - target
        d_fruit_total, d_tax = layer_tax.backward(d_price)
        d_apple_total, d_mango_total = layer_fruit.backward(d_fruit_total)
        d_mango_price, d_mango_count = layer_mango.backward(d_mango_total)
        d_apple_price, d_apple_count = layer_apple.backward(d_apple_total)

        apple_price -= 0.1 * d_apple_price
        mango_price -= 0.1 * d_mango_price

    print('apple :', apple_price)
    print('mango :', mango_price)
    print('price :',
          (apple_price * apple_count + mango_price * mango_count) * tax)
    print()

# apple_network()
# fruit_network()

back_propagation(10, 15)
back_propagation(10, -15)
back_propagation(-10, 15)
back_propagation(-10, -15)


