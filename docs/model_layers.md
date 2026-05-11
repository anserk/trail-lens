



## Conv2D Formula (PyTorch Docs)
For a convolution layer:


$$
H_{out} =
\left\lfloor
\frac{
H_{in} + 2P - D(K - 1) - 1
}{
S
}
+ 1
\right\rfloor
$$

Where:

- `H_in` = input size
- `H_out` = output size
- `K` = kernel size
- `P` = padding
- `S` = stride
- `D` = dilation

---

## Simplified Formula

When:

- `padding = 0`
- `stride = 1`
- `dilation = 1`

the formula simplifies to:

$$
H_{out} = H_{in} - K + 1
$$

## MaxPool2D Formula

$$
H_{out} =
\left\lfloor
\frac{
H_{in} + 2P - D(K - 1) - 1
}{
S
}
+ 1
\right\rfloor
$$

Where:

- `H_in` = input size
- `H_out` = output size
- `K` = kernel size
- `P` = padding
- `S` = stride
- `D` = dilation

---

## Simplified Formula

when:

- `K == S`
- `P = 0`
- `D = 1`

Substitute:

$$
H_{out} =
\left\lfloor
\frac{
H_{in} - 2
}{
2
}
+ 1
\right\rfloor
$$

you can usually think:

$$
H_{out} \approx \frac{H_{in}}{2}
$$

because it halves width and height.

### Example

## Conv2D 1
*using simplified formula*
$$
128 - 5 + 1 = 124
$$

$$
6 x 124 x 124
$$

## MaxPool2D Output Size
*using simplified formula*


$$
H_{out} =
\left\lfloor
\frac{
124 - 2
}{
2
}
+ 1
\right\rfloor
$$

$$
=
\left\lfloor
61 + 1
\right\rfloor
=
62
$$

$$
6 x 62 x 62
$$

## Conv2D 2
*using simplified formula*

$$
62 - 5 + 1 = 58
$$

$$
16 x 58 x 58
$$

## MaxPool2D Output Size

$$
58 / 2 = 29
$$

$$
16 x 29 x 29
$$

## Flatten operation:

$$
16 x 29 x 29 = 13456
$$

## fc1

$$
120
$$

## fc2

$$
84
$$

## fc3

$$
10
$$

matching our classes count