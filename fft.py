from cmath import sin, exp
from math import pi, sqrt

#Complex Utility Functions
def conj(z):
    return complex(z.real,-z.imag)
def array_conj(z_array):
    return [conj(z) for z in z_array]
def roots_of_unity(period):
    return [exp(-1j*2.*pi*j/period) for j in xrange(period)]

#Naive Discrete Fourier Transform
def poly_eval(coefs, xs):
    p = [0 for x in xs]
    for k, x in enumerate(xs):
        for j, c in enumerate(coefs):
            p[k]+=(c*pow(x,j))
    return p
def dft(xs):
    n = len(xs)
    return poly_eval(xs, roots_of_unity(n))
def idft(Xs):
    cXs = array_conj(Xs)
    nxs = array_conj(dft(cXs))
    const = 1./len(Xs)
    return [const*nx for nx in nxs]

#Fast Fourier Transform
def fft(xs):
    n = len(xs)
    if n%2==1:
        return dft(xs)
    xs_e = [xs[k] for k in xrange(0,n,2)]
    xs_o = [xs[k] for k in xrange(1,n,2)]
    fft_e = fft(xs_e)
    fft_o = fft(xs_o)
    tw_f = [exp(-1j*2.*pi*k/n) for k in xrange(n/2)]
    fft_t = [fft_e[k]+tw_f[k]*fft_o[k] for k in xrange(n/2)]
    fft_t+= [fft_e[k]-tw_f[k]*fft_o[k] for k in xrange(n/2)]
    return fft_t
def ifft(Xs):
    cXs = array_conj(Xs)
    nxs = array_conj(fft(cXs))
    const = 1./len(Xs)
    return [const*nx for nx in nxs]

#Convolutions
def convolve(a, b):
    """A naive method of calculating a discrete convolution"""
    x = [0 for k in xrange(len(a)+len(b))]
    for j in xrange(len(a)):
        for k in xrange(len(b)):
            x[j+k]+=a[j]*b[k]
    return x
def fconvolve(a,b):
    """A fast method of calculating a discrete convolution"""
    assert(len(a)==len(b))
    length = len(a)*2
    a2 = a+[0]*len(a)
    b2 = b+[0]*len(b)
    f_a = fft(a2)
    f_b = fft(b2)
    f_ab = [f_a[j]*f_b[j] for j in xrange(length)]
    return ifft(f_ab)

#Tests
def __dist__(z1, z2):
    return sqrt((z1.real-z2.real)**2+(z1.imag-z2.imag)**2)
def __test1__():
    n = 10
    xs = [sin(j) for j in xrange(n)]
    Xs = dft(xs)
    xs2 = idft(Xs)
    for j in xrange(n):
        assert __dist__(xs[j]-xs2[j], 0)<.0001
def __test2__():
    a = [3,6,4,2,9,1,0,3]
    b = [3,8,5,1,9,3,4,7]
    c = convolve(a,b)
    fc = fconvolve(a,b)
    for j in xrange(len(c)):
        assert __dist__(c[j]-fc[j],0)<.0001
def __test3__():
    n = 16
    xs = [sin(j) for j in xrange(n)]
    Xs = dft(xs)
    Xs2 = fft(xs)
    for j in xrange(n):
        assert __dist__(Xs[j]-Xs2[j], 0)<.0001

if __name__=="__main__":
    __test1__()
    __test2__()
    __test3__()
    print "All tests passed."
