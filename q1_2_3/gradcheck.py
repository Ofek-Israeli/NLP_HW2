import random

import numpy as np
from numpy.testing import assert_allclose


def gradcheck_naive(f, x, gradient_text=""):
    """ Gradient check for a function f.
    Arguments:
    f -- a function that takes a single argument and outputs the
         loss and its gradients
    x -- the point (numpy array) to check the gradient at
    gradient_text -- a string detailing some context about the gradient computation
    """

    rndstate = random.getstate()
    random.setstate(rndstate)
    fx, grad = f(x)  # Evaluate function value at original point
    h = 1e-4         # Do not change this!

    # Iterate over all indexes ix in x to check the gradient.
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index

        # Try modifying x[ix] with h defined above to compute numerical
        # gradients (numgrad).

        # Use the centered difference of the gradient.
        # It has smaller asymptotic error than forward / backward difference
        # methods. If you are curious, check out here:
        # https://math.stackexchange.com/questions/2326181/when-to-use-forward-or-central-difference-approximations

        # Make sure you call random.setstate(rndstate)
        # before calling f(x) each time. This will make it possible
        # to test cost functions with built in randomness later.

        ### YOUR CODE HERE:
        # raise NotImplementedError
        x[ix] += h  # increment by h
        random.setstate(rndstate)
        fxh, _ = f(x)  # evaluate f(x + h)
        x[ix] -= 2 * h  # increment by 2h
        random.setstate(rndstate)
        fxnh, _ = f(x)  # evaluate f(x - h)
        x[ix] += h  # restore original value
        numgrad = (fxh - fxnh) / 2 / h
        ### END YOUR CODE

        # Compare gradients
        assert_allclose(numgrad, grad[ix], rtol=1e-5,
                        err_msg=f"Gradient check failed for {gradient_text}.\n"
                                f"First gradient error found at index {ix} in the vector of gradients\n"
                                f"Your gradient: {grad[ix]} \t Numerical gradient: {numgrad}")

        it.iternext()  # Step to next dimension

    print("Gradient check passed!")


def test_gradcheck_basic():
    """
    Some basic sanity checks.
    """
    quad = lambda x: (np.sum(x ** 2), 2*x)

    print("Running sanity checks...")
    gradcheck_naive(quad, np.array(123.456))       # scalar test
    gradcheck_naive(quad, np.random.randn(3,))     # 1-D test
    gradcheck_naive(quad, np.random.randn(4, 5))   # 2-D test
    print()


def your_gradcheck_test():
    """
    Use this space add any additional sanity checks by running:
        python q2_gradcheck.py
    This function will not be called by the autograder, nor will
    your additional tests be graded.
    """
    print("Running your sanity checks...")
    ### YOUR OPTIONAL CODE HERE
    # Test 1: Linear function f(x) = a^T x + b
    # For a fixed vector a and scalar b, gradient is a
    a = np.array([1.5, -2.3, 0.7, 3.1])
    b = 5.0
    linear_func = lambda x: (np.dot(a, x) + b, a)
    gradcheck_naive(linear_func, np.random.randn(4,), "linear function")
    
    # Test 2: Exponential function f(x) = sum(exp(x))
    # Gradient is exp(x) (element-wise)
    exp_func = lambda x: (np.sum(np.exp(x)), np.exp(x))
    gradcheck_naive(exp_func, np.random.randn(5,), "exponential function")
    
    # Test 3: Product function f(x) = prod(x) = product of all elements
    # Gradient: for each element i, gradient[i] = prod(x) / x[i]
    def product_func(x):
        prod = np.prod(x)
        grad = np.zeros_like(x)
        for i in range(len(x)):
            grad[i] = prod / x[i]
        return prod, grad
    gradcheck_naive(product_func, np.random.randn(4,) + 1.0, "product function")  # +1 to avoid zeros
    
    # Test 4: Sum of squares function f(x) = sum(x^2)
    # Gradient is 2*x
    sum_squares = lambda x: (np.sum(x ** 2), 2 * x)
    gradcheck_naive(sum_squares, np.random.randn(3, 2), "sum of squares (2D)")
    
    # Test 5: Edge case - very small values
    # Using sum of squares function
    small_vals = np.array([1e-8, 1e-9, 1e-10])
    gradcheck_naive(sum_squares, small_vals, "very small values")
    
    # Test 6: Edge case - mixed positive and negative values (no zero to avoid issues)
    mixed_vals = np.array([-5.0, 0.001, 5.0, -0.001, 0.001])
    gradcheck_naive(sum_squares, mixed_vals, "mixed positive/negative values")
    
    print()
    ### END YOUR CODE


if __name__ == "__main__":
    test_gradcheck_basic()
    your_gradcheck_test()
