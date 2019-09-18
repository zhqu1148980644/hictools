import sys

import pytest

sys.path.insert(0, '../')


def _test_multi_methods_with_dups(self):
    from hictools.utils.utils import multi_methods
    with pytest.raises(RuntimeError):
        class Test(object):
            @multi_methods()
            def method_a(self):
                pass

            @method_a
            def a1(self):
                pass

            @method_a()
            def a1(self):
                pass


def _test_multi_methods_features(self):
    from hictools.utils.utils import multi_methods
    class Test(object):
        def __init__(self, name, msg):
            self.name = name
            self.msg = msg

        @multi_methods
        def method_a(self):
            """There are {num} A methods. {methods}"""

        @method_a
        def a1(self, sample_name, balance=False):
            """This is a1 method"""
            return sample_name, balance, self.name, self.msg

        @method_a()
        def a2(self, normalize, show=False, balance=False):
            """This is a2 method"""
            return normalize, show, balance, self.name, self.msg

        @multi_methods
        def method_b(self):
            """There are {num} B methods. {methods}"""

        @method_b.register()
        def b1(self, numvecs=3, balance=False):
            """This is b1 method."""
            return numvecs, balance, self.name, self.msg

        @method_b.register
        def b2(self, **kwargs):
            """This is b2 method."""
            return kwargs

    test = Test('NAME', 'TEST MSG')
    assert test.name == 'NAME'
    assert test.msg == 'TEST MSG'

    for key, func in Test.__dict__['method_a']._methods.items():
        assert func.__doc__ in test.method_a.__doc__

    assert (test.method_a.a1(sample_name='test_sample', balance=False)
            == ('test_sample', False, 'NAME', 'TEST MSG'))
    assert (test.method_a().a1(sample_name='test_sample', balance=False)
            == ('test_sample', False, 'NAME', 'TEST MSG'))

    balanced_method_b = test.method_b(balance=True)
    assert balanced_method_b.b1(numvecs=4) == (4, True, 'NAME', 'TEST MSG')

    assert (balanced_method_b.b2(test='test', name='123')
            == {'balance': True, 'test': 'test', 'name': '123'})


class TestMultiMethods(object):
    test_multi_methods_with_dups = _test_multi_methods_with_dups
    test_multi_methods_features = _test_multi_methods_features


class TestRayWrap(object):

    def test_mimic_actor(self):
        from hictools.utils.utils import RayWrap, MethodWithRemote
        from types import FunctionType
        ray = RayWrap(enable_ray=False)

        class A(object):
            def mth1(self, x):
                return x

        A_ = ray.remote(A)
        a = A_.remote()
        assert type(A_.mth1) is FunctionType
        assert type(a.mth1) is MethodWithRemote
        with pytest.raises(Exception):
            a.mth1(1)
        assert ray.get(a.mth1.remote(1)) == 1


class TestNumTools(object):

    def test_convolve1d(self, load_pyx):
        import timeit
        import random
        from functools import partial
        import numpy as np
        from hictools.utils.numtools import convolve1d as cy
        from scipy.ndimage import convolve1d as sci_convolve1d
        sci = partial(sci_convolve1d, mode='constant')
        array = np.random.rand(10000).reshape(100, 100)
        kernel = np.random.randn(11)
        array1 = np.random.rand(10000)
        kernel1 = np.random.randn(21)
        random_cval = random.random() * 10
        result_cy = cy(array, kernel)
        result_sci = sci(array, kernel)
        # test convolve1d functions properly.
        assert np.allclose(result_cy, result_sci)
        assert np.allclose(cy(array, kernel, cval=random_cval),
                           sci(array, kernel, cval=random_cval))

        assert np.allclose(cy(array, kernel, axis=0),
                           sci(array, kernel, axis=0))

        assert np.allclose(cy(array1, kernel1, cval=random_cval),
                           sci(array1, kernel1, cval=random_cval))

        assert np.allclose(cy(array1, kernel1, axis=0),
                           sci(array1, kernel1, axis=0))

        # test convolve1d support points and nonzero options.
        indices = np.vstack(np.nonzero(array))
        mask = (indices[0] > 10) & (indices[0] < 90) & (indices[1] > 10) & (indices[1] < 90)
        indices = indices[:, mask].T
        np.random.shuffle(indices)
        points = indices[100: 200].T
        assert np.allclose(cy(array, kernel, points=points),
                           result_cy[points[0], points[1]])
        assert np.allclose(cy(array, kernel, points=points),
                           sci(array, kernel)[points[0], points[1]])

        array[points[0], points[1]] = 0
        result_cy = cy(array, kernel, nonzero=True)
        assert np.allclose(result_cy[points[0], points[1]],
                           array[points[0], points[1]])

        array = np.arange(1000000).reshape(1000, 1000).astype(np.float32)
        kernel = np.random.randn(11).astype(np.float32)
        time1 = timeit.timeit("cy(array, kernel)", number=200, globals=locals())
        time2 = timeit.timeit("sci(array, kernel)", number=200, globals=locals())
        assert time1 < time2 * 1.1

    def test_gaussian_filter1d(self, load_pyx):
        from functools import partial
        import numpy as np
        from hictools.utils.numtools import gaussian_filter1d as gf_cy
        from scipy.ndimage import gaussian_filter1d
        gf_sci = partial(gaussian_filter1d, mode='constant')
        array = np.random.rand(10000).reshape(100, 100)

        assert np.allclose(gf_cy(array, sigma=2.567, axis=-1, cval=2.3, truncate=5.1),
                           gf_sci(array, sigma=2.567, axis=-1, cval=2.3, truncate=5.1))

        assert np.allclose(gf_cy(array, sigma=3.124, axis=0, cval=3.2, truncate=3.8),
                           gf_sci(array, sigma=3.124, axis=0, cval=3.2, truncate=3.8))

    def test_gaussian_filter(self, load_pyx):
        # TODO Fix error: Segmentation fault if kernel size is bigger than the array size.
        from functools import partial
        import numpy as np
        from hictools.utils.numtools import gaussian_filter as gf_cy
        from scipy.ndimage import gaussian_filter
        gf_sci = partial(gaussian_filter, mode='constant')
        array = np.random.rand(10000).reshape(100, 100)

        assert np.allclose(gf_cy(array, sigma=2.567, cval=2.3, truncate=5.1),
                           gf_sci(array, sigma=2.567, cval=2.3, truncate=5.1))
