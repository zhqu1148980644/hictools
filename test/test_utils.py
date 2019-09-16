import sys
import pytest

backup = sys.path
print(backup)
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
