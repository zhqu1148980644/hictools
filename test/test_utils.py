import sys

sys.path.append('../')

from hictools.utils import multi_methods


def test_multi_methods():
    class Test(object):
        def __init__(self, name, msg):
            self.name = name
            self.msg = msg

        @multi_methods()
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
            "There are {num} B methods. {methods}"

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


if __name__ == '__main__':
    test_multi_methods()
