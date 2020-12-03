import sys
sys.path.insert(0, '../')


def test_get_logger():
    from hictools.utils.utils import get_logger
    l1 = get_logger("logger1")
    assert l1.name == 'logger1'

    def a():
        return get_logger()

    l2 = a()
    assert l2.name == 'test_utils:a'

