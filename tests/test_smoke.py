def test_package_imports():
    import personabind

    assert isinstance(personabind.__version__, str)
    assert personabind.__version__
