
scenario1 = ("cpu", {"device": "cpu"})
scenario2 = ("cuda", {"device": "cuda"})

scenarios = [scenario1,
             scenario2]

def pytest_generate_tests(metafunc):
    idlist = []
    argvalues = []
    for scenario in scenarios:
        idlist.append(scenario[0])
        items = scenario[1].items()
        argnames = [x[0] for x in items]
        argvalues.append([x[1] for x in items])
    metafunc.parametrize(argnames, argvalues, ids=idlist)