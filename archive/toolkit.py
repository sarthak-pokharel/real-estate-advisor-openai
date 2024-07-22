decomma = lambda x: x.replace(",", "")
getprice = lambda x: decomma(x[1:].split(" ")[0])
gerrange = lambda x: "-".join([str(decomma(w)) for w in x.split(" ")[0].split("-")])
