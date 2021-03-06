import functools

def report(func):
    @functools.wraps(func)
    def wrapper_report(*args,**kwargs):
        if "df" in kwargs.keys():
            df = kwargs["df"]
        else:
            df = args[0]
        print(f"{df.shape[0]} rows before filter operation.")
        df = func(*args, **kwargs)
        print((f"{df.shape[0]} rows after filter operation."))
        return df

    return wrapper_report