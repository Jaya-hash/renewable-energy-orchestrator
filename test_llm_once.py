import llm

print("Provider:", llm.PROVIDER, "| model:", llm.MODELS[llm.PROVIDER])
try:
    print("SUCCESS:", llm._call_once("Reply with the single word OK.", "ping"))
except Exception as err:
    print("FAILED:", type(err).__name__)
    print(err)