import llm

print("Provider:", llm.PROVIDER, "| model:", llm.MODELS[llm.PROVIDER])
try:
    answer = llm.call_llm("You are a test. Reply with the single word OK.", "ping")
    print("SUCCESS. The LLM said:", answer)
except Exception as err:
    print("FAILED:", type(err).__name__, "-", err)