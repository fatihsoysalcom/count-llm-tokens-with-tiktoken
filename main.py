import tiktoken

def count_tokens(text: str, model_name: str = "gpt-4") -> tuple[list[int], int]:
    """
    Counts tokens for a given text using a specified OpenAI model's encoding.
    Returns the list of token IDs and the total token count.
    """
    try:
        # Get the encoding for the specified model.
        # The article discusses LLMs and token budgets, so using an actual LLM tokenizer is key.
        encoding = tiktoken.encoding_for_model(model_name)
    except KeyError:
        print(f"Warning: Model '{model_name}' not found. Falling back to 'cl100k_base'.")
        encoding = tiktoken.get_encoding("cl100k_base") # Default for GPT-4, GPT-3.5-turbo
    
    # Encode the text into token IDs. Each ID represents a token.
    token_ids = encoding.encode(text)
    
    # The number of tokens is simply the length of the token_ids list.
    token_count = len(token_ids)
    
    return token_ids, token_count

if __name__ == "__main__":
    # Example 1: A simple Turkish sentence
    text1 = "Yapay zeka sistemlerinde token bütçesi yönetimi çok önemlidir."
    model = "gpt-4" # Using a common LLM model for tokenization example

    print(f"--- Text 1: '{text1}' ---")
    print(f"Using encoding for model: '{model}'")
    
    tokens1, count1 = count_tokens(text1, model)
    
    # Display the token IDs and count
    print(f"Token IDs (first 10): {tokens1[:10]}...")
    print(f"Total tokens: {count1}\n") # This represents the 'token budget' consumed by this text

    # Example 2: A slightly longer paragraph to demonstrate increased token usage
    text2 = """
    Modern yapay zeka (YZ) uygulamalarında karşılaşılan en kritik zorluklardan biri,
    özellikle büyük dil modelleri (LLM'ler) ile çalışırken ortaya çıkan "token bütçesi"
    olarak bilinen kaynak sınırlamalarıdır. Bu makale, token bütçesinin ne olduğunu,
    neden bu kadar önemli olduğunu ve geliştiricilerin bu kısıtlarla nasıl başa çıkarak
    uygulamalarının verimliliğini ve performansını artırabileceğini derinlemesine inceleyecektir.
    """
    
    print(f"--- Text 2 (Longer paragraph) ---")
    print(f"Using encoding for model: '{model}'")
    
    tokens2, count2 = count_tokens(text2, model)
    
    # Display the token IDs and count
    print(f"Total tokens: {count2}\n") # Demonstrates how longer text consumes more of the token budget

    # Example 3: Demonstrating how different characters or languages can affect token count
    text3 = "Hello world! Merhaba dünya! 👋"
    
    print(f"--- Text 3: '{text3}' ---")
    print(f"Using encoding for model: '{model}'")
    
    tokens3, count3 = count_tokens(text3, model)
    
    print(f"Token IDs: {tokens3}")
    print(f"Total tokens: {count3}\n")
    
    print("Note: Token counts can vary slightly between different models or tokenizers.")
    print("This example uses 'tiktoken' which is used by OpenAI models, providing a realistic estimate.")
