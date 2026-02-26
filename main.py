from arch.model import GPTConfig, GPTModel
import tiktoken

from generation import generate_text, load_hf_gpt2_weights


if __name__ == "__main__":
    model_config = GPTConfig(qkv_bias=True)
    gpt2_model = GPTModel(gpt_config=model_config)

    # Load the weights
    gpt2_model = load_hf_gpt2_weights(custom_model=gpt2_model, model_name="gpt2")

    tokenizer = tiktoken.get_encoding("gpt2")

    prompt = "Openchip hands-on workshops are awesome because"

    output = generate_text(
        model=gpt2_model,
        tokenizer=tokenizer,
        text=prompt,
        max_new_tokens=54,
        temperature=0.9,
        top_k=10,
        context_length=model_config.context_length,
    )

    print(f"\nGenerated text:\n{output}")
