import pandas as pd
from transformers import pipeline
import sys
from unsloth import FastLanguageModel
from transformers import TextStreamer
from unsloth.chat_templates import get_chat_template
from unsloth.chat_templates import standardize_sharegpt

bm_dataset = pd.read_excel("/leonardo_work/EUHPC_R04_192/fmohamma/standardized_bm_dataset.xlsx")



list_of_labels= ['normal', 'hatespeech','hatespeech,stereotype', 'stereotype', 'toxic', 'offensive' ]
fourbit_models= ["unsloth/Qwen2.5-0.5B-Instruct",
                 "unsloth/Phi-3.5-mini-instruct",
                 "unsloth/mistral-7b",
                 "unsloth/gemma-3-1b-it-unsloth",
                 "unsloth/Meta-Llama-3.1-8B-Instruct"]

def get_model_base_name(model_name):
    """
    Extracts the base model name (e.g., 'Qwen') from a full model identifier like 'unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit'
    """
    return model_name.split('/')[-1].split('.')[0].split('-')[0]


def convert_to_conversation_format(df, 
                                   list_of_labels, 
                                   text_column='text', 
                                   language_column='language', 
                                   output_column='conversations'):
    """
    Converts raw text into a list of conversation dicts for chat-based LLMs.

    Parameters:
    - df: pandas DataFrame
    - list_of_labels: list of classification labels (e.g., ["hate", "normal", "stereotype"])
    - text_column: column with raw input text
    - language_column: column indicating language ('en' or 'it')
    - output_column: name of the new column to store conversation dictionaries

    Returns:
    - df: DataFrame with new column containing the conversation format
    """
    def build_convo(row):
        sentence = row[text_column]
        lang = row[language_column].lower()
        if lang == 'it':
            prompt = f"Assegna l'etichetta appropriata da {list_of_labels} alla seguente frase."
        else:
            prompt = f"Assign the appropriate label from {list_of_labels} to the following sentence."
        
        return [
            {"role": "system", "content": prompt},
            {"role": "user", "content": sentence}
        ]
    
    df[output_column] = df.apply(build_convo, axis=1)
    return df


# Apply conversion
bm_dataset_formatted = convert_to_conversation_format(bm_dataset, list_of_labels)


def run_unsloth_inference(
    df,
    model_name,
    text_column='conversations',
    output_path='output.xlsx',
    token=None
):
    """
    Run inference using Unsloth's FastLanguageModel on a DataFrame
    with a preformatted 'conversations' column.

    Parameters:
    - df: pandas DataFrame with 'conversations' column
    - model_name: Model identifier (e.g., 'unsloth/Meta-Llama-3.1-8B-Instruct')
    - text_column: column containing formatted conversation messages (list of dicts)
    - output_path: path to save the resulting Excel file
    - token: optional Hugging Face token

    Returns:
    - df: DataFrame with an additional column of model responses
    """
    print(f"\n🔄 Loading model: {model_name}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=8192,
        load_in_4bit=True,
        token=token
    )

    from unsloth.chat_templates import get_chat_template
    tokenizer = get_chat_template(
        tokenizer,
        chat_template="phi-4" if "phi" in model_name.lower() else "llama-3.1",
        mapping={"role": "from", "content": "value", "user": "human", "assistant": "gpt"},
    )

    FastLanguageModel.for_inference(model)

    text_streamer = TextStreamer(tokenizer)
    model_column = f"{get_model_base_name(model_name)}_answer"

    def generate_response(row):
        sys.stdout.write(f"\rRunning row {row.name}")
        sys.stdout.flush()

        conversation = row[text_column]
        try:
            inputs = tokenizer.apply_chat_template(
                conversation,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to("cuda")

            outputs = model.generate(
                input_ids=inputs,
                max_new_tokens=500,
                use_cache=True
            )

            decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Get only the model's reply after the last user prompt
            last_user_msg = conversation[-1]["content"]
            response = decoded.split(last_user_msg)[-1].strip()

            return response

        except Exception as e:
            print(f"\n⚠️ Error on row {row.name}: {e}")
            return None

    print(f"\n🧠 Running inference with {model_name}...")
    df[model_column] = df.apply(generate_response, axis=1)

    df.to_excel(output_path, index=False)
    print(f"\n✅ Results saved to {output_path}")

    return df


# Assuming df is already loaded
unsloth_phi3_answers = run_unsloth_inference(
    bm_dataset_formatted,
    model_name= fourbit_models[1],
    text_column="conversations",
    output_path="unsloth_phi3_answers.xlsx"
)
